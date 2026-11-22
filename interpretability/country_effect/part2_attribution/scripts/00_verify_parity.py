"""Stage 0: parity gate for every later Part 2 script (PREREGISTRATION.md § 2).

1. the invariant test suite passes (on the GPU, so model tests run);
2. 100 study-split prompts reproduce the Part 1 d_A, d_B;
3. capture hooks and identity patches leave logits unchanged, and
   inputs_embeds reproduces input_ids logits;
4. one prompt is written out step by step (walkthrough.txt);
5. timings of one forward pass and one IG pair, for planning.

  cd interpretability && python -m country_effect.part2_attribution.scripts.00_verify_parity
"""

import random
import subprocess
import sys
import time

from country_effect.part2_attribution import common as C
from geoflip.hooks import letter_d, span_patch
from geoflip.utils import require, write_json

OUT = C.OUTPUTS / "00_parity"
TOLERANCE = 5e-5  # Part 1 stores d to 4 decimals
N_PAIRS = 50
WALK_ID = "BV|security_guard|f1|W1|BV_c00"


def run_tests():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "geoflip/tests/test_part2_invariants.py",
        ],
        cwd=C.ROOT,
        capture_output=True,
        text=True,
    )
    tail = result.stdout.strip().splitlines()[-1] if result.stdout else result.stderr
    return result.returncode == 0, tail


def sample_pairs(descriptions, names):
    rng = random.Random(C.SEED)
    all_names = ["none"] + list(names["real"]) + list(names["fictional"])
    out = []
    for b in ("BV", "PR"):
        rows = [d for d in descriptions if d["boundary"] == b]
        for d in rng.sample(rows, N_PAIRS // 2):
            out.append((d, rng.choice(all_names)))
    return out


def d_of(runner, p, task, **kwargs):
    log_probs, states = runner.run(p.ids, **kwargs)
    return (
        float(letter_d(log_probs, runner.letter_ids(p.text, p.ids), task)),
        log_probs,
        states,
    )


def walkthrough(runner, descriptions, names, tasks):
    d = next(x for x in descriptions if x["description_id"] == WALK_ID)
    lines = [f"Description {WALK_ID}", f"Evidence: {d['evidence_sentence']}", ""]
    for display in (
        None,
        "the United States",
        "the Central Region",
        "Bangladesh",
        "Atlantis",
    ):
        for order in "AB":
            task = tasks[f"BV:{order}"]
            n = (
                None
                if display is None
                else len(
                    runner.tokenizer.encode(" " + display, add_special_tokens=False)
                )
            )
            p = C.resolve_prompt(runner, d, names, display, task, n)
            if display is None and order == "A":
                lines += [
                    "--- full rendered prompt (order A, no country) ---",
                    p.text,
                    "",
                ]
            dval, log_probs, states = d_of(runner, p, task, capture=True)
            lens = runner.lens(states, p.readout)
            ids = runner.letter_ids(p.text, p.ids)
            lens_d = letter_d(lens, ids, task).tolist()
            lines.append(
                f"=== name={display!r} order={order} ({task['letter_higher']} = Battle) ==="
            )
            for key in ("name", "ev_group", "ev_rest", "ev_end", "bridge", "readout"):
                span = p.spans[key]
                text = (
                    "None"
                    if not span
                    else repr(runner.tokenizer.decode([p.ids[i] for i in span]))
                )
                lines.append(f"  {key:9s} {str(p.pieces.get(key, '')):28s} {text}")
            lines.append(
                f"  lpA={float(log_probs[ids['A']]):+.4f} lpB={float(log_probs[ids['B']]):+.4f} d={dval:+.4f}"
            )
            lines.append(
                "  logit-lens d by layer: " + " ".join(f"{v:+.2f}" for v in lens_d)
            )
            lines.append("")
    (OUT / "walkthrough.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    import torch

    from geoflip.attribution import layer_ig

    OUT.mkdir(parents=True, exist_ok=True)
    checks = {}
    tests_ok, tests_tail = run_tests()
    checks["pytest"] = {"passed": tests_ok, "summary": tests_tail}
    print("pytest:", tests_tail, flush=True)

    descriptions, names, tasks = C.part1()
    runner, revision = C.load_runner()
    ref = C.part1_scores()

    worst, rows = 0.0, []
    for d, cid in sample_pairs(descriptions, names):
        display = None if cid == "none" else C.display_name(names, cid)
        for order in "AB":
            task = tasks[f"{d['boundary']}:{order}"]
            p = C.resolve_prompt(
                runner,
                d,
                names,
                display,
                task,
                None if display is None else names["tokens"][display],
            )
            dval, _, _ = d_of(runner, p, task)
            stored = ref[(d["description_id"], cid)][0 if order == "A" else 1]
            worst = max(worst, abs(dval - stored))
            rows.append(
                {
                    "description_id": d["description_id"],
                    "name": cid,
                    "order": order,
                    "d": dval,
                    "part1": stored,
                }
            )
    checks["part1_reproduction"] = {
        "passed": worst <= TOLERANCE,
        "n_prompts": len(rows),
        "max_abs_diff": worst,
    }
    write_json(OUT / "reproduction.json", rows)
    print("reproduction max |diff|:", worst, flush=True)

    transparent = True
    for d, cid in sample_pairs(descriptions, names)[:6]:
        display = None if cid == "none" else C.display_name(names, cid)
        task = tasks[f"{d['boundary']}:A"]
        p = C.resolve_prompt(runner, d, names, display, task)
        plain, _ = runner.run(p.ids)
        captured, states = runner.run(p.ids, capture=True)
        transparent &= torch.equal(plain, captured)
        for layer in (-1, 0, 9, 18, 27):
            for key in ("readout", "ev_group", "bridge"):
                patched, _ = runner.run(
                    p.ids, patch=span_patch(states, layer, p.spans[key])
                )
                transparent &= torch.equal(plain, patched)
        with torch.inference_mode():
            embeds = runner.model.model.embed_tokens(
                torch.tensor([p.ids], device=runner.device)
            )
        by_embeds, _ = runner.run(p.ids, embeds=embeds)
        transparent &= torch.equal(plain, by_embeds)
    checks["hooks_and_embeds"] = {"passed": bool(transparent)}
    print("hooks/embeds transparent:", transparent, flush=True)

    walkthrough(runner, descriptions, names, tasks)

    # Timing and an IG completeness check on one matched pair.
    d = next(x for x in descriptions if x["description_id"] == WALK_ID)
    task = tasks["BV:A"]
    base = C.resolve_prompt(runner, d, names, "Atlantis", task, 1)
    target = C.resolve_prompt(runner, d, names, "Bangladesh", task, 1)
    torch.cuda.synchronize()
    t0 = time.monotonic()
    for _ in range(20):
        runner.run(base.ids)
    torch.cuda.synchronize()
    forward_s = (time.monotonic() - t0) / 20
    runner.model.requires_grad_(False)
    t0 = time.monotonic()
    ig = layer_ig(
        runner.model,
        base.ids,
        target.ids,
        runner.letter_ids(base.text, base.ids),
        task,
        steps=50,
    )
    ig_s = time.monotonic() - t0
    d_base, _, _ = d_of(runner, base, task)
    d_target, _, _ = d_of(runner, target, task)
    ig_ok = abs(ig["d_base"] - d_base) < 1e-6 and abs(ig["d_target"] - d_target) < 1e-6
    checks["ig_endpoints"] = {
        "passed": ig_ok,
        "d_base": d_base,
        "d_target": d_target,
        "completeness_error": ig["completeness_error"],
        "embedding_row_sum": float(ig["attr"][0].sum()),
    }
    print("IG:", checks["ig_endpoints"], flush=True)

    all_passed = all(c["passed"] for c in checks.values())
    C.write_manifest(
        OUT,
        __file__,
        [C.PART1 / "dataset" / "labels.csv.gz", C.PART1 / "inputs" / "prompts.json"],
        [OUT / "reproduction.json", OUT / "walkthrough.txt"],
        {
            "all_passed": all_passed,
            "checks": checks,
            "model_revision": revision,
            "dtype": runner.spec["dtype"],
            "timing_seconds": {"forward": forward_s, "ig_pair_50_steps": ig_s},
        },
    )
    print(
        "all passed:",
        all_passed,
        "forward",
        round(forward_s, 4),
        "s; IG pair",
        round(ig_s, 1),
        "s",
    )
    require(all_passed, "Parity checks failed; see outputs/00_parity/manifest.json")


if __name__ == "__main__":
    main()
