"""Stage 2: RMSNorm-calibrated logit lens at the read-out (PREREGISTRATION.md § 6).

For every panel description, the panel countries, the matched baselines and
the no-country prompt are run once per order with all residual states
captured; d is read from every layer through model.model.norm and lm_head.
Delta^(l) = s^(l)(country) - s^(l)(baseline) on the token-matched pairs.

  python -m country_effect.part2_attribution.scripts.04_run_logit_lens --run --boundary BV
  python -m country_effect.part2_attribution.scripts.04_run_logit_lens --analyse
"""

import argparse

import numpy as np

from country_effect.part2_attribution import common as C
from geoflip.hooks import letter_d
from geoflip.utils import read_jsonl, require, write_json

OUT = C.OUTPUTS / "04_logit_lens"
# The lens runs norm and lm_head on all layers at once; that bf16 matmul can
# round a logit one step (0.125 at logit magnitude 16-32) away from the
# single-position forward pass.
BF16_LOGIT_STEP = 0.125


def run(boundary):
    C.require_parity()
    _, names, tasks = C.part1()
    runner, _ = C.load_runner()
    store = C.JsonlStore(OUT / f"lens_{boundary}.jsonl")
    pairs = C.panel_pairs(boundary)
    jobs = {}
    for p in pairs:
        jobs[(p["d"]["description_id"], p["display"])] = (
            p["d"],
            p["display"],
            p["tokens"],
        )
        jobs[(p["d"]["description_id"], p["baseline"])] = (
            p["d"],
            p["baseline"],
            p["tokens"],
        )
        jobs[(p["d"]["description_id"], None)] = (p["d"], None, None)
    for n, ((did, display), (d, disp, tokens)) in enumerate(jobs.items(), 1):
        for order in "AB":
            key = f"{did}|{disp}|{order}"
            if key in store:
                continue
            task = tasks[f"{boundary}:{order}"]
            prompt = C.resolve_prompt(runner, d, names, disp, task, tokens)
            log_probs, states = runner.run(prompt.ids, capture=True)
            ids = runner.letter_ids(prompt.text, prompt.ids)
            lens = letter_d(runner.lens(states, prompt.readout), ids, task).tolist()
            store.add(
                {
                    "key": key,
                    "description_id": did,
                    "name": disp,
                    "order": order,
                    "d": float(letter_d(log_probs, ids, task)),
                    "lens": lens,
                }
            )
        if n % 100 == 0:
            print(f"{boundary} {n}/{len(jobs)} prompts", flush=True)
    print("done", boundary, flush=True)


def final_layer_check(rows):
    """The last lens layer must reproduce the model output up to one bf16 step."""
    diff = np.abs([r["lens"][-1] - r["d"] for r in rows])
    require(
        diff.max() <= BF16_LOGIT_STEP + 1e-6, "final lens layer differs from the output"
    )
    return {
        "n_prompts": len(rows),
        "share_differing": float((diff > 1e-3).mean()),
        "max_abs_diff": float(diff.max()),
    }


def load(boundary):
    rows = read_jsonl(OUT / f"lens_{boundary}.jsonl")
    s = {}
    for r in rows:
        s.setdefault((r["description_id"], r["name"]), {})[r["order"]] = r
    out = {}
    for key, by in s.items():
        require(set(by) == {"A", "B"}, f"missing order for {key}")
        lens = (np.array(by["A"]["lens"]) + np.array(by["B"]["lens"])) / 2
        out[key] = {"lens": lens, "s": (by["A"]["d"] + by["B"]["d"]) / 2}
    return out


def half_layer(curve):
    """First layer at which the curve reaches half its final value (same sign)."""
    final = curve[-1]
    if final == 0:
        return None
    hits = np.nonzero(np.sign(final) * curve >= abs(final) / 2)[0]
    return int(hits[0])


def analyse():
    res = {}
    for b in ("BV", "PR"):
        lens = load(b)
        pairs = C.panel_pairs(b)
        rows = []
        for p in pairs:
            did = p["d"]["description_id"]
            c, base = lens[(did, p["display"])], lens[(did, p["baseline"])]
            # the final lens layer must equal the actual output score
            rows.append(
                {
                    "country": p["country"],
                    "country_group": p["country_group"],
                    "item": p["d"]["item"],
                    "item_group": p["item_group"],
                    "delta_lens": c["lens"] - base["lens"],
                    "delta": c["s"] - base["s"],
                }
            )
        out = {
            "final_layer_check": final_layer_check(read_jsonl(OUT / f"lens_{b}.jsonl")),
            "by_country": {},
            "by_country_group": {},
            "by_item_group": {},
        }
        for key, field in (
            ("by_country", "country"),
            ("by_country_group", "country_group"),
        ):
            for g in sorted({r[field] for r in rows}):
                sel = [r for r in rows if r[field] == g]
                curve = np.mean([r["delta_lens"] for r in sel], 0)
                out[key][g] = {
                    "n": len(sel),
                    "curve": curve.tolist(),
                    "final": float(curve[-1]),
                    "half_layer": half_layer(curve),
                }
        for cg in sorted({r["country_group"] for r in rows}):
            for ig in sorted({r["item_group"] for r in rows}):
                sel = [
                    r
                    for r in rows
                    if r["country_group"] == cg and r["item_group"] == ig
                ]
                curve = np.mean([r["delta_lens"] for r in sel], 0)
                out["by_item_group"][f"{cg}|{ig}"] = {
                    "n": len(sel),
                    "curve": curve.tolist(),
                    "final": float(curve[-1]),
                    "half_layer": half_layer(curve),
                }
        # none and baselines relative to each other: how much the baseline itself moves s
        res[b] = out
    write_json(OUT / "results.json", res)
    C.write_manifest(
        OUT,
        __file__,
        [C.OUTPUTS / "01_controls" / "controls.json"],
        [OUT / "results.json", OUT / "lens_BV.jsonl", OUT / "lens_PR.jsonl"],
    )
    for b in ("BV", "PR"):
        print("==", b)
        for g, v in res[b]["by_country"].items():
            print(
                f"  {g:16s} final {v['final']:+.3f} half-layer {v['half_layer']}  "
                + " ".join(f"{x:+.2f}" for x in v["curve"])
            )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="store_true")
    p.add_argument("--boundary", choices=["BV", "PR"])
    p.add_argument("--analyse", action="store_true")
    args = p.parse_args(argv)
    if args.run:
        require(args.boundary, "--run needs --boundary")
        run(args.boundary)
    if args.analyse:
        analyse()


if __name__ == "__main__":
    main()
