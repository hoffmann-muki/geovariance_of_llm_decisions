"""Stage 2: token-length-matched activation patching (PREREGISTRATION.md § 6).

Denoising: run the baseline prompt and replace the residual output of layer l
at span S with the country prompt's. Pairs have equal length and identical
tokens outside the name span (checked). Spans name, ev_group, ev_rest, ev_all
and bridge at l = 0..26 (a non-final position can reach the read-out only
through later layers), readout at l = 0..27.

  python -m country_effect.part2_attribution.scripts.05_run_activation_patching --run --boundary BV --shard 0/2
  python -m country_effect.part2_attribution.scripts.05_run_activation_patching --analyse
"""

import argparse

import numpy as np

from country_effect.part2_attribution import common as C
from geoflip.hooks import letter_d, span_patch
from geoflip.spans import check_matched
from geoflip.utils import read_jsonl, require, write_json

OUT = C.OUTPUTS / "05_patching"
SPANS = ("name", "ev_group", "ev_rest", "ev_all", "bridge", "readout")
LATE = (18, 26)
H1_READOUT, H1_EVIDENCE, H2_GROUP, H2_NULL = 0.5, 0.2, 0.3, 0.1


def positions(prompt, span):
    if span == "ev_all":
        return sorted(prompt.spans["ev_group"] + prompt.spans["ev_rest"])
    return prompt.spans[span]


def run(boundary, shard):
    C.require_parity()
    _, names, tasks = C.part1()
    runner, _ = C.load_runner()
    index, count = shard
    store = C.JsonlStore(OUT / f"patch_{boundary}_{index}of{count}.jsonl")
    pairs = C.panel_pairs(boundary)[index::count]
    n_layers = runner.n_layers
    for n, p in enumerate(pairs, 1):
        for order in "AB":
            key = f"{p['d']['description_id']}|{p['country']}|{order}"
            if key in store:
                continue
            task = tasks[f"{boundary}:{order}"]
            base = C.resolve_prompt(
                runner, p["d"], names, p["baseline"], task, p["tokens"]
            )
            target = C.resolve_prompt(
                runner, p["d"], names, p["display"], task, p["tokens"]
            )
            check_matched(base, target)
            ids = runner.letter_ids(base.text, base.ids)
            lp_base, _ = runner.run(base.ids)
            lp_target, states = runner.run(target.ids, capture=True)
            patched = {}
            for span in SPANS:
                pos = positions(base, span)
                layers = range(n_layers if span == "readout" else n_layers - 1)
                patched[span] = [
                    float(
                        letter_d(
                            runner.run(base.ids, patch=span_patch(states, l, pos))[0],
                            ids,
                            task,
                        )
                    )
                    for l in layers
                ]
            store.add(
                {
                    "key": key,
                    "description_id": p["d"]["description_id"],
                    "country": p["country"],
                    "baseline": p["baseline"],
                    "order": order,
                    "d_base": float(letter_d(lp_base, ids, task)),
                    "d_target": float(letter_d(lp_target, ids, task)),
                    "patched": patched,
                }
            )
        if n % 10 == 0:
            print(
                f"{boundary} shard {index}/{count}: {n}/{len(pairs)} pairs", flush=True
            )
    print("done", flush=True)


def load(boundary):
    rows = []
    for path in sorted(OUT.glob(f"patch_{boundary}_*of*.jsonl")):
        rows += read_jsonl(path)
    by = {}
    for r in rows:
        by.setdefault((r["description_id"], r["country"]), {})[r["order"]] = r
    out = []
    for (did, cid), o in by.items():
        require(set(o) == {"A", "B"}, f"missing order for {did} {cid}")
        avg = lambda f: (np.array(f(o["A"])) + np.array(f(o["B"]))) / 2
        out.append(
            {
                "description_id": did,
                "country": cid,
                "item": did.split("|")[1],
                "s_base": float(avg(lambda r: r["d_base"])),
                "s_target": float(avg(lambda r: r["d_target"])),
                "patched": {s: avg(lambda r, s=s: r["patched"][s]) for s in SPANS},
            }
        )
    return out


def fractions(sel, rng):
    """F_S(l) = mean(s_patched - s_base) / mean(s_target - s_base), with a
    bootstrap interval over pairs."""
    total = np.array([r["s_target"] - r["s_base"] for r in sel])
    out = {"n": len(sel), "delta_mean": float(total.mean())}
    if abs(total.mean()) < 1e-6:  # no effect to transfer: fractions undefined
        return {**out, **{span: None for span in SPANS}}
    for span in SPANS:
        eff = np.array([r["patched"][span] - r["s_base"] for r in sel])
        f = eff.mean(0) / total.mean()
        boots = []
        for _ in range(C.N_BOOT // 4):
            idx = rng.integers(0, len(sel), len(sel))
            with np.errstate(divide="ignore", invalid="ignore"):
                boots.append(eff[idx].mean(0) / total[idx].mean())
        boots = np.where(np.isfinite(boots), boots, np.nan)
        out[span] = {
            "F": f.tolist(),
            "lo": np.nanpercentile(boots, 2.5, 0).tolist(),
            "hi": np.nanpercentile(boots, 97.5, 0).tolist(),
            "effect_nats": eff.mean(0).tolist(),
        }
    return out


def analyse():
    rng = np.random.default_rng(C.SEED)
    res = {}
    for b in ("BV", "PR"):
        rows = load(b)
        require(
            len(rows) == len(C.panel_pairs(b)),
            f"{b}: patching incomplete ({len(rows)})",
        )
        for r in rows:
            r["country_group"] = C.country_group(b, r["country"])
            r["item_group"] = C.item_group(b, r["item"])
        out = {
            "by_country_group": {},
            "by_country": {},
            "by_group_item_group": {},
            "by_group_item": {},
        }
        for g in sorted({r["country_group"] for r in rows}):
            sel = [r for r in rows if r["country_group"] == g]
            out["by_country_group"][g] = fractions(sel, rng)
            for ig in sorted({r["item_group"] for r in rows}):
                out["by_group_item_group"][f"{g}|{ig}"] = fractions(
                    [r for r in sel if r["item_group"] == ig], rng
                )
            for item in sorted({r["item"] for r in rows}):
                part = [r for r in sel if r["item"] == item]
                out["by_group_item"][f"{g}|{item}"] = {
                    "n": len(part),
                    "delta_mean": float(
                        np.mean([r["s_target"] - r["s_base"] for r in part])
                    ),
                }
        for c in sorted({r["country"] for r in rows}):
            out["by_country"][c] = fractions(
                [r for r in rows if r["country"] == c], rng
            )
        out["rules"] = rules(b, out)
        res[b] = out
    write_json(OUT / "results.json", res)
    files = [OUT / "results.json"] + sorted(OUT.glob("patch_*.jsonl"))
    C.write_manifest(
        OUT,
        __file__,
        [C.OUTPUTS / "01_controls" / "controls.json"],
        files,
        {"rules": {b: res[b]["rules"] for b in res}},
    )
    for b in res:
        print("==", b)
        for g, v in res[b]["by_country_group"].items():
            print(f"  {g} n={v['n']} delta={v['delta_mean']:+.3f}")
            for span in SPANS if v["readout"] is not None else ():
                print(f"    {span:9s} " + " ".join(f"{x:+.2f}" for x in v[span]["F"]))
        print("  rules:", res[b]["rules"])


def rules(boundary, out):
    lo, hi = LATE
    verdict = {}
    for g, v in out["by_country_group"].items():
        if v["readout"] is None:
            verdict[g] = None
            continue
        readout_late = float(np.mean(v["readout"]["F"][lo : hi + 1]))
        ev_all_max = float(max(v["ev_all"]["F"]))
        group_F = np.array(v["ev_group"]["F"])
        l_star = int(group_F.argmax())
        group_lo = v["ev_group"]["lo"][l_star]
        items = out["by_group_item"]
        a = items.get(f"{g}|anchor_lo", {}).get("delta_mean")
        sg = items.get(f"{g}|security_guard", {}).get("delta_mean")
        anchor_ok = (
            None
            if a is None or sg is None
            else bool(np.sign(a) == np.sign(sg) and abs(a) >= abs(sg))
        )
        F_of = lambda k: (
            out["by_group_item_group"].get(f"{g}|{k}", {}).get("ev_group") or {}
        ).get("F")
        under = F_of("underspecified")
        clear = [F_of("clear_lower"), F_of("clear_higher")]
        under_gt_clear = None
        if under and all(clear):
            under_gt_clear = bool(under[l_star] > max(c[l_star] for c in clear))
        readout_never = all(x < H1_READOUT for x in v["readout"]["F"][: hi + 1])
        verdict[g] = {
            "readout_F_late_mean": readout_late,
            "ev_all_F_max": ev_all_max,
            "ev_group_F_max": float(group_F.max()),
            "ev_group_F_max_layer": l_star,
            "ev_group_F_max_ci_lo": group_lo,
            "anchor_lo_ge_security_guard": anchor_ok,
            "underspecified_ev_group_gt_clear": under_gt_clear,
            "H1_supported": bool(
                readout_late >= H1_READOUT and ev_all_max < H1_EVIDENCE and anchor_ok
            ),
            "H1_falsified_patching": bool(readout_never and ev_all_max >= H1_READOUT),
            "H2_supported_patching": bool(
                group_F.max() >= H2_GROUP and group_lo > 0 and under_gt_clear
            ),
            "H2_evidence_null": bool(ev_all_max < H2_NULL),
        }
    return verdict


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="store_true")
    p.add_argument("--boundary", choices=["BV", "PR"])
    p.add_argument("--shard", default="0/1")
    p.add_argument("--analyse", action="store_true")
    args = p.parse_args(argv)
    if args.run:
        require(args.boundary, "--run needs --boundary")
        i, n = (int(x) for x in args.shard.split("/"))
        run(args.boundary, (i, n))
    if args.analyse:
        analyse()


if __name__ == "__main__":
    main()
