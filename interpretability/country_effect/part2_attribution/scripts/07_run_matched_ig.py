"""Stage 2: token-matched integrated gradients (PREREGISTRATION.md § 6).

One-token panel countries against their one-token baseline on the IG contexts.
For each order, layer-wise IG (geoflip.attribution.layer_ig, 50 midpoint
steps) attributes d(country) - d(baseline) to every residual position at every
layer; attributions are summed within spans and averaged over orders.

  python -m country_effect.part2_attribution.scripts.07_run_matched_ig --run --boundary BV
  python -m country_effect.part2_attribution.scripts.07_run_matched_ig --analyse
"""

import argparse

import numpy as np

from country_effect.part2_attribution import common as C
from geoflip.attribution import GRAD_MODE, layer_ig
from geoflip.spans import check_matched
from geoflip.utils import read_jsonl, require, write_json

OUT = C.OUTPUTS / "07_ig"
SPANS = ("name", "ev_group", "ev_rest", "bridge", "readout")


def pairs_for(boundary):
    cfg = C.panel()
    return C.panel_pairs(
        boundary, cfg["boundaries"][boundary]["ig_countries"], cfg["ig_contexts"]
    )


def run(boundary, shard):
    C.require_parity()
    _, names, tasks = C.part1()
    runner, _ = C.load_runner()
    runner.model.requires_grad_(False)
    steps = C.panel()["ig"]["steps"]
    index, count = shard
    store = C.JsonlStore(OUT / f"ig_{boundary}_{index}of{count}.jsonl")
    pairs = pairs_for(boundary)[index::count]
    for n, p in enumerate(pairs, 1):
        for order in "AB":
            key = f"{p['d']['description_id']}|{p['country']}|{order}"
            if key in store:
                continue
            task = tasks[f"{boundary}:{order}"]
            base = C.resolve_prompt(runner, p["d"], names, p["baseline"], task, 1)
            target = C.resolve_prompt(runner, p["d"], names, p["display"], task, 1)
            check_matched(base, target)
            r = layer_ig(
                runner.model,
                base.ids,
                target.ids,
                runner.letter_ids(base.text, base.ids),
                task,
                steps,
            )
            attr = r["attr"].numpy()
            sums = {s: attr[:, base.spans[s]].sum(1).tolist() for s in SPANS}
            covered = [i for s in SPANS for i in base.spans[s]]
            other = np.delete(attr, covered, axis=1).sum(1).tolist()
            store.add(
                {
                    "key": key,
                    "description_id": p["d"]["description_id"],
                    "country": p["country"],
                    "baseline": p["baseline"],
                    "order": order,
                    "d_base": r["d_base"],
                    "d_target": r["d_target"],
                    "completeness_error": r["completeness_error"],
                    "span_sums": sums,
                    "other": other,
                    "total": attr.sum(1).tolist(),
                }
            )
        if n % 10 == 0:
            print(f"{boundary} {n}/{len(pairs)}", flush=True)
    print("done", flush=True)


def analyse():
    res = {"grad_mode": GRAD_MODE}
    for b in ("BV", "PR"):
        rows = []
        for path in sorted(OUT.glob(f"ig_{b}_*of*.jsonl")):
            rows += read_jsonl(path)
        by = {}
        for r in rows:
            by.setdefault((r["description_id"], r["country"]), {})[r["order"]] = r
        require(
            len(by) == len(pairs_for(b)) and all(len(v) == 2 for v in by.values()),
            f"{b}: IG incomplete",
        )
        groups = {}
        for (did, cid), o in by.items():
            g = C.country_group(b, cid)
            avg = {
                s: (np.array(o["A"]["span_sums"][s]) + np.array(o["B"]["span_sums"][s]))
                / 2
                for s in SPANS
            }
            avg["other"] = (np.array(o["A"]["other"]) + np.array(o["B"]["other"])) / 2
            total = (np.array(o["A"]["total"]) + np.array(o["B"]["total"])) / 2
            delta = (
                (o["A"]["d_target"] - o["A"]["d_base"])
                + (o["B"]["d_target"] - o["B"]["d_base"])
            ) / 2
            err = max(o["A"]["completeness_error"], o["B"]["completeness_error"])
            groups.setdefault(g, []).append(
                {
                    "avg": avg,
                    "total": total,
                    "delta": delta,
                    "err": err,
                    "item": did.split("|")[1],
                }
            )
        out = {}
        for g, sel in groups.items():
            total = np.sum([r["total"] for r in sel], 0)
            shares = {
                s: (np.sum([r["avg"][s] for r in sel], 0) / total).tolist()
                for s in list(SPANS) + ["other"]
            }
            out[g] = {
                "n_pairs": len(sel),
                "delta_mean": float(np.mean([r["delta"] for r in sel])),
                "layer_total_mean": (total / len(sel)).tolist(),
                "shares": shares,
                "completeness_error_median": float(np.median([r["err"] for r in sel])),
                "completeness_error_max": float(np.max([r["err"] for r in sel])),
            }
        res[b] = out
    write_json(OUT / "results.json", res)
    C.write_manifest(
        OUT,
        __file__,
        [C.OUTPUTS / "01_controls" / "controls.json"],
        [OUT / "results.json"] + sorted(OUT.glob("ig_*.jsonl")),
        {"grad_mode": GRAD_MODE, "steps": C.panel()["ig"]["steps"]},
    )
    for b in ("BV", "PR"):
        for g, v in res[b].items():
            print(
                f"== {b} {g} n={v['n_pairs']} delta={v['delta_mean']:+.3f} err median {v['completeness_error_median']:.3f}"
            )
            for s, sh in v["shares"].items():
                print(f"   {s:9s} " + " ".join(f"{x:+.2f}" for x in sh))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="store_true")
    p.add_argument("--boundary", choices=["BV", "PR"])
    p.add_argument("--shard", default="0/1")
    p.add_argument("--analyse", action="store_true")
    args = p.parse_args(argv)
    if args.run:
        require(args.boundary, "--run needs --boundary")
        run(args.boundary, tuple(int(x) for x in args.shard.split("/")))
    if args.analyse:
        analyse()


if __name__ == "__main__":
    main()
