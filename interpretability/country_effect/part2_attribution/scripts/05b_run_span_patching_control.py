"""Span patching with a split bridge and a matched-norm random control (follow-up to 05).

05 pools two kinds of positions in its `bridge` span: the description text
between the name and the evidence (`context`) and the chat-template tokens
after the evidence (`template`). Here they are patched separately.

Each patch is paired with a control: the same positions and layer, but the
baseline state is moved by a random direction of the same norm as the
country - baseline difference at each position. Any perturbation of a
bf16 forward pass moves d slightly; the control measures that floor, so a
span counts as carrying the country only by its excess over the control.

d is read in float32 from the final residual (Runner.d_fp32), because the
bf16 logit step (0.125 nats) is close to the effects of small spans.

Subset: the 32 panel descriptions in context c00, for the one-token High-VAC
countries (Bangladesh, Nigeria, Sudan) and the United States and United Kingdom.

  python -m country_effect.part2_attribution.scripts.05b_run_span_patching_control --run --shard 0/2
  python -m country_effect.part2_attribution.scripts.05b_run_span_patching_control --analyse
"""

import argparse
import zlib

import numpy as np

from country_effect.part2_attribution import common as C
from geoflip.spans import check_matched
from geoflip.utils import read_jsonl, require, write_json

OUT = C.OUTPUTS / "05b_span_control"
COUNTRIES = ("Bangladesh", "Nigeria", "Sudan", "United States", "United Kingdom")
GROUPS = {
    "high_vac": ("Bangladesh", "Nigeria", "Sudan"),
    "us_uk": ("United States", "United Kingdom"),
}
CONTEXT = "c00"
SPANS = ("name", "context", "ev_group", "ev_rest", "ev_all", "template", "readout")


def pairs():
    return C.panel_pairs("BV", list(COUNTRIES), [CONTEXT])


def span_positions(prompt):
    s = prompt.spans
    ev = sorted(s["ev_group"] + s["ev_rest"])
    return {
        "name": s["name"],
        "context": [p for p in s["bridge"] if p < ev[0]],
        "ev_group": s["ev_group"],
        "ev_rest": s["ev_rest"],
        "ev_all": ev,
        "template": [p for p in s["bridge"] if p > ev[-1]],
        "readout": s["readout"],
    }


def run(shard):
    import torch

    C.require_parity()
    _, names, tasks = C.part1()
    runner, _ = C.load_runner()
    index, count = shard
    store = C.JsonlStore(OUT / f"span_BV_{index}of{count}.jsonl")
    # pairs finished under an earlier sharding are not redone
    done = {r["key"] for f in OUT.glob("span_BV_*of*.jsonl") for r in read_jsonl(f)}
    todo = pairs()[index::count]
    for n, p in enumerate(todo, 1):
        for order in "AB":
            key = f"{p['d']['description_id']}|{p['country']}|{order}"
            if key in store or key in done:
                continue
            task = tasks[f"BV:{order}"]
            base = C.resolve_prompt(
                runner, p["d"], names, p["baseline"], task, p["tokens"]
            )
            target = C.resolve_prompt(
                runner, p["d"], names, p["display"], task, p["tokens"]
            )
            check_matched(base, target)
            ids = runner.letter_ids(base.text, base.ids)
            _, bs = runner.run(base.ids, capture=True)
            _, ts = runner.run(target.ids, capture=True)
            spans = span_positions(base)
            require(all(spans.values()), f"empty span in {key}")
            gen = torch.Generator(device=runner.device).manual_seed(
                C.SEED + zlib.crc32(key.encode())
            )

            def d_with(layer, pos, values):
                _, s = runner.run(base.ids, capture=True, patch={layer: (pos, values)})
                return runner.d_fp32(s, ids, task)

            real, control = {}, {}
            for span, pos in spans.items():
                layers = range(
                    runner.n_layers if span == "readout" else runner.n_layers - 1
                )
                real[span], control[span] = [], []
                for l in layers:
                    hb, ht = bs[l + 1][pos].float(), ts[l + 1][pos].float()
                    noise = torch.randn(hb.shape, generator=gen, device=hb.device)
                    noise *= (ht - hb).norm(dim=-1, keepdim=True) / noise.norm(
                        dim=-1, keepdim=True
                    )
                    real[span].append(d_with(l, pos, ts[l + 1][pos]))
                    control[span].append(
                        d_with(l, pos, (hb + noise).to(bs[l + 1].dtype))
                    )
            store.add(
                {
                    "key": key,
                    "description_id": p["d"]["description_id"],
                    "country": p["country"],
                    "baseline": p["baseline"],
                    "order": order,
                    "span_sizes": {k: len(v) for k, v in spans.items()},
                    "d_base": runner.d_fp32(bs, ids, task),
                    "d_target": runner.d_fp32(ts, ids, task),
                    "real": real,
                    "control": control,
                }
            )
        print(f"shard {index}/{count}: {n}/{len(todo)} pairs", flush=True)
    print("done", flush=True)


def analyse():
    rows = []
    for path in sorted(OUT.glob("span_BV_*of*.jsonl")):
        rows += read_jsonl(path)
    by = {}
    for r in rows:
        by.setdefault((r["description_id"], r["country"]), {})[r["order"]] = r
    require(
        len(by) == len(pairs()) and all(len(v) == 2 for v in by.values()),
        "span control incomplete",
    )
    rng = np.random.default_rng(C.SEED)
    res = {"n_pairs": len(by), "groups": {}}
    for g, members in GROUPS.items():
        sel = [o for (_, c), o in by.items() if c in members]
        base = np.array([(o["A"]["d_base"] + o["B"]["d_base"]) / 2 for o in sel])
        delta = (
            np.array([(o["A"]["d_target"] + o["B"]["d_target"]) / 2 for o in sel])
            - base
        )
        out = {"n_pairs": len(sel), "delta_mean": float(delta.mean()), "spans": {}}
        for span in SPANS:
            real = np.array([(avg_span(o, "real", span)) for o in sel]) - base[:, None]
            ctrl = (
                np.array([(avg_span(o, "control", span)) for o in sel]) - base[:, None]
            )
            boots_r, boots_x = [], []
            for _ in range(C.N_BOOT):
                i = rng.integers(0, len(sel), len(sel))
                dm = delta[i].mean()
                boots_r.append(real[i].mean(0) / dm)
                boots_x.append((real[i] - ctrl[i]).mean(0) / dm)
            out["spans"][span] = {
                "F": (real.mean(0) / delta.mean()).tolist(),
                "F_control": (ctrl.mean(0) / delta.mean()).tolist(),
                "F_excess": ((real - ctrl).mean(0) / delta.mean()).tolist(),
                "F_excess_lo": np.percentile(boots_x, 2.5, 0).tolist(),
                "F_excess_hi": np.percentile(boots_x, 97.5, 0).tolist(),
            }
        res["groups"][g] = out
    write_json(OUT / "results.json", res)
    C.write_manifest(
        OUT,
        __file__,
        [C.OUTPUTS / "01_controls" / "controls.json"],
        [OUT / "results.json"] + sorted(OUT.glob("span_BV_*.jsonl")),
    )
    for g, v in res["groups"].items():
        print(f"== {g} n={v['n_pairs']} delta={v['delta_mean']:+.3f}")
        for span, s in v["spans"].items():
            print(f"   {span:9s} F    " + " ".join(f"{x:+.2f}" for x in s["F"]))
            print(f"   {'':9s} ctrl " + " ".join(f"{x:+.2f}" for x in s["F_control"]))


def avg_span(o, kind, span):
    return (np.array(o["A"][kind][span]) + np.array(o["B"][kind][span])) / 2


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="store_true")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--analyse", action="store_true")
    args = p.parse_args(argv)
    if args.run:
        run(tuple(int(x) for x in args.shard.split("/")))
    if args.analyse:
        analyse()


if __name__ == "__main__":
    main()
