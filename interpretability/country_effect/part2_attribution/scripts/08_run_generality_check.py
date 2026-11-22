"""Generality: Qwen3-4B on the BV and PR panels (PREREGISTRATION.md § 7).

All 58 Part 1 names, both orders, float32, thinking disabled (geoflip.scoring
spec "qwen3-4b"). One forward pass per prompt with residual capture gives both
the answer score and the logit lens at the read-out (last prompt position).
The Runner's scores are first checked against geoflip.scoring.score_first_token.
Needs a GPU with at least 24 GB.

  python -m country_effect.part2_attribution.scripts.08_run_generality_check --run --boundary BV
  python -m country_effect.part2_attribution.scripts.08_run_generality_check --analyse
"""

import argparse

import numpy as np
import pandas as pd

from country_effect.part2_attribution import common as C
from geoflip import scoring
from geoflip.hooks import Runner, letter_d
from geoflip.render import country_conditions, country_phrase
from geoflip.utils import read_jsonl, require, write_json

OUT = C.OUTPUTS / "08_generality"
MODEL = "qwen3-4b"


def run(boundary):
    C.require_parity()
    _, names, tasks = C.part1()
    spec = dict(scoring.MODELS[MODEL])
    tokenizer, model, revision = scoring.load_model(spec, "cuda")
    require(revision == spec["revision"], "Qwen revision differs")
    runner = Runner(model, tokenizer, spec, "cuda")
    store = C.JsonlStore(OUT / f"qwen_{boundary}.jsonl")
    checked = 0
    rows = C.panel_descriptions(boundary)
    for n, d in enumerate(rows, 1):
        for cid, display, kind in country_conditions(
            names, ("none", "real", "fictional")
        ):
            text = d["template"].replace("{COUNTRY}", country_phrase(names, display))
            for order in "AB":
                key = f"{d['description_id']}|{cid}|{order}"
                if key in store:
                    continue
                task = tasks[f"{boundary}:{order}"]
                prompt_text, ids = scoring.render_prompt(
                    tokenizer, spec, task["system_prompt"], text
                )
                letters = runner.letter_ids(prompt_text, ids)
                log_probs, states = runner.run(ids, capture=True)
                if checked < 10:  # parity with the reference scorer
                    ref = scoring.score_first_token(
                        model,
                        tokenizer,
                        spec,
                        task["system_prompt"],
                        text,
                        ("A", "B"),
                        "cuda",
                    )
                    require(
                        abs(float(log_probs[letters["A"]]) - ref["lp"]["A"]) < 1e-5,
                        "Qwen Runner differs from scorer",
                    )
                    checked += 1
                lens = letter_d(
                    runner.lens(states, len(ids) - 1), letters, task
                ).tolist()
                mass = float(
                    log_probs[letters["A"]].exp() + log_probs[letters["B"]].exp()
                )
                store.add(
                    {
                        "key": key,
                        "description_id": d["description_id"],
                        "name": cid,
                        "kind": kind,
                        "order": order,
                        "d": float(letter_d(log_probs, letters, task)),
                        "lens": lens,
                        "mass": mass,
                    }
                )
        print(f"{boundary} {n}/{len(rows)}", flush=True)
    print("done", flush=True)


def analyse():
    _, names, _ = C.part1()
    part1 = pd.read_csv(
        C.PART1 / "dataset" / "labels.csv.gz",
        usecols=["description_id", "boundary", "name", "s"],
    )
    cfg = C.panel()
    real, plac = list(names["real"]), list(names["fictional"])
    res = {
        "model": MODEL,
        "revision": scoring.MODELS[MODEL]["revision"],
        "dtype": scoring.MODELS[MODEL]["dtype"],
    }
    for b in ("BV", "PR"):
        rows = read_jsonl(OUT / f"qwen_{b}.jsonl")
        by = {}
        for r in rows:
            by.setdefault((r["description_id"], r["name"]), {})[r["order"]] = r
        require(all(len(v) == 2 for v in by.values()), "missing orders")
        s = pd.Series(
            {k: (v["A"]["d"] + v["B"]["d"]) / 2 for k, v in by.items()}
        ).unstack()
        lens = {
            k: (np.array(v["A"]["lens"]) + np.array(v["B"]["lens"])) / 2
            for k, v in by.items()
        }
        delta = s.sub(s[plac].mean(1), axis=0)
        eff = delta.mean()
        llama = part1[(part1.boundary == b) & part1.description_id.isin(s.index)].pivot(
            index="description_id", columns="name", values="s"
        )
        llama_eff = llama.sub(llama[plac].mean(1), axis=0).mean()
        groups = {}
        for g, members in cfg["boundaries"][b]["countries"].items():
            curves = []
            for did in s.index:
                pl = np.mean([lens[(did, p)] for p in plac], 0)
                curves += [lens[(did, c)] - pl for c in members]
            curve = np.mean(curves, 0)
            groups[g] = {"curve": curve.tolist(), "final": float(curve[-1])}
        masses = [r["mass"] for r in rows]
        res[b] = {
            "n_descriptions": int(len(s)),
            "sd_real": float(eff[real].std(ddof=1)),
            "sd_placebo": float(eff[plac].std(ddof=1)),
            "effects": {c: float(eff[c]) for c in real + plac},
            "r_with_llama_real": float(np.corrcoef(eff[real], llama_eff[real])[0, 1]),
            "llama_effects": {c: float(llama_eff[c]) for c in real + plac},
            "llama_sd_real_panel": float(llama_eff[real].std(ddof=1)),
            "llama_sd_placebo_panel": float(llama_eff[plac].std(ddof=1)),
            "lens_by_country_group": groups,
            "mass_p5": float(np.percentile(masses, 5)),
            "s_none_mean": float(s["none"].mean()),
        }
    write_json(OUT / "results.json", res)
    C.write_manifest(
        OUT,
        __file__,
        [C.PART1 / "dataset" / "labels.csv.gz"],
        [OUT / "results.json", OUT / "qwen_BV.jsonl", OUT / "qwen_PR.jsonl"],
    )
    for b in ("BV", "PR"):
        r = res[b]
        top = sorted(r["effects"].items(), key=lambda kv: kv[1])
        print(
            f"{b}: sd real {r['sd_real']:.3f} placebo {r['sd_placebo']:.3f}; r with Llama {r['r_with_llama_real']:.2f}; mass p5 {r['mass_p5']:.3f}"
        )
        print("   lowest", top[:4], "highest", top[-4:])


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
