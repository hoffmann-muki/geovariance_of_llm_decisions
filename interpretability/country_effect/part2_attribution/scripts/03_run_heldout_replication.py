"""Stage 1B: held-out replication on context indices 12-19 (PREREGISTRATION.md § 4).

Builds the held-out descriptions with the Part 1 rules, scores them once with
the Part 1 names and the reference scorer, and applies the replication rule:
the correlation of BV country effects (37 real names) between the study split
(Part 1) and the held-out split exceeds 0.90.

  python -m country_effect.part2_attribution.scripts.03_run_heldout_replication --score --boundary BV
  python -m country_effect.part2_attribution.scripts.03_run_heldout_replication --analyse
"""

import argparse
import gzip

import numpy as np
import pandas as pd

from country_effect.part2_attribution import common as C
from geoflip import study as S
from geoflip.render import country_conditions, country_phrase
from geoflip.utils import require, write_csv, write_json

OUT = C.OUTPUTS / "03_heldout"
MARGIN = 0.5
R_RULE = 0.90
CHUNK = 25


def descriptions_path(boundary):
    return OUT / f"descriptions_{boundary}.csv"


def score(boundary):
    C.require_parity()
    _, names, tasks = C.part1()
    rows = C.build_descriptions(boundary, "heldout")
    for r in rows:  # every rendering differs from text_none by the country phrase only
        for _, display, _ in country_conditions(names, ("real", "fictional")):
            phrase = country_phrase(names, display)
            require(
                r["template"].replace("{COUNTRY}", phrase).replace(phrase, "", 1)
                == r["text_none"],
                r["description_id"],
            )
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(descriptions_path(boundary), list(rows[0]), rows)
    scorer = S.Scorer(C.MODEL_KEY, C.WORK / f"03_{boundary}")
    out = []
    conditions = country_conditions(names, ("none", "real", "fictional"))
    for start in range(0, len(rows), CHUNK):
        chunk = rows[start : start + CHUNK]
        keys, pairs = [], []
        for d in chunk:
            for cid, display, kind in conditions:
                text = d["template"].replace(
                    "{COUNTRY}", country_phrase(names, display)
                )
                for order in S.ORDERS:
                    keys.append((d["description_id"], cid, kind, order))
                    pairs.append((tasks[f"{boundary}:{order}"]["system_prompt"], text))
        for (did, cid, kind, order), r in zip(
            keys, scorer.score(pairs, progress_every=10**9)
        ):
            t = tasks[f"{boundary}:{order}"]
            out.append(
                {
                    "description_id": did,
                    "name": cid,
                    "kind": kind,
                    "order": order,
                    "d": r["lp"][t["letter_higher"]] - r["lp"][t["letter_lower"]],
                    "mass": r["mass"],
                    "top1": r["top"][0][0],
                }
            )
        print(
            f"{boundary} descriptions {start + len(chunk)}/{len(rows)} rate {scorer.last_rate or 0:.1f}/s",
            flush=True,
        )
    df = pd.DataFrame(out)
    wide = df.pivot_table(
        index=["description_id", "name", "kind"], columns="order", values="d"
    ).reset_index()
    wide.columns = ["description_id", "name", "kind", "d_A", "d_B"]
    mass = (
        df.groupby(["description_id", "name"])
        .mass.min()
        .rename("mass_min")
        .reset_index()
    )
    wide = wide.merge(mass, on=["description_id", "name"])
    wide["s"] = (wide.d_A + wide.d_B) / 2
    with gzip.open(OUT / f"scores_{boundary}.csv.gz", "wt", encoding="utf-8") as f:
        wide.to_csv(f, index=False, float_format="%.4f")


def effects(piv, real, plac):
    delta = piv.sub(piv[plac].mean(1), axis=0)
    return delta, delta.mean()


def flips(piv, cols):
    lab = np.where(
        piv[cols].values > MARGIN, 1, np.where(piv[cols].values < -MARGIN, -1, 0)
    )
    hi, lo = (lab == 1), (lab == -1)
    flip = hi.any(1) & lo.any(1)
    return flip, hi, lo


def analyse():
    _, names, _ = C.part1()
    part1 = pd.read_csv(
        C.PART1 / "dataset" / "labels.csv.gz",
        usecols=["description_id", "boundary", "name", "kind", "s"],
    )
    rng = np.random.default_rng(C.SEED)
    res = {}
    for b in ("BV", "PR"):
        h = pd.read_csv(OUT / f"scores_{b}.csv.gz")
        desc = pd.read_csv(descriptions_path(b))
        require(len(h) == len(desc) * 58, f"{b}: incomplete held-out scores")
        real = [c for c in names["real"]]
        plac = [c for c in names["fictional"]]
        ph = h.pivot(index="description_id", columns="name", values="s")
        p1 = part1[part1.boundary == b].pivot(
            index="description_id", columns="name", values="s"
        )
        dh, eh = effects(ph, real, plac)
        d1, e1 = effects(p1, real, plac)
        r_real = float(np.corrcoef(e1[real], eh[real])[0, 1])
        n = len(dh)
        boots = []
        for _ in range(C.N_BOOT):
            idx = rng.integers(0, n, n)
            boots.append(np.corrcoef(e1[real], dh.iloc[idx][real].mean())[0, 1])
        flip, hi, lo = flips(ph, real)
        us = real.index("United States")
        res[b] = {
            "n_descriptions": n,
            "r_real_study_vs_heldout": r_real,
            "r_real_ci": [
                float(np.percentile(boots, 2.5)),
                float(np.percentile(boots, 97.5)),
            ],
            "r_placebo_study_vs_heldout": float(np.corrcoef(e1[plac], eh[plac])[0, 1]),
            "sd_real_heldout": float(eh[real].std(ddof=1)),
            "sd_placebo_heldout": float(eh[plac].std(ddof=1)),
            "sd_real_study": float(e1[real].std(ddof=1)),
            "sd_placebo_study": float(e1[plac].std(ddof=1)),
            "effects_heldout": {c: float(eh[c]) for c in real + plac},
            "effects_study": {c: float(e1[c]) for c in real + plac},
            "flip_rate_real": float(flip.mean()),
            "flip_rate_placebo": float(flips(ph, plac)[0].mean()),
            "us_on_higher_side_of_flips": int((flip & hi[:, us]).sum()),
            "n_flips": int(flip.sum()),
            "mass_min": float(h.mass_min.min()),
            "none_minus_placebo": float((ph["none"] - ph[plac].mean(1)).mean()),
        }
    res["verdict"] = {
        "rule": f"BV r > {R_RULE}",
        "r": res["BV"]["r_real_study_vs_heldout"],
        "H1_heldout_prediction_met": res["BV"]["r_real_study_vs_heldout"] > R_RULE,
    }
    write_json(OUT / "results.json", res)
    files = [
        OUT / f
        for f in (
            "results.json",
            "scores_BV.csv.gz",
            "scores_PR.csv.gz",
            "descriptions_BV.csv",
            "descriptions_PR.csv",
        )
    ]
    C.write_manifest(
        OUT,
        __file__,
        [C.PART1 / "dataset" / "labels.csv.gz"],
        files,
        {"verdict": res["verdict"]},
    )
    for b in ("BV", "PR"):
        r = res[b]
        print(
            f"{b}: r={r['r_real_study_vs_heldout']:.3f} {r['r_real_ci']} sd real {r['sd_real_heldout']:.3f} (study {r['sd_real_study']:.3f}) "
            f"placebo {r['sd_placebo_heldout']:.3f}; flips real {r['flip_rate_real']:.3f} placebo {r['flip_rate_placebo']:.3f}; US {r['us_on_higher_side_of_flips']}/{r['n_flips']}"
        )
    print(res["verdict"])


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--score", action="store_true")
    p.add_argument("--boundary", choices=["BV", "PR"])
    p.add_argument("--analyse", action="store_true")
    args = p.parse_args(argv)
    if args.score:
        require(args.boundary, "--score needs --boundary")
        score(args.boundary)
    if args.analyse:
        analyse()


if __name__ == "__main__":
    main()
