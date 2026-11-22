"""Stage 1A: score the control names on the study split and apply the § 3 rules.

Scoring uses the Part 1 reference scorer (geoflip.study.Scorer: one prompt per
forward pass, JSONL cache), so control scores are directly comparable with the
Part 1 placebo and real-country scores they are referenced to.

  python -m country_effect.part2_attribution.scripts.02_run_behavioral_controls --score --boundary BV
  python -m country_effect.part2_attribution.scripts.02_run_behavioral_controls --analyse
"""

import argparse
import gzip

import numpy as np
import pandas as pd

from country_effect.part2_attribution import common as C
from geoflip import study as S
from geoflip.utils import require, write_json

OUT = C.OUTPUTS / "02_controls"
MARGIN = 0.5
CHUNK = 50
SD_NULL, SD_WIDE, REGION_SHIFT = 0.08, 0.18, 0.30  # PREREGISTRATION.md § 3


def score(boundary):
    C.require_parity()
    descriptions, names, tasks = C.part1()
    ctrl = C.controls()["names"]
    scorer = S.Scorer(C.MODEL_KEY, C.WORK / f"02_{boundary}")  # one cache per process
    rows = [d for d in descriptions if d["boundary"] == boundary]
    out = []
    for start in range(0, len(rows), CHUNK):
        chunk = rows[start : start + CHUNK]
        keys, pairs = [], []
        for d in chunk:
            for name in ctrl:
                text = C.text_for(d, names, name)
                for order in S.ORDERS:
                    keys.append((d["description_id"], name, order))
                    pairs.append((tasks[f"{boundary}:{order}"]["system_prompt"], text))
        records = scorer.score(pairs, progress_every=10**9)
        for (did, name, order), r in zip(keys, records):
            t = tasks[f"{boundary}:{order}"]
            out.append(
                {
                    "description_id": did,
                    "name": name,
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
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(out)
    wide = df.pivot_table(
        index=["description_id", "name"], columns="order", values="d"
    ).reset_index()
    wide.columns = ["description_id", "name", "d_A", "d_B"]
    extra = (
        df.groupby(["description_id", "name"])
        .agg(mass_min=("mass", "min"), top1=("top1", lambda x: "".join(sorted(set(x)))))
        .reset_index()
    )
    wide = wide.merge(extra, on=["description_id", "name"])
    wide["s"] = (wide.d_A + wide.d_B) / 2
    wide["family"] = wide.name.map(lambda n: ctrl[n]["family"])
    wide["tokens"] = wide.name.map(lambda n: ctrl[n]["tokens"])
    path = OUT / f"scores_{boundary}.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        wide.to_csv(f, index=False, float_format="%.4f")
    print("wrote", path)


def boot_ci(fn, n, rng):
    stats = [fn(rng.integers(0, n, n)) for _ in range(C.N_BOOT)]
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def label(s):
    return np.where(s > MARGIN, 1, np.where(s < -MARGIN, -1, 0))


def flip_rate(S_matrix):
    lab = label(S_matrix)
    return float(((lab == 1).any(1) & (lab == -1).any(1)).mean())


def analyse_boundary(boundary, part1, ctrl, rng):
    scores = pd.read_csv(OUT / f"scores_{boundary}.csv.gz")
    p1 = part1[part1.boundary == boundary]
    piv1 = p1.pivot(index="description_id", columns="name", values="s")
    kinds = p1.drop_duplicates("name").set_index("name")
    plac = [n for n in piv1.columns if kinds.kind[n] == "fictional"]
    real = [n for n in piv1.columns if kinds.kind[n] == "real"]
    real1 = [n for n in real if int(kinds.tokens[n]) == 1]
    pivc = scores.pivot(index="description_id", columns="name", values="s").loc[
        piv1.index
    ]
    require(pivc.notna().all().all(), f"{boundary}: missing control scores")
    ref = piv1[plac].mean(1)
    delta = pd.concat([piv1[real + plac], pivc], axis=1).sub(ref, axis=0)
    n = len(delta)
    effects = {c: float(delta[c].mean()) for c in delta.columns}

    groups = {"real_all": real, "real_1tok": real1, "placebo": plac}
    for fam in sorted({v["family"] for v in ctrl.values()}):
        groups[fam] = sorted(c for c, v in ctrl.items() if v["family"] == fam)
    fam_stats = {}
    sub = np.random.default_rng(C.SEED)
    for g, cols in groups.items():
        eff = np.array([effects[c] for c in cols])
        X = delta[cols].values
        sd_fn = lambda idx, X=X: float(X[idx].mean(0).std(ddof=1))
        stat = {
            "n_names": len(cols),
            "sd": float(eff.std(ddof=1)),
            "sd_ci": boot_ci(sd_fn, n, rng),
            "mean": float(eff.mean()),
            "mean_ci": boot_ci(lambda idx, X=X: float(X[idx].mean()), n, rng),
            "min": float(eff.min()),
            "max": float(eff.max()),
            "flip_rate": flip_rate(pd.concat([piv1, pivc], axis=1)[cols].values),
        }
        if len(cols) < len(real1):
            # real one-token names, subsampled to the family's size
            R = pd.concat([piv1, pivc], axis=1)[real1].values
            stat["flip_rate_real1_size_matched"] = float(
                np.mean(
                    [
                        flip_rate(
                            R[:, sub.choice(len(real1), len(cols), replace=False)]
                        )
                        for _ in range(200)
                    ]
                )
            )
        fam_stats[g] = stat

    # Frequency proxy: token id of one-token names (BPE merge rank).
    one_tok = {c: ctrl[c]["token_ids"][0] for c in ctrl if ctrl[c]["tokens"] == 1}
    validity = {
        "mass_min": float(scores.mass_min.min()),
        "top1_not_letter": int((~scores.top1.isin(["A", "B", "AB"])).sum()),
    }
    return {
        "n_descriptions": n,
        "families": fam_stats,
        "effects": effects,
        "one_token_controls_token_id": one_tok,
        "validity": validity,
    }, delta


def spearman(x, y):
    from scipy.stats import spearmanr

    r = spearmanr(x, y)
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": len(x)}


def analyse():
    import transformers  # noqa: F401  (tokenizer for real-name token ids)
    from geoflip import scoring

    ctrl = C.controls()["names"]
    _, names, _ = C.part1()
    part1 = pd.read_csv(
        C.PART1 / "dataset" / "labels.csv.gz",
        usecols=["description_id", "boundary", "name", "kind", "tokens", "s"],
    )
    rng = np.random.default_rng(C.SEED)
    spec = scoring.MODELS[C.MODEL_KEY]
    tok = transformers.AutoTokenizer.from_pretrained(
        spec["hf_id"], revision=spec["revision"]
    )
    res = {}
    for b in ("BV", "PR"):
        r, delta = analyse_boundary(b, part1, ctrl, rng)
        # Spearman of effect with token id over all one-token names (real + controls).
        one = {}
        for cid, display in names["real"].items():
            ids = tok.encode(" " + display, add_special_tokens=False)
            if len(ids) == 1:
                one[cid] = ids[0]
        one.update({c: v for c, v in r["one_token_controls_token_id"].items()})
        r["token_id_spearman_one_token"] = spearman(
            [one[c] for c in one], [r["effects"][c] for c in one]
        )
        real1 = [c for c in one if c in names["real"]]
        r["token_id_spearman_real_one_token"] = spearman(
            [one[c] for c in real1], [r["effects"][c] for c in real1]
        )
        counts = {
            c: int(names["tokens"][C.display_name(names, c)])
            for c in list(names["real"]) + list(names["fictional"])
        }
        counts.update({c: v["tokens"] for c, v in ctrl.items()})
        cs = list(counts)
        r["token_count_pearson_all"] = float(
            np.corrcoef([counts[c] for c in cs], [r["effects"][c] for c in cs])[0, 1]
        )
        res[b] = r
    bv = res["BV"]["families"]
    verdict = {
        "nonplace_1tok_sd": bv["nonplace_1tok"]["sd"],
        "real_1tok_sd": bv["real_1tok"]["sd"],
        "region_3tok_mean": bv["region_3tok"]["mean"],
        "H3_falsified": bv["nonplace_1tok"]["sd"] <= SD_NULL
        and bv["real_1tok"]["sd"] >= SD_WIDE,
        "H3_supported": bv["nonplace_1tok"]["sd"] >= SD_WIDE
        or bv["region_3tok"]["mean"] >= REGION_SHIFT,
        "H1_stage1A_falsified": bv["nonplace_1tok"]["sd"] >= bv["real_1tok"]["sd"],
    }
    res["verdict"] = verdict
    write_json(OUT / "results.json", res)
    inputs = [
        C.PART1 / "dataset" / "labels.csv.gz",
        C.OUTPUTS / "01_controls" / "controls.json",
    ]
    outputs = [OUT / "results.json", OUT / "scores_BV.csv.gz", OUT / "scores_PR.csv.gz"]
    C.write_manifest(
        OUT, __file__, inputs, outputs, {"verdict": verdict, "margin": MARGIN}
    )
    for b in ("BV", "PR"):
        print(f"== {b}")
        for g, s in res[b]["families"].items():
            print(
                f"  {g:18s} n={s['n_names']:2d} sd={s['sd']:.3f} {s['sd_ci']} mean={s['mean']:+.3f} range [{s['min']:+.2f},{s['max']:+.2f}] flip={s['flip_rate']:.3f}"
            )
        print("  spearman token id (1-tok):", res[b]["token_id_spearman_one_token"])
    print("verdict:", verdict)


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
