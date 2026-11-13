"""Part 1: labels per description x name, the pre-specified statistics and figures
(DESIGN.md § 5). Reads the score cache directly (no model needed); a description
is analysed only when all of its 58 names x 2 orders are scored.

  python -m country_effect.part1_behavioral.analyse [--partial]

Writes dataset/labels.csv.gz, dataset/description_summary.csv,
report/results.json and report/figures/*.png.
"""

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

from geoflip import study as S
from geoflip.render import country_conditions, country_phrase

from .score import WORK, load

HERE = Path(__file__).resolve().parent
M = 0.5
SEED = 20261107
N_BOOT = 2000
N_SUBSETS = 500
CLASSES = {"PR": ("Protest", "Riot"), "BV": ("Violence against civilians", "Battle")}

# Reference palette (dataviz skill, references/palette.md), light mode.
HIGH, LOW, NEUTRAL = "#2a78d6", "#eb6834", "#bdbcb5"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def read_cache():
    path = WORK / "cache" / "llama3.2-3b_bfloat16_0cb88a4f76_first_token.jsonl"
    cache = {}
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # a line being written
                continue
            cache[r["key"]] = r
    return cache


def label(a, b, m):
    """Strict label from the two orders: 1 higher, -1 lower, 0 order-dependent."""
    return np.where((a > m) & (b > m), 1, np.where((a < -m) & (b < -m), -1, 0))


def labels_table(descriptions, names, tasks, cache):
    rows = []
    conds = country_conditions(names, ("none", "real", "fictional"))
    for d in descriptions:
        recs = []
        for cid, display, kind in conds:
            text = d["template"].replace("{COUNTRY}", country_phrase(names, display))
            got = []
            for order in S.ORDERS:
                t = tasks[f"{d['boundary']}:{order}"]
                r = cache.get(S.Scorer.key(t["system_prompt"], text))
                if r is None:
                    break
                got.append((r, t))
            if len(got) < 2:
                recs = None
                break
            (ra, ta), (rb, tb) = got
            recs.append({
                "description_id": d["description_id"], "boundary": d["boundary"],
                "name": cid, "kind": kind,
                "tokens": names["tokens"].get(display, 0) if display else 0,
                "d_A": ra["lp"][ta["letter_higher"]] - ra["lp"][ta["letter_lower"]],
                "d_B": rb["lp"][tb["letter_higher"]] - rb["lp"][tb["letter_lower"]],
                "mass_A": ra["mass"], "mass_B": rb["mass"],
                "top1_A": ra["top"][0][0], "top1_B": rb["top"][0][0],
            })
        if recs:
            rows += recs
    df = pd.DataFrame(rows)
    df["s"] = (df.d_A + df.d_B) / 2
    df["b"] = (df.d_A - df.d_B) / 2
    for m, tag in ((M, ""), (0.0, "_m0")):
        df["strict" + tag] = label(df.d_A.values, df.d_B.values, m)
        df["avg" + tag] = np.where(df.s > m, 1, np.where(df.s < -m, -1, 0))
    return df


def word(boundary, v, kind):
    lo, hi = CLASSES[boundary]
    if v == 1:
        return hi
    if v == -1:
        return lo
    return "order-dependent" if kind == "strict" else "near boundary"


def summary_table(df, desc):
    out = []
    for did, g in df.groupby("description_id", sort=False):
        real, plac = g[g.kind == "real"], g[g.kind == "fictional"]
        none = g[g.kind == "none"].iloc[0]
        b = none.boundary
        row = {"description_id": did, "boundary": b}
        for col in ("strict", "avg"):
            for kname, part in (("real", real), ("placebo", plac)):
                row[f"{col}_{kname}_higher"] = int((part[col] == 1).sum())
                row[f"{col}_{kname}_lower"] = int((part[col] == -1).sum())
                row[f"{col}_{kname}_other"] = int((part[col] == 0).sum())
            row[f"{col}_flip"] = bool(row[f"{col}_real_higher"] and row[f"{col}_real_lower"])
            row[f"{col}_placebo_flip"] = bool(row[f"{col}_placebo_higher"] and row[f"{col}_placebo_lower"])
            row[f"{col}_none"] = word(b, none[col], col)
            hi = sorted(real[real[col] == 1].name)
            lo = sorted(real[real[col] == -1].name)
            row[f"{col}_real_higher_names"] = ";".join(hi)
            row[f"{col}_real_lower_names"] = ";".join(lo)
        row["s_none"] = none.s
        row["s_real_mean"] = real.s.mean()
        row["s_real_sd"] = real.s.std()
        row["most_higher_country"] = real.loc[real.s.idxmax(), "name"]
        row["most_lower_country"] = real.loc[real.s.idxmin(), "name"]
        out.append(row)
    summ = pd.DataFrame(out)
    return desc.merge(summ, on=["description_id", "boundary"])


def boot_ci(values_fn, n, rng):
    """95% interval of a statistic over descriptions resampled with replacement."""
    stats = [values_fn(rng.integers(0, n, n)) for _ in range(N_BOOT)]
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def matrices(df, b, col):
    g = df[df.boundary == b]
    piv = g.pivot(index="description_id", columns="name", values=col)
    kinds = g.drop_duplicates("name").set_index("name").kind
    real = [c for c in piv.columns if kinds[c] == "real"]
    plac = [c for c in piv.columns if kinds[c] == "fictional"]
    return piv, real, plac


def flip_stats(df, b, col, rng):
    piv, real, plac = matrices(df, b, col)
    R, P = piv[real].values, piv[plac].values
    n = len(piv)
    nh_r, nl_r = (R == 1).sum(1), (R == -1).sum(1)
    nh_p, nl_p = (P == 1).sum(1), (P == -1).sum(1)
    flip_r = (nh_r > 0) & (nl_r > 0)
    flip_p = (nh_p > 0) & (nl_p > 0)
    pair_r = nh_r * nl_r / comb(len(real), 2)
    pair_p = nh_p * nl_p / comb(len(plac), 2)
    sub = np.random.default_rng(SEED)
    subsets = [sub.choice(len(real), len(plac), replace=False) for _ in range(N_SUBSETS)]
    flip_r20 = np.mean([((R[:, s] == 1).any(1) & (R[:, s] == -1).any(1)) for s in subsets], axis=0)
    return {
        "n_descriptions": n,
        "flip_real": float(flip_r.mean()), "flip_real_ci": boot_ci(lambda i: flip_r[i].mean(), n, rng),
        "flip_placebo": float(flip_p.mean()), "flip_placebo_ci": boot_ci(lambda i: flip_p[i].mean(), n, rng),
        "flip_real_size_matched": float(flip_r20.mean()),
        "flip_real_size_matched_ci": boot_ci(lambda i: flip_r20[i].mean(), n, rng),
        "size_matched_minus_placebo_ci": boot_ci(lambda i: flip_r20[i].mean() - flip_p[i].mean(), n, rng),
        "pair_real": float(pair_r.mean()), "pair_real_ci": boot_ci(lambda i: pair_r[i].mean(), n, rng),
        "pair_placebo": float(pair_p.mean()), "pair_placebo_ci": boot_ci(lambda i: pair_p[i].mean(), n, rng),
        "pair_ratio": float(pair_r.mean() / pair_p.mean()) if pair_p.mean() else None,
        "pair_diff_ci": boot_ci(lambda i: pair_r[i].mean() - pair_p[i].mean(), n, rng),
    }


def country_effects(df, b, desc, rng):
    piv, real, plac = matrices(df, b, "s")
    delta = piv.sub(piv[plac].mean(1), axis=0)
    n = len(piv)
    eff = {}
    for c in real + plac:
        v = delta[c].values
        eff[c] = {"mean": float(v.mean()), "ci": boot_ci(lambda i: v[i].mean(), n, rng)}
    half = desc.set_index("description_id").loc[piv.index, "half"].astype(int).values
    e0 = delta[real][half == 0].mean()
    e1 = delta[real][half == 1].mean()
    p0 = delta[plac][half == 0].mean()
    p1 = delta[plac][half == 1].mean()
    out = {"effects": eff,
           "split_half_r_real": float(np.corrcoef(e0, e1)[0, 1]),
           "split_half_r_placebo": float(np.corrcoef(p0, p1)[0, 1])}
    for kname, cols in (("real", real), ("placebo", plac)):
        X = piv[cols]
        centred = X.sub(X.mean(1), axis=0)
        beta = centred.mean(0)
        out[f"beta_sd_{kname}"] = float(beta.std())
        out[f"variance_share_{kname}"] = float((beta ** 2).sum() * n / (centred ** 2).values.sum())
        out[f"within_sd_{kname}"] = float(np.sqrt((centred ** 2).values.mean()))
    tok = df[df.boundary == b].drop_duplicates("name").set_index("name").tokens
    beta_r = piv[real].sub(piv[real].mean(1), axis=0).mean(0)
    beta_p = piv[plac].sub(piv[plac].mean(1), axis=0).mean(0)
    out["token_r_real"] = float(np.corrcoef(beta_r, tok[real])[0, 1])
    out["token_r_placebo"] = float(np.corrcoef(beta_p, tok[plac])[0, 1])
    return out, delta, real, plac


def validity(df, desc, b):
    g = df[df.boundary == b].merge(desc[["description_id", "role"]], on="description_id")
    out = {
        "n_prompts": int(2 * len(g)),
        "top1_letter_share": float(np.mean(np.r_[g.top1_A.isin(["A", "B"]), g.top1_B.isin(["A", "B"])])),
        "mass_p05": float(np.percentile(np.r_[g.mass_A, g.mass_B], 5)),
        "order_effect_mean": float(g.b.mean()), "order_effect_abs_mean": float(g.b.abs().mean()),
        "order_dependent_share": float((g.strict == 0).mean()),
        "order_dependent_share_m0": float((g.strict_m0 == 0).mean()),
    }
    for role, want in (("anchor_lower", -1), ("anchor_higher", 1)):
        a = g[g.role == role]
        for kind in ("none", "real", "fictional"):
            k = a[a.kind == kind]
            out[f"{role}_{kind}_strict_acc"] = float((k.strict == want).mean())
            out[f"{role}_{kind}_avg_acc"] = float((k.avg == want).mean())
    return out


def by_factor(summ, b, col):
    s = summ[summ.boundary == b]
    out = {}
    factors = ["frame", "wording", "place"] + (["actors", "grievance"] if b == "PR" else ["time"])
    for f in factors:
        out[f] = {k: float(v) for k, v in s.groupby(f)[f"{col}_flip"].mean().items()}
    it = s.groupby("item").agg(p=("acled_p_higher", "first"), flip=(f"{col}_flip", "mean"),
                               role=("role", "first")).sort_values("p")
    out["item"] = {k: {"acled_p_higher": float(r.p), "flip": float(r.flip), "role": r.role} for k, r in it.iterrows()}
    return out


def none_baseline(df, b):
    g = df[df.boundary == b]
    none = g[g.kind == "none"].set_index("description_id")
    real = g[g.kind == "real"]
    out = {}
    for col in ("strict", "avg"):
        ref = real.description_id.map(none[col])
        both = (real[col] != 0) & (ref != 0)
        out[f"{col}_differs_from_none"] = float((real[col][both] != ref[both]).mean()) if both.any() else None
    return out


# ------------------------------------------------------------------ figures

def style(plt):
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 200,
        "legend.frameon": False,
    })


def fig_ladder(plt, df, desc, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.2))
    for ax, b in zip(axes, ("PR", "BV")):
        g = df[(df.boundary == b) & (df.kind == "real")].merge(
            desc[["description_id", "item", "acled_p_higher"]], on="description_id")
        t = g.groupby("item").agg(p=("acled_p_higher", "first"),
                                  hi=("strict", lambda x: (x == 1).mean()),
                                  lo=("strict", lambda x: (x == -1).mean()),
                                  od=("strict", lambda x: (x == 0).mean())).sort_values("p")
        y = np.arange(len(t))
        lo_, hi_ = CLASSES[b]
        ax.barh(y, t.lo, color=LOW, height=0.8, edgecolor="#fcfcfb", linewidth=1, label=lo_)
        ax.barh(y, t.od, left=t.lo, color=NEUTRAL, height=0.8, edgecolor="#fcfcfb", linewidth=1, label="order-dependent")
        ax.barh(y, t.hi, left=t.lo + t.od, color=HIGH, height=0.8, edgecolor="#fcfcfb", linewidth=1, label=hi_)
        ax.set_yticks(y, [f"{i}  ({p:.2f})" for i, p in zip(t.index, t.p)], fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of description × real-country cells (strict label, m = 0.5)")
        ax.set_title(f"{b}: items ordered by ACLED P({hi_} | act)", color=INK, fontsize=10, loc="left")
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_ladder_avg(plt, df, desc, path):
    """Order-averaged score per item: one dot per real country (mean over the item's descriptions)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.2))
    for ax, b in zip(axes, ("PR", "BV")):
        g = df[df.boundary == b].merge(desc[["description_id", "item", "acled_p_higher"]], on="description_id")
        order = g.groupby("item").acled_p_higher.first().sort_values()
        y = {it: k for k, it in enumerate(order.index)}
        lo_, hi_ = CLASSES[b]
        ax.axvspan(-M, M, color=NEUTRAL, alpha=0.3, lw=0, label="near boundary (|s| ≤ 0.5)")
        ax.axvline(0, color=INK2, lw=0.8)
        rc = g[g.kind == "real"].groupby(["item", "name"]).s.mean().reset_index()
        jit = rc.name.map({n: (k % 7 - 3) * 0.05 for k, n in enumerate(sorted(rc.name.unique()))})
        ax.scatter(rc.s, rc.item.map(y) + jit, s=6, color=NEUTRAL, edgecolor="none", label="real country")
        for cname, colr in (("United States", HIGH), ("Bangladesh", LOW)):
            r1 = rc[rc.name == cname]
            ax.scatter(r1.s, r1.item.map(y), s=14, color=colr, edgecolor="none", label=cname, zorder=3)
        nc = g[g.kind == "none"].groupby("item").s.mean()
        ax.scatter(nc.values, [y[i] for i in nc.index], marker="|", s=90, color=INK, label="no country", zorder=4)
        ax.set_yticks(range(len(order)), [f"{i}  ({p:.2f})" for i, p in order.items()], fontsize=7)
        ax.set_xlabel(f"order-averaged score s (nats)\n← {lo_}     {hi_} →")
        ax.set_title(f"{b}: items ordered by ACLED P({hi_} | act)", color=INK, fontsize=10, loc="left")
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def letter_pref(g, b):
    """log P(A) − log P(B) averaged over orders. Order A lists the higher class as letter A in BV, B in PR."""
    return -g.b if b == "PR" else g.b


def fig_flip_matrix(plt, df, desc, res, path, b="BV"):
    """Every flip description (rows) × every name (columns), coloured by the order-averaged label."""
    from matplotlib.colors import ListedColormap
    g = df[df.boundary == b]
    piv = g.pivot_table(index="description_id", columns="name", values="avg")
    R = g[g.kind == "real"]
    flip = R.groupby("description_id").avg.agg(lambda v: bool((v == 1).any() and (v == -1).any()))
    rows = flip[flip].index
    e, kinds = res[b]["country"]["effects"], res[b]["country"]["kinds"]
    real = sorted([c for c in e if kinds[c] == "real"], key=lambda c: -e[c]["mean"])
    plac = sorted([c for c in e if kinds[c] != "real"], key=lambda c: -e[c]["mean"])
    d = desc.set_index("description_id").loc[rows]
    snone = g[g.kind == "none"].set_index("description_id").s.loc[rows]
    item_order = d.groupby("item").size().sort_values(ascending=False).index
    rows = sorted(rows, key=lambda r: (list(item_order).index(d.loc[r, "item"]), -snone[r]))
    lo_, hi_ = CLASSES[b]
    cmap = ListedColormap([LOW, "#efeeea", HIGH])
    fig, axes = plt.subplots(1, 3, figsize=(11, 7.2), sharey=True,
                             gridspec_kw={"width_ratios": [len(real), len(plac), 1.6], "wspace": 0.04})
    for ax, cols, title in ((axes[0], real, "37 real countries, ordered by country effect"),
                            (axes[1], plac, "20 placebo names"), (axes[2], ["none"], "none")):
        ax.imshow(piv.loc[rows, cols].values, cmap=cmap, vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=6.5)
        ax.set_title(title, fontsize=8.5, loc="left", color=INK)
        ax.grid(False)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.tick_params(length=0)
    # item bands on the left
    labels, starts = [], []
    for k, r in enumerate(rows):
        it = d.loc[r, "item"]
        if not labels or labels[-1] != it:
            labels.append(it)
            starts.append(k)
    starts.append(len(rows))
    axes[0].set_yticks([(a + z - 1) / 2 for a, z in zip(starts[:-1], starts[1:])],
                       [f"{l} ({z - a})" for l, a, z in zip(labels, starts[:-1], starts[1:])], fontsize=7)
    for a in starts[1:-1]:
        for ax in axes:
            ax.axhline(a - 0.5, color="#fcfcfb", lw=1.5)
    axes[0].set_ylabel(f"the {len(rows)} BV descriptions whose label flips across real countries (grouped by evidence item)")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=HIGH, label=hi_), Patch(color="#efeeea", label="near boundary (|s| ≤ 0.5)"),
                        Patch(color=LOW, label=lo_)], loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(top=0.9, bottom=0.17, left=0.2, right=0.99)
    fig.savefig(path)
    plt.close(fig)


def posthoc(df, b, res_b):
    """Robustness checks added after the pre-specified analysis (DESIGN.md § 7)."""
    g = df[df.boundary == b]
    R = g[g.kind == "real"]
    flip = lambda x: x.groupby("description_id").avg.agg(lambda v: bool((v == 1).any() and (v == -1).any())).mean()
    loo = {c: float(flip(R[R.name != c])) for c in sorted(R.name.unique())}
    e, kinds = res_b["country"]["effects"], res_b["country"]["kinds"]
    tok = g.groupby("name").tokens.first()
    one = [e[c]["mean"] for c in e if kinds[c] == "real" and tok[c] == 1]
    ell = letter_pref(g, b)
    return {"avg_flip_leave_one_out": loo,
            "effect_sd_single_token_real": float(np.std(one, ddof=1)), "n_single_token_real": len(one),
            "letter_pref_mean": float(ell.mean()), "letter_pref_share_favours_A": float((ell > 0).mean())}


def fig_flips(plt, res, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, col in zip(axes, ("strict", "avg")):
        x = np.arange(2)
        for k, (key, colr, lab) in enumerate((("flip_real_size_matched", HIGH, "20 real countries (size-matched)"),
                                              ("flip_placebo", NEUTRAL, "20 placebo names"))):
            vals = [res[b][col]["flips"][key] for b in ("PR", "BV")]
            cis = [res[b][col]["flips"][key + "_ci"] for b in ("PR", "BV")]
            err = np.array([[v - c[0] for v, c in zip(vals, cis)], [c[1] - v for v, c in zip(vals, cis)]])
            ax.bar(x + (k - 0.5) * 0.36, vals, width=0.34, color=colr, label=lab, yerr=err,
                   error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 3})
        ax.set_xticks(x, ["PR", "BV"])
        ax.set_title({"strict": "Strict label (same answer in both orders)",
                      "avg": "Order-averaged label"}[col], fontsize=9, loc="left", color=INK)
        ax.grid(axis="x", visible=False)
        if col == "strict" and max(res[b][col]["flips"]["flip_real_size_matched"] for b in ("PR", "BV")) == 0:
            ax.text(0.5, 0.5, "no description flips\nunder the strict label", transform=ax.transAxes,
                    ha="center", va="center", color=INK2, fontsize=9)
    axes[0].set_ylabel("share of descriptions that flip")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_countries(plt, res, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 7.5))
    for ax, b in zip(axes, ("PR", "BV")):
        e = res[b]["country"]["effects"]
        kinds = res[b]["country"]["kinds"]
        real = sorted([c for c in e if kinds[c] == "real"], key=lambda c: e[c]["mean"])
        plac = [e[c]["mean"] for c in e if kinds[c] == "fictional"]
        lo_, hi_ = CLASSES[b]
        ax.axvspan(min(plac), max(plac), color=NEUTRAL, alpha=0.35, lw=0, label="range of placebo names")
        ax.axvline(0, color=INK2, lw=0.8)
        y = np.arange(len(real))
        m = [e[c]["mean"] for c in real]
        lo = [e[c]["mean"] - e[c]["ci"][0] for c in real]
        hi = [e[c]["ci"][1] - e[c]["mean"] for c in real]
        ax.errorbar(m, y, xerr=[lo, hi], fmt="o", ms=4, color=HIGH, ecolor=HIGH, elinewidth=1, capsize=0,
                    label="real country (95% CI)")
        ax.set_yticks(y, real, fontsize=7)
        ax.set_xlabel(f"mean shift of order-averaged score vs placebos (nats)\n← {lo_}     {hi_} →")
        ax.set_title(f"{b}: country effects", fontsize=10, loc="left", color=INK)
        ax.grid(axis="y", visible=False)
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_order(plt, df, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, b in zip(axes, ("PR", "BV")):
        g = df[df.boundary == b]
        ell = letter_pref(g, b)
        ax.hist(ell, bins=np.arange(-4, 4.01, 0.125), color=HIGH, edgecolor="#fcfcfb", linewidth=0.5)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(f"{b}: letter preference, mean {ell.mean():+.2f} nats", fontsize=9, loc="left", color=INK)
        ax.set_xlabel("log P(A) − log P(B), averaged over both orders (nats)\n← favours B        favours A →")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("prompts pairs")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_split(plt, res, path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, b in zip(axes, ("PR", "BV")):
        h = res[b]["country"]["halves"]
        ax.scatter(h["half0"], h["half1"], s=22, color=HIGH, edgecolor="#fcfcfb", linewidth=1)
        lim = max(np.abs(h["half0"]).max(), np.abs(h["half1"]).max()) * 1.15
        ax.plot([-lim, lim], [-lim, lim], color=INK2, lw=0.8, ls="--")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        r = res[b]["country"]["split_half_r_real"]
        ax.set_title(f"{b}: split-half country effects, r = {r:.2f}", fontsize=9, loc="left", color=INK)
        ax.set_xlabel("contexts half 0 (nats)")
        ax.set_ylabel("contexts half 1 (nats)")
        for c, x0, y0 in zip(h["names"], h["half0"], h["half1"]):
            if abs(x0) > 0.6 * lim or abs(y0) > 0.6 * lim:
                ax.annotate(c, (x0, y0), fontsize=6, color=INK2, xytext=(3, 2), textcoords="offset points")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--partial", action="store_true", help="allow an incomplete cache (exploratory)")
    args = p.parse_args(argv)
    descriptions, names, tasks = load()
    desc = pd.DataFrame(descriptions)
    desc["acled_p_higher"] = desc.acled_p_higher.astype(float)
    df = labels_table(descriptions, names, tasks, read_cache())
    done = df.description_id.nunique()
    print(f"complete descriptions: {done}/{len(descriptions)}")
    if done < len(descriptions) and not args.partial:
        raise SystemExit("cache incomplete; rerun score.py or pass --partial")
    out_ds = HERE / "dataset"
    summ = summary_table(df, desc)
    if not args.partial:
        df.to_csv(out_ds / "labels.csv.gz", index=False, float_format="%.4f")
        summ.drop(columns=["template", "text_none"]).to_csv(out_ds / "description_summary.csv", index=False, float_format="%.4f")

    rng = np.random.default_rng(SEED)
    res = {"complete_descriptions": int(done), "margin": M}
    for b in ("PR", "BV"):
        res[b] = {"validity": validity(df, desc, b), "none_baseline": none_baseline(df, b)}
        for col in ("strict", "avg", "strict_m0", "avg_m0"):
            res[b][col] = {"flips": flip_stats(df, b, col, rng)}
        for col in ("strict", "avg"):
            res[b][col]["by_factor"] = by_factor(summ, b, col)
        ce, delta, real, plac = country_effects(df, b, desc, rng)
        half = desc.set_index("description_id").loc[delta.index, "half"].astype(int).values
        ce["halves"] = {"names": real, "half0": delta[real][half == 0].mean().tolist(),
                        "half1": delta[real][half == 1].mean().tolist()}
        ce["kinds"] = {c: ("real" if c in real else "fictional") for c in real + plac}
        res[b]["country"] = ce
        res[b]["posthoc"] = posthoc(df, b, res[b])
    rep = HERE / "report"
    name = "results_partial.json" if args.partial else "results.json"
    (rep / name).write_text(json.dumps(res, indent=1) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    figs = rep / ("figures_partial" if args.partial else "figures")
    figs.mkdir(exist_ok=True)
    fig_ladder(plt, df, desc, figs / "fig2_ladder_strict.png")
    fig_flips(plt, res, figs / "fig4_flips_vs_placebo.png")
    fig_countries(plt, res, figs / "fig5_country_effects.png")
    fig_order(plt, df, figs / "fig1_letter_preference.png")
    fig_split(plt, res, figs / "fig6_split_half.png")
    fig_ladder_avg(plt, df, desc, figs / "fig3_ladder_averaged.png")
    fig_flip_matrix(plt, df, desc, res, figs / "fig7_flip_matrix.png")
    print(json.dumps({b: {c: {k: res[b][c]["flips"][k] for k in ("flip_real", "flip_real_size_matched", "flip_placebo", "pair_real", "pair_placebo")} for c in ("strict", "avg")} for b in ("PR", "BV")}, indent=1))


if __name__ == "__main__":
    main()
