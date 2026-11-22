"""Compile the Part 2 paper figures and facts from the stage outputs.

Reads outputs/*/results.json (each produced under the parity gate) and writes
report/figures/*.png and report/facts.json, the numbers the report quotes.

  python -m country_effect.part2_attribution.scripts.09_compile_report_artifacts
"""

import numpy as np

from country_effect.part2_attribution import common as C
from geoflip.utils import read_json, write_json

REPORT = C.HERE / "report"
FIGS = REPORT / "figures"
# Fixed categorical order (validated default palette, light mode); gray for references.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
GRAY, INK, MUTED = "#9a9993", "#0b0b0b", "#52514e"
SPAN_COLOR = {
    "name": SERIES[0],
    "ev_group": SERIES[1],
    "ev_rest": SERIES[2],
    "ev_all": SERIES[3],
    "bridge": SERIES[4],
    "context": SERIES[6],
    "template": SERIES[4],
    "readout": SERIES[5],
}
SPAN_LABEL = {
    "name": "country name",
    "ev_group": "evidence: group phrase",
    "ev_rest": "evidence: rest",
    "ev_all": "evidence sentence",
    "bridge": "intervening tokens",
    "context": "description text",
    "template": "chat template",
    "readout": "final position",
}
GROUP_COLOR = {"high_higher": SERIES[0], "high_lower": SERIES[1]}
GROUP_LABEL = {"high_higher": "Battle-leaning", "high_lower": "VAC-leaning"}
ROLE_LABEL = {
    "clear_lower": "clear VAC evidence",
    "underspecified": "underspecified evidence",
    "clear_higher": "clear Battle evidence",
}
ITEM_ORDER = (
    "anchor_lo",
    "unarmed",
    "students",
    "security_guard",
    "miners",
    "drivers",
    "returned_fire",
    "anchor_hi",
)


def style(plt):
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.grid": True,
            "grid.color": "#e6e5e1",
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.6,
            "legend.frameon": False,
            "legend.fontsize": 7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def fig_lens(plt, res):
    bv = res["BV"]
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5), sharey=True)
    for g, v in bv["by_country_group"].items():
        axes[0].plot(v["curve"], color=GROUP_COLOR[g], label=GROUP_LABEL[g])
    axes[0].set_title("(a) by country group", loc="left")
    for k, role in enumerate(ROLE_LABEL):
        v = bv["by_item_group"][f"high_lower|{role}"]
        axes[1].plot(v["curve"], color=SERIES[[1, 3, 2][k]], label=ROLE_LABEL[role])
    axes[1].set_title("(b) VAC-leaning countries, by evidence", loc="left")
    for ax in axes:
        ax.axhline(0, color=MUTED, lw=0.7)
        ax.set_xlabel("layer")
        ax.legend(loc="lower left")
    axes[0].set_ylabel("logit-lens $\\Delta$ (nats)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_logit_lens.png")
    plt.close(fig)


def fig_patching(plt, res):
    bv = res["BV"]["by_country_group"]
    spans = ("name", "ev_all", "bridge", "readout")
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), sharey=True)
    for ax, g in zip(axes, ("high_lower", "high_higher")):
        v = bv[g]
        for span in spans:
            F = v[span]["F"]
            ax.plot(range(len(F)), F, color=SPAN_COLOR[span], label=SPAN_LABEL[span])
            ax.fill_between(
                range(len(F)),
                v[span]["lo"],
                v[span]["hi"],
                color=SPAN_COLOR[span],
                alpha=0.15,
                lw=0,
            )
        ax.axhline(0, color=MUTED, lw=0.7)
        ax.axhline(1, color=MUTED, lw=0.5, ls=":")
        ax.set_title(
            f"{GROUP_LABEL[g]} countries ($\\Delta$ = {v['delta_mean']:+.2f} nats)",
            loc="left",
        )
        ax.set_xlabel("patched layer")
    axes[0].set_ylabel("fraction of $\\Delta$ restored")
    axes[0].set_ylim(-0.2, 1.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIGS / "fig2_patching.png")
    plt.close(fig)


def fig_span_control(plt, res):
    spans = ("context", "ev_group", "ev_rest", "template", "readout")
    groups = [g for g in ("high_vac", "us_uk") if g in res["groups"]]
    titles = {"high_vac": "Bangladesh, Nigeria, Sudan", "us_uk": "United States, UK"}
    fig, axes = plt.subplots(1, len(groups), figsize=(7, 2.7), sharey=True)
    for ax, g in zip(np.atleast_1d(axes), groups):
        v = res["groups"][g]
        for span in spans:
            s = v["spans"][span]
            x = range(len(s["F"]))
            ax.plot(x, s["F"], color=SPAN_COLOR[span], label=SPAN_LABEL[span])
            ax.plot(x, s["F_control"], color=SPAN_COLOR[span], lw=0.8, ls="--")
        ax.axhline(0, color=MUTED, lw=0.7)
        ax.axhline(1, color=MUTED, lw=0.5, ls=":")
        ax.set_title(
            f"{titles[g]} ($\\Delta$ = {v['delta_mean']:+.2f} nats)", loc="left"
        )
        ax.set_xlabel("patched layer")
    np.atleast_1d(axes)[0].set_ylabel("fraction of $\\Delta$ restored")
    np.atleast_1d(axes)[0].set_ylim(-0.15, 1.1)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIGS / "fig3_span_control.png")
    plt.close(fig)


def fig_steering(plt, res):
    sc = res["steering_country"]
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))
    for k, (l, v) in enumerate(sorted(sc["layers"].items(), key=lambda x: int(x[0]))):
        ba = v["by_alpha"]
        al = sorted(ba, key=float)
        axes[0].plot(
            [0.0] + [float(a) for a in al],
            [0.0] + [ba[a] for a in al],
            marker="o",
            ms=3.5,
            ls="",
            color=SERIES[[6, 1, 2, 3][k]],
        )
        xs = sorted([0.0] + [float(a) for a in al])
        ys = [0.0 if x == 0 else ba[next(a for a in al if float(a) == x)] for x in xs]
        axes[0].plot(xs, ys, color=SERIES[[6, 1, 2, 3][k]], label=f"layer {l}")
    axes[0].axhline(0, color=MUTED, lw=0.7)
    axes[0].set_xlabel("steering coefficient $\\alpha$")
    axes[0].set_ylabel("shift in $s$ (nats)")
    axes[0].set_title(
        "(a) adding $\\alpha\\,u_\\ell$ at the final position", loc="left"
    )
    axes[0].legend(loc="upper left")
    obs = sc["observed_delta_by_item"]
    ind = sc["layers"]["14"]["induced_by_item"]
    x = [obs[i] for i in ITEM_ORDER]
    y = [ind[i] for i in ITEM_ORDER]
    axes[1].scatter(x, y, s=22, color=SERIES[1], edgecolor="white", lw=0.8, zorder=3)
    for i in ITEM_ORDER:
        axes[1].annotate(
            i.replace("_", " "),
            (obs[i], ind[i]),
            fontsize=6.5,
            color=MUTED,
            xytext=(3, 2),
            textcoords="offset points",
        )
    axes[1].set_xlabel("observed country gap by item (nats)")
    axes[1].set_ylabel("steering-induced shift, layer 14")
    axes[1].set_title(
        f"(b) item profile, r = {sc['layers']['14']['item_profile_r']:.2f}", loc="left"
    )
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_steering.png")
    plt.close(fig)


def fig_ig(plt, res):
    sh = res["BV"]["high_lower"]["shares"]
    layers = np.arange(-1, len(sh["name"]) - 1)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for span in ("name", "ev_group", "ev_rest", "bridge", "readout"):
        ax.plot(layers, sh[span], color=SPAN_COLOR[span], label=SPAN_LABEL[span])
    ax.axhline(0, color=MUTED, lw=0.7)
    ax.set_xlabel("layer (−1 = embedding)")
    ax.set_ylabel("share of attribution")
    ax.legend(loc="center right", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIGS / "fig5_ig.png")
    plt.close(fig)


def fig_qwen(plt, res):
    _, names, _ = C.part1()
    q = res["BV"]
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))
    for kind, cols, color in (
        ("placebo", list(names["fictional"]), GRAY),
        ("real country", list(names["real"]), SERIES[0]),
    ):
        axes[0].scatter(
            [q["llama_effects"][c] for c in cols],
            [q["effects"][c] for c in cols],
            s=14,
            color=color,
            edgecolor="white",
            lw=0.6,
            label=kind,
        )
    axes[0].axhline(0, color=MUTED, lw=0.7)
    axes[0].axvline(0, color=MUTED, lw=0.7)
    axes[0].set_xlabel("Llama-3.2-3B name effect (nats)")
    axes[0].set_ylabel("Qwen3-4B name effect (nats)")
    axes[0].set_title(
        f"(a) r = {q['r_with_llama_real']:.2f} (real countries)", loc="left"
    )
    axes[0].legend(loc="upper left")
    for g, v in q["lens_by_country_group"].items():
        axes[1].plot(v["curve"], color=GROUP_COLOR[g], label=GROUP_LABEL[g])
    axes[1].axhline(0, color=MUTED, lw=0.7)
    axes[1].set_xlabel("Qwen3-4B layer")
    axes[1].set_ylabel("logit-lens effect (nats)")
    axes[1].set_title("(b) logit lens at the final position", loc="left")
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGS / "fig6_qwen.png")
    plt.close(fig)


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style(plt)
    FIGS.mkdir(parents=True, exist_ok=True)
    for old in FIGS.glob("*.png"):
        old.unlink()
    keys = (
        "04_logit_lens",
        "05_patching",
        "05b_span_control",
        "06_probes_steering",
        "07_ig",
        "08_generality",
    )
    res = {
        k: read_json(C.OUTPUTS / k / "results.json")
        for k in keys
        if (C.OUTPUTS / k / "results.json").exists()
    }
    parity = read_json(C.OUTPUTS / "00_parity" / "manifest.json")
    fig_lens(plt, res["04_logit_lens"])
    fig_patching(plt, res["05_patching"])
    if "05b_span_control" in res:
        fig_span_control(plt, res["05b_span_control"])
    fig_steering(plt, res["06_probes_steering"])
    fig_ig(plt, res["07_ig"])
    fig_qwen(plt, res["08_generality"])

    lens = res["04_logit_lens"]["BV"]
    patch = res["05_patching"]["BV"]
    steer = res["06_probes_steering"]["steering_country"]
    facts = {
        "parity": parity["checks"],
        "timing": parity["timing_seconds"],
        "logit_lens": {
            "final_layer_check": lens["final_layer_check"],
            "groups": {
                g: {k: v[k] for k in ("final", "half_layer")}
                for g, v in lens["by_country_group"].items()
            },
            "by_item_group_final": {
                g: v["final"] for g, v in lens["by_item_group"].items()
            },
        },
        "patching": {
            g: {
                "delta_mean": v["delta_mean"],
                "n_pairs": v["n"] if "n" in v else None,
                "peak": {
                    s: {
                        "F": float(np.max(v[s]["F"][:27])),
                        "layer": int(np.argmax(v[s]["F"][:27])),
                    }
                    for s in ("ev_group", "ev_rest", "ev_all", "bridge")
                },
                "readout_F_layers_13_14": v["readout"]["F"][13:15],
                "readout_F_mean_18_26": float(np.mean(v["readout"]["F"][18:27])),
            }
            for g, v in patch["by_country_group"].items()
        },
        "steering_country": {
            l: {
                "by_alpha": v["by_alpha"],
                "item_profile_r": v["item_profile_r"],
                "induced_mean": v["induced_mean"],
                "induced_by_item": v["induced_by_item"],
            }
            for l, v in steer["layers"].items()
        },
        "observed_gap_by_item": steer["observed_delta_by_item"],
        "ig": {
            g: {
                "delta_mean": v["delta_mean"],
                "completeness_error_median": v["completeness_error_median"],
            }
            for g, v in res["07_ig"]["BV"].items()
        },
        "qwen": {
            k: res["08_generality"]["BV"][k]
            for k in ("sd_real", "sd_placebo", "r_with_llama_real", "mass_p5")
        },
    }
    if "05b_span_control" in res:
        facts["span_control"] = {
            g: {
                "delta_mean": v["delta_mean"],
                "n_pairs": v["n_pairs"],
                "peak_excess": {
                    s: {
                        "F_excess": float(np.max(x["F_excess"][:27])),
                        "layer": int(np.argmax(x["F_excess"][:27])),
                        "ci": [
                            x["F_excess_lo"][int(np.argmax(x["F_excess"][:27]))],
                            x["F_excess_hi"][int(np.argmax(x["F_excess"][:27]))],
                        ],
                    }
                    for s, x in v["spans"].items()
                },
            }
            for g, v in res["05b_span_control"]["groups"].items()
        }
    write_json(REPORT / "facts.json", facts)
    outputs = [REPORT / "facts.json"] + sorted(FIGS.glob("*.png"))
    C.write_manifest(
        REPORT, __file__, [C.OUTPUTS / k / "results.json" for k in res], outputs
    )
    print("figures:", [p.name for p in sorted(FIGS.glob("*.png"))])


if __name__ == "__main__":
    main()
