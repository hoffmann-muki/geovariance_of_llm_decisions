"""Stage 2: armed-status probe, projection test and steering (PREREGISTRATION.md § 6).

Pilot split (context indices 0-7), BV only:
  * v_armed(l): difference of class means of h_l at ev_end on `none` prompts of
    the labelled items (both orders averaged); layer by two-fold (by context)
    accuracy; scale g = pilot armed - unarmed gap of the projection.
  * u(l): mean over pilot descriptions of h_l at the read-out for High-Battle
    minus High-VAC one-token countries, l in the configured steering layers.
Study panel:
  * projection shift (p(country) - p(baseline)) / g at ev_end (primary) and
    over ev_group (secondary), High-Battle minus High-VAC;
  * steering (a): + alpha g v_armed at ev_end at the probe layer;
  * steering (b): + alpha u(l) at the read-out at layer l.

  python -m country_effect.part2_attribution.scripts.06_run_probes_and_steering --run
  python -m country_effect.part2_attribution.scripts.06_run_probes_and_steering --analyse
"""

import argparse
import json

import numpy as np

from country_effect.part2_attribution import common as C
from geoflip.hooks import letter_d
from geoflip.utils import read_json, require, write_json

OUT = C.OUTPUTS / "06_probes_steering"
ARRAYS = C.WORK / "06_arrays"
H2_SIZE = 0.05


def order_avg_states(runner, d, names, display, tasks, tokens, spans):
    """Order-averaged mean states {label: [L, hidden]} over the given spans
    ({label: span key}), and the order-averaged score s."""
    import torch

    acc, s = {}, 0.0
    for order in "AB":
        task = tasks[f"BV:{order}"]
        p = C.resolve_prompt(runner, d, names, display, task, tokens)
        log_probs, states = runner.run(p.ids, capture=True)
        s += float(letter_d(log_probs, runner.letter_ids(p.text, p.ids), task)) / 2
        H = torch.stack(states[1:])  # [L, seq, hidden]
        for label, span in spans.items():
            v = H[:, p.spans[span]].float().mean(1).cpu().numpy() / 2
            acc[label] = acc.get(label, 0) + v
    return acc, s


def fit_probe(X, y, folds):
    """Mass-mean probe per layer: accuracy by two-fold CV, then fit on all."""
    n_layers = X.shape[1]
    acc = []
    for l in range(n_layers):
        correct = 0
        for f in (0, 1):
            tr, te = folds != f, folds == f
            mu1, mu0 = X[tr & (y == 1), l].mean(0), X[tr & (y == 0), l].mean(0)
            v = (mu1 - mu0) / np.linalg.norm(mu1 - mu0)
            thr = ((mu1 + mu0) / 2) @ v
            correct += ((X[te, l] @ v > thr) == (y[te] == 1)).sum()
        acc.append(correct / len(y))
    best = int(np.argmax(np.round(acc, 6)))
    mu1, mu0 = X[y == 1, best].mean(0), X[y == 0, best].mean(0)
    v = (mu1 - mu0) / np.linalg.norm(mu1 - mu0)
    return best, v, [float(a) for a in acc]


def run():
    import torch

    C.require_parity()
    _, names, tasks = C.part1()
    cfg = C.panel()
    runner, _ = C.load_runner()
    ARRAYS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pilot = C.build_descriptions("BV", "pilot")
    labelled = {i: 1 for i in cfg["probe"]["armed"]} | {
        i: 0 for i in cfg["probe"]["unarmed"]
    }

    # ---- probe on pilot none prompts
    X, Xg, y, folds = [], [], [], []
    for n, d in enumerate([d for d in pilot if d["item"] in labelled], 1):
        acc, _ = order_avg_states(
            runner,
            d,
            names,
            None,
            tasks,
            None,
            {"ev_end": "ev_end", "ev_group": "ev_group"},
        )
        X.append(acc["ev_end"].astype(np.float32))
        Xg.append(acc["ev_group"].astype(np.float32))
        y.append(labelled[d["item"]])
        cls = cfg["probe"]["armed" if labelled[d["item"]] else "unarmed"]
        folds.append(cls.index(d["item"]) % 2)  # folds by item: test items are unseen
        if n % 100 == 0:
            print("probe", n, flush=True)
    X, Xg, y, folds = np.array(X), np.array(Xg), np.array(y), np.array(folds)
    layer, v, acc = fit_probe(X, y, folds)
    proj = X[:, layer] @ v
    gap = float(proj[y == 1].mean() - proj[y == 0].mean())
    projg = Xg[:, layer] @ v
    gap_group = float(projg[y == 1].mean() - projg[y == 0].mean())
    np.save(ARRAYS / "v_armed.npy", v)
    probe = {
        "layer": layer,
        "cv_accuracy": acc,
        "gap": gap,
        "gap_ev_group": gap_group,
        "n": int(len(y)),
        "n_armed": int(y.sum()),
    }
    write_json(OUT / "probe.json", probe)
    print("probe layer", layer, "acc", acc[layer], "gap", gap, flush=True)

    # ---- country direction on pilot read-out states
    hb = [
        c
        for c in cfg["boundaries"]["BV"]["ig_countries"]
        if C.country_group("BV", c) == "high_higher"
    ]
    hv = [
        c
        for c in cfg["boundaries"]["BV"]["ig_countries"]
        if C.country_group("BV", c) == "high_lower"
    ]
    layers = cfg["steering"]["country_layers"]
    u = np.zeros((len(layers), 3072))
    for n, d in enumerate(pilot, 1):
        diff = 0
        for group, sign in ((hb, 1), (hv, -1)):
            for c in group:
                acc_c, _ = order_avg_states(
                    runner,
                    d,
                    names,
                    C.display_name(names, c),
                    tasks,
                    1,
                    {"readout": "readout"},
                )
                diff = diff + sign * acc_c["readout"][layers] / len(group)
        u += diff / len(pilot)
        if n % 100 == 0:
            print("country direction", n, flush=True)
    np.save(ARRAYS / "u_country.npy", u)

    # ---- study panel: projections and steering
    store = C.JsonlStore(OUT / "panel.jsonl")
    v_t = torch.tensor(v, device=runner.device)
    u_t = torch.tensor(u, device=runner.device, dtype=torch.float32)
    alphas = cfg["steering"]["alphas"]
    pairs = C.panel_pairs("BV")
    seen = set()
    for n, p in enumerate(pairs, 1):
        d = p["d"]
        for disp in (p["display"], p["baseline"]):
            key = f"proj|{d['description_id']}|{disp}"
            if key in store or key in seen:
                continue
            seen.add(key)
            acc, s = order_avg_states(
                runner,
                d,
                names,
                disp,
                tasks,
                p["tokens"],
                {"ev_end": "ev_end", "ev_group": "ev_group"},
            )
            store.add(
                {
                    "key": key,
                    "description_id": d["description_id"],
                    "name": disp,
                    "s": s,
                    "p_end": float(acc["ev_end"][layer] @ v),
                    "p_group": float(acc["ev_group"][layer] @ v),
                }
            )
        if p["tokens"] != 1:
            continue
        key = f"steer|{d['description_id']}|{p['baseline']}"
        if key in store or key in seen:
            continue
        seen.add(key)
        res = {"armed": {}, "country": {}}
        for order in "AB":
            task = tasks[f"BV:{order}"]
            prompt = C.resolve_prompt(runner, d, names, p["baseline"], task, 1)
            ids = runner.letter_ids(prompt.text, prompt.ids)
            for a in alphas:
                vec = (a * gap) * v_t
                lp, _ = runner.run(
                    prompt.ids, add={layer: (prompt.spans["ev_end"], vec)}
                )
                res["armed"].setdefault(str(a), 0.0)
                res["armed"][str(a)] += float(letter_d(lp, ids, task)) / 2
                for k, l in enumerate(layers):
                    lp, _ = runner.run(
                        prompt.ids, add={l: (prompt.spans["readout"], a * u_t[k])}
                    )
                    res["country"].setdefault(f"{l}|{a}", 0.0)
                    res["country"][f"{l}|{a}"] += float(letter_d(lp, ids, task)) / 2
        store.add(
            {
                "key": key,
                "description_id": d["description_id"],
                "name": p["baseline"],
                **res,
            }
        )
        if n % 50 == 0:
            print("panel", n, "/", len(pairs), flush=True)
    print("done", flush=True)


def boot(fn, n, rng):
    stats = [fn(rng.integers(0, n, n)) for _ in range(C.N_BOOT)]
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def analyse():
    from geoflip.utils import read_jsonl

    rng = np.random.default_rng(C.SEED)
    cfg = C.panel()
    probe = read_json(OUT / "probe.json")
    rows = {r["key"]: r for r in read_jsonl(OUT / "panel.jsonl")}
    pairs = C.panel_pairs("BV")
    shift = []
    for p in pairs:
        did = p["d"]["description_id"]
        c, b = rows[f"proj|{did}|{p['display']}"], rows[f"proj|{did}|{p['baseline']}"]
        shift.append(
            {
                "group": p["country_group"],
                "item": p["d"]["item"],
                "item_group": p["item_group"],
                "country": p["country"],
                "end": (c["p_end"] - b["p_end"]) / probe["gap"],
                "group_proj": (c["p_group"] - b["p_group"]) / probe["gap_ev_group"],
                "delta": c["s"] - b["s"],
                "description_id": did,
            }
        )
    out = {"probe": probe}
    for field in ("end", "group_proj"):
        by_desc = {}
        for r in shift:
            by_desc.setdefault(r["description_id"], {}).setdefault(
                r["group"], []
            ).append(r[field])
        dids = sorted(by_desc)
        diff = np.array(
            [
                np.mean(by_desc[k]["high_higher"]) - np.mean(by_desc[k]["high_lower"])
                for k in dids
            ]
        )
        out[f"projection_{field}"] = {
            "high_higher_mean": float(
                np.mean([r[field] for r in shift if r["group"] == "high_higher"])
            ),
            "high_lower_mean": float(
                np.mean([r[field] for r in shift if r["group"] == "high_lower"])
            ),
            "difference": float(diff.mean()),
            "difference_ci": boot(lambda i: diff[i].mean(), len(diff), rng),
            "by_country": {
                c: float(np.mean([r[field] for r in shift if r["country"] == c]))
                for c in sorted({r["country"] for r in shift})
            },
            "by_item_group": {
                g: float(
                    np.mean(
                        [
                            np.mean(by_desc[k]["high_higher"])
                            - np.mean(by_desc[k]["high_lower"])
                            for k in dids
                            if C.item_group("BV", k.split("|")[1]) == g
                        ]
                    )
                )
                for g in cfg["boundaries"]["BV"]["item_groups"]
            },
        }
    pe = out["projection_end"]
    out["H2_projection"] = {
        "supported": bool(pe["difference_ci"][0] > 0 and pe["difference"] >= H2_SIZE),
        "falsified": bool(
            pe["difference_ci"][0] <= 0 <= pe["difference_ci"][1]
            or pe["difference"] < H2_SIZE
        ),
    }

    # steering (a): slope of s on alpha (in units of g), through the baseline score
    steer = [r for k, r in rows.items() if k.startswith("steer|")]
    base_s = {
        r["description_id"]: rows[f"proj|{r['description_id']}|{r['name']}"]["s"]
        for r in steer
    }
    alphas = cfg["steering"]["alphas"]
    A = np.array(alphas)
    slopes = np.array(
        [
            np.dot(
                A, [r["armed"][str(a)] - base_s[r["description_id"]] for a in alphas]
            )
            / np.dot(A, A)
            for r in steer
        ]
    )
    hh = [r for r in shift if r["group"] == "high_higher"]
    hl = [r for r in shift if r["group"] == "high_lower"]
    proj_diff = pe["difference"]
    delta_diff = float(
        np.mean([r["delta"] for r in hh]) - np.mean([r["delta"] for r in hl])
    )
    out["steering_armed"] = {
        "slope_nats_per_g": float(slopes.mean()),
        "slope_ci": boot(lambda i: slopes[i].mean(), len(slopes), rng),
        "by_alpha": {
            str(a): float(
                np.mean(
                    [r["armed"][str(a)] - base_s[r["description_id"]] for r in steer]
                )
            )
            for a in alphas
        },
        "delta_high_higher_minus_high_lower": delta_diff,
        "predicted_via_armed_axis": float(slopes.mean() * proj_diff),
        "share_explained": (
            float(slopes.mean() * proj_diff / delta_diff) if delta_diff else None
        ),
    }
    # steering (b): induced shift by layer, item profile vs observed delta profile
    items = cfg["boundaries"]["BV"]["items"]
    observed = {
        i: float(
            np.mean([r["delta"] for r in hh if r["item"] == i])
            - np.mean([r["delta"] for r in hl if r["item"] == i])
        )
        for i in items
    }
    country = {}
    for l in cfg["steering"]["country_layers"]:
        induced = {}
        for i in items:
            sel = [r for r in steer if r["description_id"].split("|")[1] == i]
            induced[i] = float(
                np.mean(
                    [
                        (r["country"][f"{l}|1.0"] - r["country"][f"{l}|-1.0"]) / 2
                        for r in sel
                    ]
                )
            )
        prof = [induced[i] for i in items]
        country[str(l)] = {
            "by_alpha": {
                str(a): float(
                    np.mean(
                        [
                            r["country"][f"{l}|{a}"] - base_s[r["description_id"]]
                            for r in steer
                        ]
                    )
                )
                for a in alphas
            },
            "induced_by_item": induced,
            "item_profile_r": float(
                np.corrcoef(prof, [observed[i] for i in items])[0, 1]
            ),
            "induced_sd_over_items": float(np.std(prof, ddof=1)),
            "induced_mean": float(np.mean(prof)),
        }
    out["steering_country"] = {"observed_delta_by_item": observed, "layers": country}
    write_json(OUT / "results.json", out)
    C.write_manifest(
        OUT,
        __file__,
        [C.OUTPUTS / "01_controls" / "controls.json"],
        [OUT / "results.json", OUT / "probe.json", OUT / "panel.jsonl"],
        {"H2_projection": out["H2_projection"]},
    )
    print(json.dumps({k: v for k, v in out.items() if k != "probe"}, indent=1)[:4000])


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="store_true")
    p.add_argument("--analyse", action="store_true")
    args = p.parse_args(argv)
    if args.run:
        run()
    if args.analyse:
        analyse()


if __name__ == "__main__":
    main()
