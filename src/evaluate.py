"""Apply the frozen geography analysis plan to a completed inference run."""

import argparse
import importlib.metadata
import json
import math
import tempfile
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from . import analysis, model_inputs
from .utils import (
    digest,
    file_digest,
    read_csv,
    read_json,
    require,
    verified_snapshot,
    write_csv,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
SENSITIVITIES = ("primary", "originally_correct", "unqualified_real")


def triple_records(stimuli, predictions, model):
    """One endpoint per independent source/base family, country and edit."""
    responses = {
        r["stimulus_id"]: r
        for r in predictions
        if r["model"] == model and r["replicate"] == "0"
    }
    require(
        len(responses)
        == sum(r["model"] == model and r["replicate"] == "0" for r in predictions),
        "Duplicate baseline response identity",
    )
    groups = defaultdict(dict)
    for row in stimuli:
        key = (row["cohort"], row["base_id"], row["country"])
        require(row["condition"] not in groups[key], "Duplicate input condition")
        groups[key][row["condition"]] = row
    triples = []
    for conditions in groups.values():
        original = conditions["original"]
        baseline = responses.get(original["stimulus_id"])
        neutral_input = conditions.get("neutral_control")
        neutral = responses.get(neutral_input["stimulus_id"]) if neutral_input else None
        for condition, row in conditions.items():
            if condition == "original":
                continue
            edited = responses.get(row["stimulus_id"])
            valid = all(
                record
                and record["status"] == "valid"
                and record["prediction"] in model_inputs.CLASSES
                for record in (baseline, neutral, edited)
            )
            flip = (
                int(edited["prediction"] != baseline["prediction"]) if valid else None
            )
            neutral_flip = (
                int(neutral["prediction"] != baseline["prediction"]) if valid else None
            )
            triples.append(
                {
                    "cohort": row["cohort"],
                    "base_id": row["base_id"],
                    "source_family_id": row["source_family_id"],
                    "country": row["country"],
                    "label": row["label"],
                    "population": (
                        f"real:{row['label']}"
                        if row["cohort"] == "real"
                        else "synthetic:all_eligible_classes"
                    ),
                    "condition": condition,
                    "prepared_triple": neutral_input is not None,
                    "edit_attempted": edited is not None,
                    "edit_valid": bool(edited and edited["status"] == "valid"),
                    "valid": bool(valid),
                    "d": flip - neutral_flip if valid else None,
                    "flip": flip,
                    "neutral_flip": neutral_flip,
                    "originally_correct": bool(
                        baseline
                        and baseline["status"] == "valid"
                        and baseline["prediction"] == original["label"]
                    ),
                    "qualified": original.get("label_support") == "qualified",
                    "original_prediction": (
                        baseline["prediction"]
                        if baseline and baseline["status"] == "valid"
                        else None
                    ),
                    "edit_prediction": (
                        edited["prediction"]
                        if edited and edited["status"] == "valid"
                        else None
                    ),
                }
            )
    return triples


def unit_maps(triples, sensitivity):
    maps = defaultdict(dict)
    for row in triples:
        if not row["valid"] or row["condition"] == "neutral_control":
            continue
        if sensitivity == "originally_correct" and not row["originally_correct"]:
            continue
        if sensitivity == "unqualified_real" and (
            row["cohort"] != "real" or row["qualified"]
        ):
            continue
        key = (row["population"], row["condition"], row["country"])
        require(
            row["source_family_id"] not in maps[key],
            "Repeated source/base family within an analysis cell",
        )
        maps[key][row["source_family_id"]] = row
    return maps


def bootstrap_means(values, rng, replicates):
    """Multinomial endpoint counts exactly implement the empirical family-mean bootstrap."""
    import numpy as np

    require(len(values) > 0, "Cannot bootstrap an empty family sample")
    values = np.asarray(values, dtype=float)
    unique, counts = np.unique(values, return_counts=True)
    draws = rng.multinomial(len(values), counts / len(values), size=replicates)
    return draws @ unique / len(values)


def mean_test(a, b, paired):
    """Approximate mean tests; a constant contrast has no manufactured p-value."""
    import numpy as np
    from scipy.stats import t

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    effect = float(a.mean() - b.mean())
    if paired:
        require(len(a) == len(b), "Paired samples differ in length")
        variance = float(np.var(a - b, ddof=1)) / len(a)
        df = len(a) - 1
    else:
        va, vb = float(np.var(a, ddof=1)) / len(a), float(np.var(b, ddof=1)) / len(b)
        variance = va + vb
        df = (
            variance**2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
            if variance
            else None
        )
    if variance == 0:
        return effect, None, "zero_contrast_variance"
    p = float(2 * t.sf(abs(effect / math.sqrt(variance)), df))
    require(math.isfinite(p) and 0 <= p <= 1, "Invalid test p-value")
    return effect, p, "tested"


def holm(pvalues):
    """Unavailable planned tests remain in the family internally at p=1."""
    result, previous = [None] * len(pvalues), 0.0
    for rank, index in enumerate(
        sorted(
            range(len(pvalues)),
            key=lambda i: (pvalues[i] if pvalues[i] is not None else 1.0, i),
        )
    ):
        value = pvalues[index] if pvalues[index] is not None else 1.0
        require(math.isfinite(value) and 0 <= value <= 1, "Invalid planned p-value")
        previous = max(previous, min(1.0, (len(pvalues) - rank) * value))
        if pvalues[index] is not None:
            result[index] = previous
    return result


def estimate(contrast, units, plan, rng, sensitivity):
    import numpy as np

    population, condition = contrast["population"], contrast["condition"]
    a = units.get((population, condition, contrast["country_a"]), {})
    b = units.get((population, condition, contrast["country_b"]), {})
    paired = contrast["test"] == "paired_t"
    if paired:
        families = sorted(set(a) & set(b))
        require(
            all(a[key]["label"] == b[key]["label"] for key in families),
            "Matched base source classes differ",
        )
        a, b = {key: a[key] for key in families}, {key: b[key] for key in families}
    va = np.asarray([a[key]["d"] for key in sorted(a)], dtype=float)
    vb = np.asarray([b[key]["d"] for key in sorted(b)], dtype=float)
    row = {
        **contrast,
        "sensitivity": sensitivity,
        "valid_n_a": len(va),
        "valid_n_b": len(vb),
        "valid_class_counts_a": json.dumps(
            dict(Counter(r["label"] for r in a.values())), sort_keys=True
        ),
        "valid_class_counts_b": json.dumps(
            dict(Counter(r["label"] for r in b.values())), sort_keys=True
        ),
        "mean_d_a": float(va.mean()) if len(va) else None,
        "mean_d_b": float(vb.mean()) if len(vb) else None,
        "effect": float(va.mean() - vb.mean()) if len(va) and len(vb) else None,
        "ci_low": None,
        "ci_high": None,
        "p_value": None,
        "p_holm": None,
        "test_status": "insufficient_valid_families",
        "valid_families_sha256": digest(
            "\n".join(sorted(set(a) & set(b) if paired else set(a) | set(b)))
        ),
    }
    if min(len(va), len(vb)) < plan["minimum_independent_families"]:
        return row
    replicates = plan["uncertainty"]["replicates"]
    if paired:
        grouped = defaultdict(list)
        for key in sorted(a):
            grouped[a[key]["label"]].append(a[key]["d"] - b[key]["d"])
        draws = sum(
            bootstrap_means(values, rng, replicates) * len(values)
            for _, values in sorted(grouped.items())
        ) / len(va)
    else:
        draws = bootstrap_means(va, rng, replicates) - bootstrap_means(
            vb, rng, replicates
        )
    alpha = 1 - plan["uncertainty"]["confidence_level"]
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    row.update(ci_low=float(low), ci_high=float(high))
    if sensitivity == "primary":
        row["effect"], row["p_value"], row["test_status"] = mean_test(va, vb, paired)
    else:
        row["test_status"] = "secondary_descriptive"
    return row


def country_edits(triples, model):
    groups = defaultdict(list)
    for row in triples:
        groups[(row["population"], row["country"], row["condition"])].append(row)
    records = []
    for (population, country, condition), rows in sorted(groups.items()):
        valid = [r for r in rows if r["valid"]]
        attempted = sum(r["edit_attempted"] for r in rows)
        valid_edit = sum(r["edit_valid"] for r in rows)
        records.append(
            {
                "model": model,
                "population": population,
                "country": country,
                "condition": condition,
                "prepared_n": len(rows),
                "prepared_class_counts": json.dumps(
                    dict(Counter(r["label"] for r in rows)), sort_keys=True
                ),
                "attempted_n": attempted,
                "valid_edit_n": valid_edit,
                "invalid_edit_n": attempted - valid_edit,
                "invalid_edit_rate": (
                    (attempted - valid_edit) / attempted if attempted else None
                ),
                "valid_triples_n": len(valid),
                "valid_class_counts": json.dumps(
                    dict(Counter(r["label"] for r in valid)), sort_keys=True
                ),
                "flip_rate": (
                    sum(r["flip"] for r in valid) / len(valid) if valid else None
                ),
                "neutral_flip_rate": (
                    sum(r["neutral_flip"] for r in valid) / len(valid)
                    if valid
                    else None
                ),
                "mean_excess_flip": (
                    sum(r["d"] for r in valid) / len(valid) if valid else None
                ),
            }
        )
    return records


def baseline_and_swaps(stimuli, predictions, model):
    response = {
        r["stimulus_id"]: r
        for r in predictions
        if r["model"] == model and r["replicate"] == "0"
    }
    groups, bases, prepared_bases = (
        defaultdict(list),
        defaultdict(dict),
        defaultdict(set),
    )
    for row in stimuli:
        if row["condition"] != "original":
            continue
        record = response.get(row["stimulus_id"])
        groups[(row["cohort"], row["country"], row["label"])].append((row, record))
        if row["cohort"] == "synthetic":
            prepared_bases[row["source_family_id"]].add(row["country"])
        if row["cohort"] == "synthetic" and record and record["status"] == "valid":
            bases[row["source_family_id"]][row["country"]] = record["prediction"]
    accuracy = []
    for (cohort, country, label), pairs in sorted(groups.items()):
        valid = [
            (row, result)
            for row, result in pairs
            if result and result["status"] == "valid"
        ]
        correct = sum(row["label"] == result["prediction"] for row, result in valid)
        accuracy.append(
            {
                "model": model,
                "cohort": cohort,
                "country": country,
                "label": label,
                "prepared_n": len(pairs),
                "attempted_n": sum(result is not None for _, result in pairs),
                "valid_n": len(valid),
                "correct": correct,
                "accuracy": correct / len(valid) if valid else None,
            }
        )
    countries = sorted(
        {row["country"] for row in stimuli if row["cohort"] == "synthetic"}
    )
    swaps = []
    for a, b in combinations(countries, 2):
        values = [
            labels[a] != labels[b]
            for labels in bases.values()
            if a in labels and b in labels
        ]
        swaps.append(
            {
                "model": model,
                "country_a": a,
                "country_b": b,
                "prepared_pairs": sum(
                    a in countries and b in countries
                    for countries in prepared_bases.values()
                ),
                "valid_pairs": len(values),
                "disagreements": sum(values),
                "disagreement_rate": sum(values) / len(values) if values else None,
            }
        )
    return accuracy, swaps


def heatmap(rows, models, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    selected = [
        r
        for r in rows
        if r["population"] == "synthetic:all_eligible_classes"
        and r["condition"] != "neutral_control"
    ]
    countries = sorted({r["country"] for r in selected})
    conditions = sorted({r["condition"] for r in selected})
    figure, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
    for axis, model in zip(axes, models):
        lookup = {
            (r["condition"], r["country"]): r for r in selected if r["model"] == model
        }
        data = np.full((len(conditions), len(countries)), np.nan)
        for y, condition in enumerate(conditions):
            for x, country in enumerate(countries):
                row = lookup[condition, country]
                if row["mean_excess_flip"] is not None:
                    data[y, x] = row["mean_excess_flip"] * 100
                    axis.text(
                        x,
                        y,
                        f"{data[y, x]:.0f}\n(n={row['valid_triples_n']})",
                        ha="center",
                        va="center",
                        fontsize=7,
                    )
        image = axis.imshow(data, cmap="RdBu_r", vmin=-100, vmax=100, aspect="auto")
        axis.set_xticks(range(len(countries)), countries, rotation=35, ha="right")
        axis.set_yticks(
            range(len(conditions)), [c.replace("_", " ") for c in conditions]
        )
        axis.set_title(model)
        figure.colorbar(
            image,
            ax=axis,
            label="Excess flips beyond neutral rewrite (percentage points)",
        )
    figure.suptitle("Matched descriptions: descriptive country-by-edit sensitivity")
    figure.savefig(output_dir / "country_edits.png", dpi=180)
    plt.close(figure)


def validate_prediction_inventory(stimuli, predictions, schedule):
    require(
        Counter(
            (r["request_id"], r["model"], r["stimulus_id"], r["replicate"])
            for r in predictions
        )
        == Counter(
            (r["request_id"], r["model"], r["stimulus_id"], r["replicate"])
            for r in schedule
        ),
        "Prediction identities differ from the frozen schedule",
    )
    sources = {r["stimulus_id"]: r for r in stimuli}
    for row in predictions:
        source = sources[row["stimulus_id"]]
        require(
            all(
                row[key] == source[key]
                for key in (
                    "source_family_id",
                    "cohort",
                    "base_id",
                    "country",
                    "condition",
                    "text_sha256",
                )
            )
            and row["expected_label"] == source["label"],
            "Prediction provenance or reference label changed",
        )
        require(
            (row["status"] == "valid" and row["prediction"] in model_inputs.CLASSES)
            or (row["status"] != "valid" and row["prediction"] == ""),
            "Missing responses cannot be interpreted as labels",
        )


def evaluate(run_dir, output_dir):
    import numpy as np

    require(not output_dir.exists(), "Use an unused analysis directory")
    run = read_json(run_dir / "manifest.json")
    require(run["complete"], "Complete the frozen schedule before analysis")
    run = verified_snapshot(run_dir)
    require(
        file_digest(ROOT / "data/experiment/manifest.json")
        == run["inputs"]["experiment_manifest_sha256"]
        and file_digest(ROOT / "data/experiment/protocol.json")
        == run["inputs"]["protocol_sha256"],
        "Adopted experiment protocol changed",
    )
    plan = read_json(ROOT / "data/analysis_plan.json")
    require(
        file_digest(ROOT / "data/analysis_plan.json")
        == run["inputs"]["analysis_plan_sha256"],
        "Analysis registry changed",
    )
    verified_snapshot(ROOT / "data/variants/main")
    for name, key in (
        ("manifest.json", "main_manifest_sha256"),
        ("stimuli.csv", "stimuli_sha256"),
        ("protocol.json", "protocol_sha256"),
    ):
        require(
            file_digest(ROOT / "data/variants/main" / name) == plan["inputs"][key],
            f"Prespecified source binding changed: {name}",
        )
    stimuli = read_csv(ROOT / "data/variants/main/stimuli.csv")
    require(
        analysis.planned_contrasts(stimuli) == plan["primary_contrasts"],
        "Prepared primary contrast registry changed",
    )
    predictions = read_csv(run_dir / "predictions.csv")
    require(
        len(predictions) == run["completed_classifications"],
        "Prediction inventory differs from run accounting",
    )
    schedule = read_csv(ROOT / "data/variants/main/request_order.csv")
    require(
        file_digest(ROOT / "data/variants/main/request_order.csv")
        == run["inputs"]["schedule_sha256"],
        "Run schedule binding changed",
    )
    validate_prediction_inventory(stimuli, predictions, schedule)
    adopted = read_json(ROOT / "data/experiment/protocol.json")
    models = [adopted["encoder_model"], adopted["decoder_model"]]
    contrasts, descriptives, accuracy, swaps, transitions, repeats = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for model_index, model in enumerate(models):
        triples = triple_records(stimuli, predictions, model)
        descriptives.extend(country_edits(triples, model))
        baseline, disagreements = baseline_and_swaps(stimuli, predictions, model)
        accuracy.extend(baseline)
        swaps.extend(disagreements)
        for sensitivity_index, sensitivity in enumerate(SENSITIVITIES):
            units = unit_maps(triples, sensitivity)
            current = []
            for index, contrast in enumerate(plan["primary_contrasts"]):
                if sensitivity == "unqualified_real" and not contrast[
                    "population"
                ].startswith("real:"):
                    continue
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [
                            plan["uncertainty"]["seed"],
                            model_index,
                            sensitivity_index,
                            index,
                        ]
                    )
                )
                current.append(
                    {
                        "model": model,
                        **estimate(contrast, units, plan, rng, sensitivity),
                    }
                )
            if sensitivity == "primary":
                require(len(current) == 438, "Primary multiplicity family changed")
                for row, adjusted in zip(
                    current, holm([r["p_value"] for r in current])
                ):
                    row["p_holm"] = adjusted
            contrasts.extend(current)
        counts = Counter(
            (
                r["cohort"],
                r["country"],
                r["condition"],
                r["original_prediction"],
                r["edit_prediction"],
            )
            for r in triples
            if r["valid"]
        )
        for (cohort, country, condition, source, target), count in sorted(
            counts.items()
        ):
            transitions.append(
                {
                    "model": model,
                    "cohort": cohort,
                    "country": country,
                    "condition": condition,
                    "from_label": source,
                    "to_label": target,
                    "count": count,
                }
            )
        originals = {
            r["stimulus_id"]: r
            for r in predictions
            if r["model"] == model and r["replicate"] == "0"
        }
        for row in predictions:
            if row["model"] != model or row["replicate"] == "0":
                continue
            original = originals[row["stimulus_id"]]
            valid = row["status"] == original["status"] == "valid"
            repeats.append(
                {
                    "model": model,
                    "stimulus_id": row["stimulus_id"],
                    "both_valid": valid,
                    "unchanged": (
                        row["prediction"] == original["prediction"] if valid else None
                    ),
                }
            )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".analysis-", dir=output_dir.parent
    ) as directory:
        stage = Path(directory) / "export"
        stage.mkdir()
        tables = {
            "contrasts.csv": contrasts,
            "country_edits.csv": descriptives,
            "baseline_accuracy.csv": accuracy,
            "country_swaps.csv": swaps,
            "transitions.csv": transitions,
            "repeats.csv": repeats,
        }
        for name, rows in tables.items():
            fields = (
                list(rows[0])
                if rows
                else [
                    "model",
                    "cohort",
                    "country",
                    "condition",
                    "from_label",
                    "to_label",
                    "count",
                ]
            )
            write_csv(stage / name, fields, rows)
        heatmap(descriptives, models, stage)
        alpha = 1 - plan["uncertainty"]["confidence_level"]
        lines = [
            "# Geography Analysis",
            "",
            "Each model is analyzed separately under the frozen registry. Effects measure excess prediction flips beyond neutral rewriting. Positive country contrasts indicate greater disruption in country A; they do not necessarily indicate accuracy loss.",
            "",
            "| Model | Primary tests available | Holm-adjusted detections |",
            "| --- | --- | --- |",
        ]
        summary = {}
        for model in models:
            primary = [
                r
                for r in contrasts
                if r["model"] == model and r["sensitivity"] == "primary"
            ]
            detections = [
                r for r in primary if r["p_holm"] is not None and r["p_holm"] <= alpha
            ]
            country_rates = [
                r["disagreement_rate"]
                for r in swaps
                if r["model"] == model and r["disagreement_rate"] is not None
            ]
            summary[model] = {
                "planned_primary_contrasts": len(primary),
                "available_tests": sum(r["p_value"] is not None for r in primary),
                "holm_detections": len(detections),
                "baseline": {
                    cohort: {
                        "prepared_n": sum(
                            r["prepared_n"]
                            for r in accuracy
                            if r["model"] == model and r["cohort"] == cohort
                        ),
                        "valid_n": sum(
                            r["valid_n"]
                            for r in accuracy
                            if r["model"] == model and r["cohort"] == cohort
                        ),
                        "correct": sum(
                            r["correct"]
                            for r in accuracy
                            if r["model"] == model and r["cohort"] == cohort
                        ),
                    }
                    for cohort in ("real", "synthetic")
                },
                "repeats": {
                    "both_valid": sum(
                        r["both_valid"] for r in repeats if r["model"] == model
                    ),
                    "unchanged": sum(
                        r["unchanged"] is True for r in repeats if r["model"] == model
                    ),
                },
                "country_only_disagreement_range": (
                    [min(country_rates), max(country_rates)] if country_rates else None
                ),
            }
            lines.append(
                f"| {model} | {summary[model]['available_tests']}/438 | {len(detections)} |"
            )
        lines += [
            "",
            "| Model | Original cohort | Correct/valid labels | Prepared originals |",
            "| --- | --- | --- | --- |",
        ]
        for model in models:
            for cohort, values in summary[model]["baseline"].items():
                lines.append(
                    f"| {model} | {cohort} | {values['correct']}/{values['valid_n']} | {values['prepared_n']} |"
                )
        for model in models:
            values = summary[model]["repeats"]
            lines += [
                "",
                f"Repeat labels for `{model}`: {values['unchanged']}/{values['both_valid']} unchanged among valid repeat pairs.",
            ]
            strongest = sorted(
                (
                    r
                    for r in contrasts
                    if r["model"] == model
                    and r["sensitivity"] == "primary"
                    and r["ci_low"] is not None
                    and r["effect"] != 0
                ),
                key=lambda r: (-abs(r["effect"]), r["id"]),
            )[:3]
            if strongest:
                lines += [
                    "",
                    f"Largest absolute primary estimates for `{model}` (illustrative selection; intervals are nominal):",
                    "",
                    "| Population/edit | A versus B | Families A/B | Effect, percentage points [95% CI] | Holm p |",
                    "| --- | --- | --- | --- | --- |",
                ]
                for row in strongest:
                    adjusted = (
                        f"{row['p_holm']:.4g}"
                        if row["p_holm"] is not None
                        else "unavailable"
                    )
                    lines.append(
                        f"| {row['population']}/{row['condition']} | {row['country_a']} vs {row['country_b']} | {row['valid_n_a']}/{row['valid_n_b']} | {100*row['effect']:.1f} [{100*row['ci_low']:.1f}, {100*row['ci_high']:.1f}] | {adjusted} |"
                    )
        lines += [
            "",
            "![Country-by-edit sensitivity](country_edits.png)",
            "",
            "`contrasts.csv` contains all planned comparisons, nominal family-bootstrap intervals and unavailable-test reasons. Holm retains all 438 planned tests per model. The originally-correct and unqualified-real sensitivities are descriptive and have no additional significance claims. All real Riots and actor-identity country contrasts remain descriptive at the frozen ten-family threshold.",
            "",
            "`country_edits.csv` retains prepared, attempted and valid denominators and source-class composition; `contrasts.csv` records the classes in each valid comparison. Synthetic intensity targets Battles only; other edit families pool their eligible source classes. `baseline_accuracy.csv`, `country_swaps.csv`, `transitions.csv` and `repeats.csv` provide secondary diagnostics. Synthetic country-only disagreement is evaluated on matched original descriptions. Repeats do not add independent events.",
            "",
            "Tests use the prespecified approximate Welch or paired mean procedures. A zero-variance contrast has no p-value. Intervals use 10,000 empirical family-bootstrap replicates, retaining joint original/edit/neutral endpoints and matched countries; synthetic draws retain source-class counts. Child seeds derive from seed 42 and fixed model/sensitivity/registry indices. Intervals are nominal, not simultaneous.",
            "",
            "Real-event country comparisons are observational and remain confounded by sources and event characteristics. Matched descriptions provide cleaner country-name evidence within this constructed dataset. Curated, balanced data do not estimate operational accuracy; E5's validation limitations remain relevant. Missing responses may bias complete-case effects. Lack of significance does not establish geographic invariance.",
        ]
        (stage / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        write_json(stage / "summary.json", summary)
        manifest = {
            "inputs": {
                "run_manifest_sha256": file_digest(run_dir / "manifest.json"),
                "registry_sha256": file_digest(ROOT / "data/analysis_plan.json"),
                "stimuli_sha256": file_digest(ROOT / "data/variants/main/stimuli.csv"),
                "evaluation_code_sha256": file_digest(Path(__file__)),
            },
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "matplotlib")
            },
            "bootstrap_replicates": plan["uncertainty"]["replicates"],
            "seed": plan["uncertainty"]["seed"],
            "outputs": {
                path.name: {"sha256": file_digest(path)} for path in stage.iterdir()
            },
        }
        write_json(stage / "manifest.json", manifest)
        stage.rename(output_dir)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/main")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/main")
    args = parser.parse_args()
    print(evaluate(args.run_dir, args.output_dir))


if __name__ == "__main__":
    main()
