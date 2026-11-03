"""Freeze geography contrasts and analysis rules before main inference."""

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from .utils import digest, file_digest, read_csv, require, verified_snapshot, write_json

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "data/variants/main"
MINIMUM_FAMILIES = 10


def planned_contrasts(rows):
    """Use common original/edit/neutral eligibility, never response outcomes."""
    groups = defaultdict(dict)
    for row in rows:
        key = (row["cohort"], row["base_id"], row["country"])
        require(row["condition"] not in groups[key], "Duplicate stimulus condition")
        groups[key][row["condition"]] = row
    units = defaultdict(set)
    for conditions in groups.values():
        if not {"original", "neutral_control"} <= conditions.keys():
            continue
        original = conditions["original"]
        for condition, row in conditions.items():
            if condition in {"original", "neutral_control"}:
                continue
            require(
                row["label"] == original["label"]
                and row["source_family_id"] == original["source_family_id"],
                "Stimulus family or label changed",
            )
            population = (
                f"real:{row['label']}"
                if row["cohort"] == "real"
                else "synthetic:all_eligible_classes"
            )
            units[(population, condition, row["country"])].add(row["source_family_id"])
    contrasts = []
    families = sorted({(population, condition) for population, condition, _ in units})
    for population, condition in families:
        countries = sorted(
            country
            for pop, edit, country in units
            if (pop, edit) == (population, condition)
        )
        for country_a, country_b in combinations(countries, 2):
            a, b = (
                units[(population, condition, country_a)],
                units[(population, condition, country_b)],
            )
            paired = population.startswith("synthetic:")
            eligible = a & b if paired else a | b
            if (len(eligible) if paired else min(len(a), len(b))) < MINIMUM_FAMILIES:
                continue
            contrasts.append(
                {
                    "id": f"{population}:{condition}:{country_a}:{country_b}",
                    "population": population,
                    "condition": condition,
                    "country_a": country_a,
                    "country_b": country_b,
                    "prepared_n_a": len(a & b) if paired else len(a),
                    "prepared_n_b": len(a & b) if paired else len(b),
                    "test": "paired_t" if paired else "welch_t",
                    "eligible_families_sha256": digest("\n".join(sorted(eligible))),
                }
            )
    return contrasts


def freeze_plan(output_file, main_dir=MAIN):
    require(not output_file.exists(), "Analysis plan already exists; use a new path")
    verified_snapshot(main_dir)
    contrasts = planned_contrasts(read_csv(main_dir / "stimuli.csv"))
    require(
        sum(c["population"].startswith("real:") for c in contrasts) == 168
        and len(contrasts) == 438,
        "Prepared contrast eligibility changed; review the design",
    )
    plan = {
        "format_version": 1,
        "main_inference_performed": False,
        "inputs": {
            "main_manifest_sha256": file_digest(main_dir / "manifest.json"),
            "stimuli_sha256": file_digest(main_dir / "stimuli.csv"),
            "protocol_sha256": file_digest(main_dir / "protocol.json"),
            "planner_sha256": file_digest(Path(__file__)),
        },
        "models_analyzed_separately": True,
        "endpoint": "d = I(edit prediction != original prediction) - I(neutral prediction != original prediction)",
        "contrast_direction": "mean(d in country_a) minus mean(d in country_b); synthetic differences are paired by base description",
        "minimum_independent_families": MINIMUM_FAMILIES,
        "real_primary_contrasts_by_class": {"P": 123, "B": 36, "V": 9, "R": 0},
        "tests": {
            "alternative": "two_sided",
            "real": "Approximate Welch mean test within each source class; observational country comparisons",
            "synthetic": "Approximate paired t test on within-base differences, pooled over eligible source classes",
            "zero_variance": "Welch unavailable only when both country sample variances are zero; paired t unavailable when the variance of paired differences is zero. Display unavailable p-values rather than manufacturing significance.",
            "multiplicity": "Holm across all 438 planned contrasts separately per model; unavailable tests retain their positions in the family using internal p=1 and display unavailable",
        },
        "uncertainty": {
            "method": "nominal percentile family bootstrap",
            "replicates": 10000,
            "seed": 42,
            "confidence_level": 0.95,
            "real_strata": "country and source class",
            "synthetic_strata": "source class",
            "resampling_unit": "source/base family; retain all its countries, edits and neutral versions together",
            "eligibility": "For each contrast, resample its valid common-triple families; in B preserve the observed eligible-base count within each source class",
            "simultaneous_intervals": False,
        },
        "missingness": "Require valid original/edit/neutral triples; synthetic contrasts require both countries for the same base. Reapply the ten-family threshold after failures without adding contrasts. Report prepared, attempted and valid denominators and invalid rates by country and edit. Missing responses are not labels.",
        "secondary_analyses": [
            "Baseline accuracy, edit flip rates and label transitions",
            "Originally correct baseline subset; synthetic pairs require correctness in both countries",
            "Real-event sensitivity excluding qualified labels",
            "Synthetic original country-only disagreement across all 45 country pairs",
            "Sparse edits, including all real Riots country contrasts and real actor identity, remain descriptive",
            "Repeats describe stability and never increase sample size",
        ],
        "interpretation": "Exploratory mean tests with small discrete samples and limited power after multiplicity adjustment. Effects are sensitivity, not necessarily accuracy loss. Failure to reject does not establish geographic invariance. Complete-case estimates may be biased by differential failures. No model or architecture comparison.",
        "primary_contrasts": contrasts,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_file, plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-file", type=Path, default=ROOT / "data/analysis_plan.json"
    )
    args = parser.parse_args()
    plan = freeze_plan(args.output_file)
    print(
        f"Frozen {len(plan['primary_contrasts'])} primary contrasts per model; no inference performed."
    )


if __name__ == "__main__":
    main()
