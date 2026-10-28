#!/usr/bin/env python3
"""Build reviewed pilot stimuli from explicit, versioned edit recipes."""

import argparse
import json
import re
import sys
import tempfile
from collections import Counter
from datetime import date
from pathlib import Path

import prepare_data as prep
from finalize_inputs import (
    FAMILIES,
    read_csv,
    require,
    similarity_hits,
    verified_snapshot,
)

ROOT = Path(__file__).resolve().parent
RECIPES = ROOT / "data/reviews/pilot_variants.json"
MONTHS = "January February March April May June July August September October November December".split()
STIMULUS_FIELDS = [
    "stimulus_id",
    "cohort",
    "base_id",
    "source_event_id",
    "source_family_id",
    "source_file",
    "source_record",
    "country",
    "label",
    "condition",
    "original_id",
    "text",
    "text_sha256",
    "case_id",
    "case_sha256",
    "review_status",
]
CHANGE_FIELDS = [
    "stimulus_id",
    "original_id",
    "original_start",
    "original_end",
    "original_span",
    "edited_start",
    "edited_end",
    "edited_span",
]
OMISSION_FIELDS = [
    "cohort",
    "base_id",
    "source_event_id",
    "condition",
    "country_versions",
    "reason",
]


def apply_edits(text, edits):
    """Resolve every anchor against the original; never chain interventions."""
    require(bool(edits), "An edited condition needs at least one edit")
    spans = []
    for edit in edits:
        old, new = edit["old"], edit["new"]
        require(
            isinstance(old, str) and old and isinstance(new, str) and old != new,
            "Empty/no-op edit",
        )
        matches = list(re.finditer(re.escape(old), text))
        occurrence = edit.get("occurrence")
        require(bool(matches), f"Edit anchor is absent: {old!r}")
        if occurrence is None:
            require(len(matches) == 1, f"Ambiguous edit anchor: {old!r}")
            occurrence = 1
        require(
            type(occurrence) is int and 1 <= occurrence <= len(matches),
            "Invalid anchor occurrence",
        )
        match = matches[occurrence - 1]
        spans.append((match.start(), match.end(), old, new))
    spans.sort()
    parts, changes, cursor, offset = [], [], 0, 0
    for start, end, old, new in spans:
        require(start >= cursor, "Overlapping edit spans")
        parts.extend((text[cursor:start], new))
        changes.append(
            {
                "original_start": start,
                "original_end": end,
                "original_span": old,
                "edited_start": start + offset,
                "edited_end": start + offset + len(new),
                "edited_span": new,
            }
        )
        cursor, offset = end, offset + len(new) - len(old)
    parts.append(text[cursor:])
    return "".join(parts), changes


def parsed_date(value):
    for pattern in (
        r"(\d{4})-(\d{2})-(\d{2})",
        r"(\d{1,2}) ([A-Za-z]+) (\d{4})",
        r"([A-Za-z]+) (\d{1,2}), (\d{4})",
    ):
        match = re.fullmatch(pattern, value)
        if not match:
            continue
        a, b, c = match.groups()
        if pattern.startswith(r"(\d{4})"):
            return date(int(a), int(b), int(c))
        if a.isdigit():
            return date(int(c), MONTHS.index(b) + 1, int(a))
        return date(int(c), MONTHS.index(a) + 1, int(b))
    raise ValueError(f"Unrecognized date-only control: {value!r}")


def validate_conditions(recipe):
    conditions, omissions = recipe["conditions"], recipe["omissions"]
    require(
        not set(conditions) & set(omissions)
        and set(conditions) | set(omissions) == set(FAMILIES),
        "Every edit family needs exactly one recipe or explicit omission",
    )
    require(
        all(
            isinstance(reason, str) and reason.strip() for reason in omissions.values()
        ),
        "Missing omission reason",
    )
    for family, condition in conditions.items():
        require(bool(condition["rationale"]), f"Missing design rationale: {family}")
        if family == "neutral_control":
            require(
                len(condition["edits"]) == 1, "Neutral control changes more than a date"
            )
            edit = condition["edits"][0]
            require(
                parsed_date(edit["old"]) == parsed_date(edit["new"]),
                "Neutral control changes the date",
            )


def fingerprint(case):
    return prep.digest(json.dumps(case, sort_keys=True, ensure_ascii=False))


def render_case(
    cohort,
    base_id,
    source,
    label,
    template,
    family,
    recipe,
    countries,
    constraints,
    derivation,
):
    edits = recipe["edits"] if family != "original" else []
    edited_template = apply_edits(template, edits)[0] if edits else template
    if cohort == "synthetic":
        require(
            template.count("{country}") == edited_template.count("{country}") == 1,
            "Synthetic conditions must retain exactly one country placeholder",
        )
    renderings = []
    for country in countries:
        original = (
            template.replace("{country}", country)
            if cohort == "synthetic"
            else template
        )
        rendered_edits = [
            {
                **edit,
                "old": edit["old"].replace("{country}", country),
                "new": edit["new"].replace("{country}", country),
            }
            for edit in edits
        ]
        text, changes = (
            apply_edits(original, rendered_edits) if edits else (original, [])
        )
        expected = (
            edited_template.replace("{country}", country)
            if cohort == "synthetic"
            else edited_template
        )
        require(
            text == expected, "Country rendering and edit application do not commute"
        )
        renderings.append(
            {
                "country": country,
                "original": original,
                "edited": text,
                "changes": changes,
            }
        )
    case = {
        "case_id": f"{cohort}:{base_id}:{family}",
        "cohort": cohort,
        "base_id": base_id,
        "source_event_id": source["event_id_cnty"],
        "source_family_id": source["event_group_id"],
        "source_file": source["source_file"],
        "source_record": source["source_record"],
        "label": label,
        "condition": family,
        "rationale": recipe["rationale"],
        "constraints": constraints,
        "derivation_note": derivation,
        "renderings": renderings,
    }
    if cohort == "synthetic" and family == "original":
        case["source_notes"] = source["notes"]
    return {**case, "case_sha256": fingerprint(case)}


def draft_cases(recipes, input_dir):
    require(
        recipes["format_version"] == 1 and bool(recipes["recipe_version"]),
        "Unsupported recipe version",
    )
    verified_snapshot(input_dir)
    pilot = read_csv(input_dir / "pilot.csv")
    require(len(pilot) == 28, "Expected the accepted 28-event pilot")
    by_id = {r["event_id_cnty"]: r for r in pilot}
    require(
        len(recipes["real"]) == len(by_id)
        and {r["event_id"] for r in recipes["real"]} == set(by_id),
        "Real recipes must cover each accepted pilot input once",
    )
    countries = recipes["countries"]
    require(
        len(countries) == 4 and set(countries) == {r["country"] for r in pilot},
        "Pilot country coverage changed",
    )
    input_review = json.loads(
        (ROOT / "data/reviews/input_review.json").read_text(encoding="utf-8")
    )
    restrictions = {r["event_id"]: r for r in input_review["cases"]}
    eligibility = {
        (r["event_id"], r["family"]): r["status"]
        for r in read_csv(input_dir / "edit_eligibility.csv")
    }
    cases, baselines, omissions = [], [], []
    for recipe in recipes["real"]:
        validate_conditions(recipe)
        source = by_id[recipe["event_id"]]
        event_id = source["event_id_cnty"]
        semantic = restrictions[event_id]["semantic"]
        base = render_case(
            "real",
            event_id,
            source,
            source["label"],
            source["notes"],
            "original",
            {"rationale": "Unmodified accepted source note."},
            [source["country"]],
            [],
            "",
        )
        baselines.append(base)
        for family, condition in recipe["conditions"].items():
            require(
                eligibility[event_id, family] != "excluded_from_initial_batch",
                f"Ineligible real edit: {event_id}/{family}",
            )
            constraints = [
                input_review["family_policy"][family],
                *restrictions[event_id]["blind"]["constraints"],
            ]
            for key in (
                "case_constraints",
                (
                    "actor_identity_constraint"
                    if family == "actor_identity"
                    else (
                        "synthetic_source_constraint"
                        if family == "source_attribution"
                        else "framing_constraint"
                    )
                ),
            ):
                if semantic.get(key):
                    constraints.append(semantic[key])
            specific = semantic.get("per_family_restrictions", {}).get(family)
            if specific:
                constraints.append(specific)
            cases.append(
                render_case(
                    "real",
                    event_id,
                    source,
                    source["label"],
                    source["notes"],
                    family,
                    condition,
                    [source["country"]],
                    constraints,
                    "",
                )
            )
        for family, reason in recipe["omissions"].items():
            omissions.append(
                {
                    "cohort": "real",
                    "base_id": event_id,
                    "source_event_id": event_id,
                    "condition": family,
                    "country_versions": 1,
                    "reason": reason,
                }
            )
    synthetic = recipes["synthetic"]
    require(
        len(synthetic) == 12 and len({r["base_id"] for r in synthetic}) == 12,
        "Expected 12 distinct synthetic bases",
    )
    require(
        Counter(r["label"] for r in synthetic)
        == Counter({label: 2 for label in prep.LABELS.values()}),
        "Synthetic pilot needs two bases in each of six classes",
    )
    require(
        len({r["source_event_id"] for r in synthetic}) == 12,
        "Synthetic sources must be distinct",
    )
    source_manifest = verified_snapshot(ROOT / "data/processed")
    sources = dict(by_id)
    needed = {(r["source_file"], str(r["source_record"])) for r in synthetic}
    for info in source_manifest["inputs"]:
        path = ROOT / "conflict_data" / info["file"]
        require(
            prep.file_digest(path) == info["sha256"], f"Raw source changed: {path.name}"
        )
        for ordinal, row in enumerate(read_csv(path), 1):
            if (path.name, str(ordinal)) in needed and row[
                "event_id_cnty"
            ] not in sources:
                sources[row["event_id_cnty"]] = {
                    **row,
                    "source_file": path.name,
                    "source_record": str(ordinal),
                    "event_group_id": "group_" + prep.digest(row["event_id_cnty"])[:16],
                }
    for pair in read_csv(ROOT / "data/processed/related_event_pairs.csv"):
        for key in ("event_id_1", "event_id_2"):
            if pair[key] in sources:
                require(
                    pair[key] not in by_id
                    or sources[pair[key]]["event_group_id"] == pair["event_group_id"],
                    "Accepted source family changed",
                )
                sources[pair[key]]["event_group_id"] = pair["event_group_id"]
    allocated = [
        r
        for name in ("candidates.csv", "pilot.csv", "reserves.csv")
        for r in read_csv(ROOT / "data/processed" / name)
    ]
    blocked = {r["event_group_id"] for r in allocated + pilot}
    source_groups = set()
    additional_sources = []
    for recipe in synthetic:
        validate_conditions(recipe)
        source = sources[recipe["source_event_id"]]
        require(
            source["source_file"] == recipe["source_file"]
            and source["source_record"] == str(recipe["source_record"]),
            "Synthetic source ordinal mismatch",
        )
        require(
            prep.LABELS[source["event_type"]] == recipe["label"],
            "Synthetic source class mismatch",
        )
        group = source["event_group_id"]
        require(group not in source_groups, "Synthetic bases reuse a source family")
        source_groups.add(group)
        require(
            source["event_id_cnty"] in by_id or group not in blocked,
            "New synthetic source overlaps allocated data",
        )
        if source["event_id_cnty"] not in by_id:
            additional_sources.append(source)
        for country in prep.DEFAULT_CONFIG["country_regions"]:
            require(
                not re.search(
                    r"\b" + re.escape(country) + r"\b", recipe["template"], re.I
                ),
                f"Synthetic base retains a country clue: {country}",
            )
        require(
            bool(recipe["derivation_note"]), "Missing synthetic construction disclosure"
        )
        constraints = [
            "Hypothetical source-inspired account, not a historical claim; preserve class criteria and all non-designated facts."
        ]
        for family, condition in [
            ("original", {"rationale": recipe["derivation_note"]}),
            *recipe["conditions"].items(),
        ]:
            cases.append(
                render_case(
                    "synthetic",
                    recipe["base_id"],
                    source,
                    recipe["label"],
                    recipe["template"],
                    family,
                    condition,
                    countries,
                    constraints,
                    recipe["derivation_note"],
                )
            )
        for family, reason in recipe["omissions"].items():
            omissions.append(
                {
                    "cohort": "synthetic",
                    "base_id": recipe["base_id"],
                    "source_event_id": source["event_id_cnty"],
                    "condition": family,
                    "country_versions": len(countries),
                    "reason": reason,
                }
            )
    require(len({r["case_id"] for r in cases}) == len(cases), "Duplicate review case")
    allocated_by_id = {r["event_id_cnty"]: r for r in allocated + pilot}
    compared, conflicts = similarity_hits(
        additional_sources, list(allocated_by_id.values())
    )
    require(
        not conflicts,
        f"Additional synthetic sources overlap allocated text: {conflicts}",
    )
    audit = {
        "additional_sources": len(additional_sources),
        "pairs_checked": compared,
        "threshold": 0.90,
        "direction": "both",
        "autojunk": False,
        "scope": "all dates and countries; additional synthetic source notes vs allocated inputs and each other",
        "conflicts": conflicts,
    }
    return baselines, cases, omissions, audit


def adjudicated_cases(recipes, cases):
    expected = {case["case_id"]: case for case in cases}
    reviews, decisions = {}, {}
    for review in recipes["reviews"]:
        case_id = review["case_id"]
        require(case_id in expected, f"Unknown review case: {case_id}")
        require(
            review["case_sha256"] == expected[case_id]["case_sha256"],
            f"Stale pair review: {case_id}",
        )
        require(
            review["decision"] in {"accept", "reject"} and review["rationale"].strip(),
            "Invalid pair judgment",
        )
        evidence = review["evidence_span"]
        require(
            evidence
            and any(
                evidence in text
                for r in expected[case_id]["renderings"]
                for text in (r["original"], r["edited"])
            ),
            f"Nonliteral pair-review evidence: {case_id}",
        )
        key = (case_id, review["reviewer_id"])
        require(
            key not in reviews and bool(review["reviewer_id"]),
            "Duplicate/missing reviewer identity",
        )
        context = recipes["reviewer_contexts"][review["reviewer_id"]]
        require(
            context["kind"] == "AI" and bool(context["context_id"]),
            "Incorrect reviewer provenance",
        )
        require(
            context["context_id"]
            != recipes["design_contexts"][expected[case_id]["cohort"]],
            "A recipe designer cannot serve as its independent pair reviewer",
        )
        reviews[key] = review
    for decision in recipes["adjudications"]:
        case_id = decision["case_id"]
        require(
            case_id in expected and case_id not in decisions,
            "Unknown/duplicate adjudication",
        )
        require(
            decision["case_sha256"] == expected[case_id]["case_sha256"],
            f"Stale adjudication: {case_id}",
        )
        votes = [r for (key, _), r in reviews.items() if key == case_id]
        contexts = {
            recipes["reviewer_contexts"][r["reviewer_id"]]["context_id"] for r in votes
        }
        require(len(contexts) >= 2, f"Two separate review contexts required: {case_id}")
        require(
            decision["decision"] in {"accept", "reject"} and decision["reason"].strip(),
            "Invalid adjudication",
        )
        if decision["decision"] == "accept":
            require(
                all(r["decision"] == "accept" for r in votes),
                f"Unresolved pair objection: {case_id}",
            )
        decisions[case_id] = decision
    require(
        set(decisions) == set(expected),
        "Every proposed pair/base requires a terminal AI adjudication",
    )
    for case in cases:
        if case["cohort"] == "synthetic" and case["condition"] == "original":
            require(
                decisions[case["case_id"]]["decision"] == "accept",
                "All 12 synthetic bases must pass review",
            )
    return decisions


def export_cases(cases, status):
    stimuli, changes = [], []
    for case in cases:
        for rendering in case["renderings"]:
            suffix = ":" + rendering["country"] if case["cohort"] == "synthetic" else ""
            stimulus_id = case["case_id"] + suffix
            original_id = f"{case['cohort']}:{case['base_id']}:original" + suffix
            stimuli.append(
                {
                    **{
                        k: case[k]
                        for k in (
                            "cohort",
                            "base_id",
                            "source_event_id",
                            "source_family_id",
                            "source_file",
                            "source_record",
                            "label",
                            "case_id",
                            "case_sha256",
                        )
                    },
                    "stimulus_id": stimulus_id,
                    "country": rendering["country"],
                    "condition": case["condition"],
                    "original_id": original_id,
                    "text": rendering["edited"],
                    "text_sha256": prep.digest(rendering["edited"]),
                    "review_status": status,
                }
            )
            for change in rendering["changes"]:
                changes.append(
                    {"stimulus_id": stimulus_id, "original_id": original_id, **change}
                )
    return stimuli, changes


def build(
    output_dir, recipes_path=RECIPES, input_dir=ROOT / "data/accepted", draft=False
):
    require(not output_dir.exists(), f"Output directory already exists: {output_dir}")
    recipes = json.loads(recipes_path.read_text(encoding="utf-8"))
    baselines, cases, omissions, source_audit = draft_cases(recipes, input_dir)
    if draft:
        accepted = cases
    else:
        decisions = adjudicated_cases(recipes, cases)
        accepted = [
            case for case in cases if decisions[case["case_id"]]["decision"] == "accept"
        ]
        for case in cases:
            if decisions[case["case_id"]]["decision"] == "reject":
                omissions.append(
                    {
                        "cohort": case["cohort"],
                        "base_id": case["base_id"],
                        "source_event_id": case["source_event_id"],
                        "condition": case["condition"],
                        "country_versions": len(case["renderings"]),
                        "reason": decisions[case["case_id"]]["reason"],
                    }
                )
    stimuli, changes = export_cases(
        baselines + accepted, "candidate" if draft else "ai_reviewed"
    )
    ids = {r["stimulus_id"] for r in stimuli}
    require(
        len(ids) == len(stimuli) and all(r["original_id"] in ids for r in stimuli),
        "Missing/duplicate baseline link",
    )
    for row in stimuli:
        if row["condition"] != "original":
            require(
                row["text"]
                != next(
                    r["text"] for r in stimuli if r["stimulus_id"] == row["original_id"]
                ),
                "No-op condition",
            )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".variants-", dir=output_dir.parent
    ) as temporary:
        stage = Path(temporary) / "export"
        stage.mkdir()
        prep.write_csv(stage / "stimuli.csv", STIMULUS_FIELDS, stimuli)
        prep.write_csv(stage / "changes.csv", CHANGE_FIELDS, changes)
        prep.write_csv(stage / "omissions.csv", OMISSION_FIELDS, omissions)
        if draft:
            (stage / "review_cases.jsonl").write_text(
                "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n",
                encoding="utf-8",
            )
        counts = Counter(r["cohort"] for r in stimuli)
        lines = [
            "# Pilot Stimuli",
            "",
            f"Status: {'design candidates; not released for inference' if draft else 'AI-reviewed experiment inputs'}.",
            "",
            f"- Real-event rows: {counts['real']} from 28 unchanged source baselines.",
            f"- Synthetic rows: {counts['synthetic']} from 12 bases, two per class, rendered under four countries.",
            f"- Review units: {len(cases)}; accepted: {len(accepted) if not draft else 'not adjudicated'}.",
            "",
            "`stimuli.csv` contains the exact model-input text and baseline links. `changes.csv` records character offsets and literal replacements; `omissions.csv` defines unavailable conditions for denominators. Send only the text to models; labels, conditions, source IDs, and review metadata are analysis fields.",
            "",
            "Synthetic descriptions, actor identities, appraisals, and attribution interventions are deliberate counterfactuals. Construction disclosures and exact-pair judgments are retained in the recipe input. AI review is not human verification; separate contexts can share systematic errors. No model inference has been run.",
            "",
        ]
        (stage / "report.md").write_text("\n".join(lines), encoding="utf-8")
        result = {
            "format_version": 1,
            "recipe_version": recipes["recipe_version"],
            "review_date": recipes["review_date"],
            "status": "draft" if draft else "ai_reviewed",
            "human_review_performed": False,
            "human_review_required": False,
            "inference_performed": False,
            "countries": recipes["countries"],
            "stimulus_counts": dict(counts),
            "review_units": len(cases),
            "accepted_review_units": len(accepted) if not draft else 0,
            "synthetic_source_audit": source_audit,
            "inputs": {
                "recipes_sha256": prep.file_digest(recipes_path),
                "accepted_manifest_sha256": prep.file_digest(
                    input_dir / "manifest.json"
                ),
                "input_review_sha256": prep.file_digest(
                    ROOT / "data/reviews/input_review.json"
                ),
                "processed_manifest_sha256": prep.file_digest(
                    ROOT / "data/processed/manifest.json"
                ),
                "builder_sha256": prep.file_digest(Path(__file__)),
            },
            "outputs": {
                p.name: {"sha256": prep.file_digest(p)} for p in sorted(stage.iterdir())
            },
        }
        prep.write_json(stage / "manifest.json", result)
        stage.rename(output_dir)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipes", type=Path, default=RECIPES)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/accepted")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/variants/pilot")
    parser.add_argument(
        "--draft",
        action="store_true",
        help="Export candidates and review packets outside inference release",
    )
    args = parser.parse_args()
    try:
        result = build(
            args.output_dir.resolve(), args.recipes, args.input_dir, args.draft
        )
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"Variant build failed: {error}\n")
    print(
        f"Built {sum(result['stimulus_counts'].values())} {result['status']} stimuli: {args.output_dir}"
    )
