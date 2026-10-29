#!/usr/bin/env python3
"""Build the main stimulus release from explicit recipes and independent AI reviews."""

import argparse
import json
import random
import re
from datetime import date
import tempfile
from collections import Counter
from pathlib import Path

from . import prepare_data as prep
from . import model_inputs
from .finalize_inputs import FAMILIES, similarity_hits
from .utils import (
    digest,
    file_digest,
    read_csv,
    require,
    verified_snapshot,
    write_csv,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / "data/reviews/main_variants.json"
EXTRA_FIELDS = [
    "label_support",
    "notes_sufficiency",
    "review_flags",
    "case_constraints",
]
# Four source families supplemented the accepted 28-event pilot; preserve their
# exclusion after retiring pilot-only derived artifacts (history: 1af36b5).
RETIRED_PILOT_EXTRA_IDS = {"IND171401", "LBY11833", "LBY12000", "PAK148400"}


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
    return digest(json.dumps(case, sort_keys=True, ensure_ascii=False))


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
                "All synthetic bases must pass review",
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
                    "text_sha256": digest(rendering["edited"]),
                    "review_status": status,
                }
            )
            for change in rendering["changes"]:
                changes.append(
                    {"stimulus_id": stimulus_id, "original_id": original_id, **change}
                )
    return stimuli, changes


def load_sources(recipes, input_dir):
    verified_snapshot(input_dir)
    main = read_csv(input_dir / "main.csv")
    require(len(main) == 590, "The accepted main cohort must retain all 590 events")
    by_id = {row["event_id_cnty"]: row for row in main}
    require(len(by_id) == 590, "Duplicate main event")
    ledger = {
        row["event_id"]: row
        for row in read_csv(ROOT / "data/reviews/sample_review.csv")
    }
    for row in main:
        review = ledger[row["event_id_cnty"]]
        require(review["notes_sha256"] == digest(row["notes"]), "Stale input review")
        require(
            review["decision"] in {"accept", "accept_with_constraints"},
            "Excluded main input",
        )
    processed = verified_snapshot(ROOT / "data/processed")
    family = {}
    for pair in read_csv(ROOT / "data/processed/related_event_pairs.csv"):
        for key in ("event_id_1", "event_id_2"):
            family[pair[key]] = pair["event_group_id"]
    needed = {
        (row["source_file"], str(row["source_record"])) for row in recipes["synthetic"]
    }
    sources = dict(by_id)
    for item in processed["inputs"]:
        path = ROOT / "conflict_data" / item["file"]
        require(file_digest(path) == item["sha256"], "Raw data changed")
        for ordinal, row in enumerate(read_csv(path), 1):
            if (path.name, str(ordinal)) in needed:
                event_id = row["event_id_cnty"]
                raw = {
                    **row,
                    "source_file": path.name,
                    "source_record": str(ordinal),
                    "event_group_id": family.get(
                        event_id, "group_" + digest(event_id)[:16]
                    ),
                }
                if event_id in sources:
                    require(
                        raw["notes"] == sources[event_id]["notes"]
                        and raw["event_group_id"]
                        == sources[event_id]["event_group_id"],
                        "Main/raw source mismatch",
                    )
                else:
                    sources[event_id] = raw
    return main, ledger, sources


def validate_pilot_exclusions(rows, accepted_pilot, family_map):
    expected = {
        row["event_id_cnty"] for row in accepted_pilot
    } | RETIRED_PILOT_EXTRA_IDS
    require(len(rows) == len(expected) == 32, "Retired pilot exclusion count changed")
    require(
        {row["source_event_id"] for row in rows} == expected,
        "Retired pilot source coverage changed",
    )
    require(
        len({row["source_family_id"] for row in rows}) == 32,
        "Duplicate retired pilot source family",
    )
    require(
        len({(row["source_file"], row["source_record"]) for row in rows}) == 32,
        "Duplicate retired pilot source ordinal",
    )
    for row in rows:
        event_id = row["source_event_id"]
        require(
            row["source_family_id"]
            == family_map.get(event_id, "group_" + digest(event_id)[:16]),
            f"Retired pilot family changed: {event_id}",
        )


def draft_cases(recipes, input_dir=ROOT / "data/accepted"):
    require(
        recipes["format_version"] == 1 and recipes["study"] == "main",
        "Wrong recipe format",
    )
    main, ledger, sources = load_sources(recipes, input_dir)
    main_by_id = {row["event_id_cnty"]: row for row in main}
    require(
        {row["event_id"] for row in recipes["real"]} == set(main_by_id)
        and len(recipes["real"]) == 590,
        "Recipes must cover every main baseline once",
    )
    countries = recipes["countries"]
    require(
        len(countries) == 10
        and set(countries) == set(prep.DEFAULT_CONFIG["country_regions"]),
        "Main country coverage changed",
    )
    policy = json.loads((ROOT / "data/reviews/input_review.json").read_text())[
        "family_policy"
    ]
    baselines, cases, omissions = [], [], []
    for recipe in recipes["real"]:
        validate_conditions(recipe)
        source = main_by_id[recipe["event_id"]]
        review = ledger[recipe["event_id"]]
        constraints = [
            review["edit_constraints"],
            "Keep source label qualifications: "
            + review["label_support"]
            + "; note sufficiency: "
            + review["notes_sufficiency"]
            + "; flags: "
            + review["flags"],
        ]
        base = render_case(
            "real",
            recipe["event_id"],
            source,
            source["label"],
            source["notes"],
            "original",
            {"rationale": "Unmodified accepted main source note."},
            [source["country"]],
            constraints,
            "",
        )
        baselines.append(base)
        for family, condition in recipe["conditions"].items():
            cases.append(
                render_case(
                    "real",
                    recipe["event_id"],
                    source,
                    source["label"],
                    source["notes"],
                    family,
                    condition,
                    [source["country"]],
                    [policy[family], *constraints],
                    "",
                )
            )
        for family, reason in recipe["omissions"].items():
            omissions.append(
                {
                    "cohort": "real",
                    "base_id": recipe["event_id"],
                    "source_event_id": recipe["event_id"],
                    "condition": family,
                    "country_versions": 1,
                    "reason": reason,
                }
            )
    synthetic = recipes["synthetic"]
    require(
        len(synthetic) == len({r["base_id"] for r in synthetic}) == 120,
        "Need 120 distinct synthetic bases",
    )
    require(
        Counter(r["label"] for r in synthetic)
        == Counter({label: 20 for label in model_inputs.CLASSES}),
        "Need 20 synthetic bases per class",
    )
    allocated = [
        r
        for name in ("candidates.csv", "pilot.csv", "reserves.csv")
        for r in read_csv(ROOT / "data/processed" / name)
    ]
    pilot_rows = read_csv(input_dir / "pilot.csv")
    pilot_exclusions = read_csv(ROOT / "data/reviews/pilot_exclusions.csv")
    family_map = {}
    for pair in read_csv(ROOT / "data/processed/related_event_pairs.csv"):
        for key in ("event_id_1", "event_id_2"):
            family_map[pair[key]] = pair["event_group_id"]
    validate_pilot_exclusions(pilot_exclusions, pilot_rows, family_map)
    blocked = {r["event_group_id"] for r in allocated + main + pilot_rows}
    pilot_groups = {r["source_family_id"] for r in pilot_exclusions} | {
        r["event_group_id"] for r in pilot_rows
    }
    allocated_by_id = {r["event_id_cnty"]: r for r in allocated + main + pilot_rows}
    # Include the pilot's additional E/S source notes in the separation screen.
    old_needed = {
        (r["source_file"], str(r["source_record"])): r for r in pilot_exclusions
    }
    found_old = set()
    for path in sorted((ROOT / "conflict_data").glob("*.csv")):
        for ordinal, raw in enumerate(read_csv(path), 1):
            if (path.name, str(ordinal)) in old_needed:
                exclusion = old_needed[path.name, str(ordinal)]
                found_old.add(exclusion["source_event_id"])
                require(
                    raw["event_id_cnty"] == exclusion["source_event_id"]
                    and digest(raw["notes"]) == exclusion["notes_sha256"],
                    "Retired pilot exclusion changed",
                )
                if raw["event_id_cnty"] not in allocated_by_id:
                    allocated_by_id[raw["event_id_cnty"]] = raw
    require(
        found_old == {r["source_event_id"] for r in pilot_exclusions},
        "Retired pilot raw source lookup incomplete",
    )
    extras, used, shared = [], set(), []
    for recipe in synthetic:
        validate_conditions(recipe)
        source = sources[recipe["source_event_id"]]
        require(
            source["source_file"] == recipe["source_file"]
            and source["source_record"] == str(recipe["source_record"]),
            "Synthetic source identity changed",
        )
        require(
            prep.LABELS[source["event_type"]] == recipe["label"],
            "Synthetic source label mismatch",
        )
        require(
            source["country"] in countries, "Synthetic source outside study countries"
        )
        group = source["event_group_id"]
        require(
            group not in used and group not in pilot_groups,
            f"Synthetic source reuses a family or overlaps pilot: {source['event_id_cnty']}",
        )
        used.add(group)
        if source["event_id_cnty"] in main_by_id:
            shared.append(source["event_id_cnty"])
        else:
            require(
                group not in blocked,
                "Additional synthetic source overlaps allocated family",
            )
            extras.append(source)
        require(
            recipe["template"].count("{country}") == 1
            and recipe["derivation_note"].strip(),
            "Missing placeholder/disclosure",
        )
        for country in countries:
            require(
                country.lower() not in recipe["template"].lower(),
                "Synthetic template retains a study country",
            )
        constraints = [
            "Hypothetical source-inspired account, not a historical claim; preserve mechanics, roles, numbers, uncertainty and event boundaries. All country versions differ only in the country placeholder."
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
                    (
                        [policy[family]] + constraints
                        if family != "original"
                        else constraints
                    ),
                    recipe["derivation_note"],
                )
            )
        for family, reason in recipe["omissions"].items():
            omissions.append(
                {
                    "cohort": "synthetic",
                    "base_id": recipe["base_id"],
                    "source_event_id": recipe["source_event_id"],
                    "condition": family,
                    "country_versions": 10,
                    "reason": reason,
                }
            )
    comparisons, conflicts = similarity_hits(extras, list(allocated_by_id.values()))
    require(
        not conflicts,
        f"Additional synthetic sources overlap allocated text: {conflicts}",
    )
    template_pairs, template_conflicts = similarity_hits(
        [
            {"event_id_cnty": row["base_id"], "notes": row["template"]}
            for row in synthetic
        ],
        [],
    )
    require(
        not template_conflicts,
        f"Constructed synthetic originals overlap: {template_conflicts}",
    )
    require(
        len({case["case_id"] for case in cases}) == len(cases), "Duplicate review unit"
    )
    source_audit = {
        "additional_source_count": len(extras),
        "shared_main_real_sources": shared,
        "distinct_synthetic_source_families": len(used),
        "pairs_checked": comparisons,
        "synthetic_template_pairs_checked": template_pairs,
        "synthetic_template_conflicts": template_conflicts,
        "threshold": 0.90,
        "direction": "both",
        "conflicts": conflicts,
        "scope": "Additional synthetic source notes vs all initially allocated/reviewed main, pilot and reserve inputs, pilot synthetic sources, and each other; all dates/countries.",
    }
    return baselines, cases, omissions, source_audit, ledger


def build(
    output_dir,
    recipes_path=RECIPES,
    input_dir=ROOT / "data/accepted",
    draft=False,
    token_counter=None,
):
    require(not output_dir.exists(), f"Output directory already exists: {output_dir}")
    recipes = json.loads(recipes_path.read_text())
    baselines, cases, omissions, source_audit, ledger = draft_cases(recipes, input_dir)
    decisions = {} if draft else adjudicated_cases(recipes, cases)
    accepted = (
        cases
        if draft
        else [
            case for case in cases if decisions[case["case_id"]]["decision"] == "accept"
        ]
    )
    for case in cases:
        if not draft and decisions[case["case_id"]]["decision"] == "reject":
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
    case_by_id = {c["case_id"]: c for c in baselines + cases}
    for row in stimuli:
        case = case_by_id[row["case_id"]]
        source_review = (
            ledger.get(row["source_event_id"], {}) if row["cohort"] == "real" else {}
        )
        row.update(
            label_support=source_review.get(
                "label_support",
                "constructed_ai_reviewed" if not draft else "constructed_candidate",
            ),
            notes_sufficiency=source_review.get("notes_sufficiency", "constructed"),
            review_flags=source_review.get("flags", ""),
            case_constraints=json.dumps(case["constraints"], ensure_ascii=False),
        )
    by_id = {row["stimulus_id"]: row for row in stimuli}
    require(
        len(by_id) == len(stimuli)
        and all(row["original_id"] in by_id for row in stimuli),
        "Invalid stimulus/baseline identity",
    )
    require(
        sum(r["cohort"] == "real" and r["condition"] == "original" for r in stimuli)
        == 590,
        "Main baseline omitted",
    )
    for row in stimuli:
        original = by_id[row["original_id"]]
        if row["condition"] != "original":
            require(row["text"] != original["text"], "No-op main edit")
    count, tokenizer = (
        model_inputs.load_token_counter() if token_counter is None else token_counter
    )
    records = [
        {"dataset": "main_stimuli", "input_id": r["stimulus_id"], **r} for r in stimuli
    ]
    audit = model_inputs.audit_lengths(records, count)
    prototype_tokens = {
        label: count(text) for label, text in model_inputs.CLASSES.items()
    }
    require(
        all(0 < n <= model_inputs.ENCODER_LIMIT for n in prototype_tokens.values()),
        "Prototype too long",
    )
    protocol = {
        "encoder_model": model_inputs.ENCODER,
        "decoder_model": model_inputs.DECODER,
        "class_definitions": model_inputs.CLASSES,
        "class_order": list(model_inputs.CLASSES),
        "encoder_instruction": model_inputs.ENCODER_INSTRUCTION,
        "encoder_input_template": "Instruct: {instruction}\nQuery: {text}",
        "encoder_classification": "Normalized cosine similarity against the six unprefixed class descriptions; reject nonfinite/zero vectors; ties use B,P,R,E,V,S order. Similarities are not probabilities.",
        "decoder_system_prompt": model_inputs.DECODER_PROMPT,
        "decoder_user_message": "Exact stimulus text only.",
        "decoder_parameters": {
            "temperature": 0,
            "seed": 42,
            "max_tokens": model_inputs.MAX_OUTPUT_TOKENS,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "event_class",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "label": {
                                "type": "string",
                                "enum": list(model_inputs.CLASSES),
                            }
                        },
                        "required": ["label"],
                        "additionalProperties": False,
                    },
                },
            },
        },
        "encoder_parameters": {"encoding_format": "float"},
        "response_caching": False,
        "request_order_seed": 42,
        "repeats_per_model": 3,
        "repeat_rule": "Shortest, median and longest original texts by E5 token count with stimulus-ID ties; one fresh additional request for each, same texts for both models.",
        "prototype_tokens": prototype_tokens,
        "tokenizer": tokenizer,
        "provider_compatibility_tested": False,
        "note": "Input preparation only. No classification or embedding requests. Served weights, provider preprocessing and decoder structured-output routing require compatibility checks before inference.",
    }
    counts = Counter(r["cohort"] for r in stimuli)
    lengths = {r["input_id"]: r["encoder_tokens"] for r in audit}
    originals = sorted(
        (r for r in stimuli if r["condition"] == "original"),
        key=lambda r: (lengths[r["stimulus_id"]], r["stimulus_id"]),
    )
    repeat_ids = [
        originals[index]["stimulus_id"]
        for index in (0, len(originals) // 2, len(originals) - 1)
    ]
    schedule = []
    for model_id in (model_inputs.ENCODER, model_inputs.DECODER):
        schedule.extend(
            {
                "model": model_id,
                "stimulus_id": r["stimulus_id"],
                "replicate": 0,
                "request_id": f"{model_id}:{r['stimulus_id']}:replicate=0",
            }
            for r in stimuli
        )
        schedule.extend(
            {
                "model": model_id,
                "stimulus_id": ident,
                "replicate": 1,
                "request_id": f"{model_id}:{ident}:replicate=1",
            }
            for ident in repeat_ids
        )
    random.Random(42).shuffle(schedule)
    for index, row in enumerate(schedule, 1):
        row["sequence"] = index
    protocol["repeat_stimulus_ids"] = repeat_ids
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".main-variants-", dir=output_dir.parent
    ) as temporary:
        stage = Path(temporary) / "export"
        stage.mkdir()
        write_csv(stage / "stimuli.csv", STIMULUS_FIELDS + EXTRA_FIELDS, stimuli)
        write_csv(stage / "changes.csv", CHANGE_FIELDS, changes)
        write_csv(stage / "omissions.csv", OMISSION_FIELDS, omissions)
        write_csv(stage / "length_audit.csv", model_inputs.AUDIT_FIELDS, audit)
        write_json(stage / "protocol.json", protocol)
        write_csv(
            stage / "request_order.csv",
            ["sequence", "model", "stimulus_id", "replicate", "request_id"],
            schedule,
        )
        if draft:
            (stage / "review_cases.jsonl").write_text(
                "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n"
            )
        balance = []
        for key, value in sorted(
            Counter(
                (r["cohort"], r["country"], r["label"], r["condition"]) for r in stimuli
            ).items()
        ):
            balance.append(
                dict(zip(["cohort", "country", "label", "condition"], key), count=value)
            )
        write_csv(
            stage / "coverage.csv",
            ["cohort", "country", "label", "condition", "count"],
            balance,
        )
        report = [
            "# Main Stimulus Preparation",
            "",
            f"Status: {'draft; not released' if draft else 'AI-reviewed data release; inference not performed'}.",
            "",
            f"- Real: {counts['real']} rows, including all 590 unchanged accepted baselines.",
            f"- Synthetic: {counts['synthetic']} rows from 120 source-inspired bases (20/class), each under ten countries.",
            f"- Review units: {len(cases)}; accepted: {len(accepted)}. Two independent AI contexts review every proposed edit/base; original real inputs retain their existing input review.",
            f"- Maximum E5 input: {max(r['encoder_tokens'] for r in audit)}/512 tokens, including instruction and special tokens. No truncation.",
            "",
            "`stimuli.csv` contains exact texts, baseline/family links and case restrictions. `changes.csv` records literal spans; `omissions.csv` and `coverage.csv` define eligible denominators. `protocol.json` freezes the shared class definitions and model-visible formatting. Source qualifications remain attached to real inputs; do not filter on later model correctness.",
            "`request_order.csv` freezes the randomized two-model schedule and three fresh repeat texts per model. Expected labels, conditions, identifiers and review metadata must not be sent to models. Reviewers saw intended labels and construction rationales; these judgments are not independent model-accuracy measurements.",
            "",
            "Synthetic notes and identity/framing/intensity/attribution edits are counterfactuals, not historical findings. Synthetic source families shared with Experiment A are listed in the manifest; keep these links when resampling. Pilot/reserve families are excluded. AI review is not human verification and contexts may share systematic errors. Provider compatibility and classification behavior have not been tested for the selected models; no main inference is authorized by this data release.",
            "",
        ]
        (stage / "report.md").write_text("\n".join(report))
        result = {
            "format_version": 1,
            "study": "main",
            "status": "draft" if draft else "ai_reviewed",
            "data_preparation_complete": not draft,
            "main_inference_ready": False,
            "inference_performed": False,
            "provider_compatibility_tested": False,
            "human_review_performed": False,
            "human_review_required": False,
            "countries": recipes["countries"],
            "stimulus_counts": dict(counts),
            "real_baselines": 590,
            "synthetic_bases": 120,
            "review_units": len(cases),
            "accepted_review_units": len(accepted),
            "synthetic_source_audit": source_audit,
            "maximum_encoder_tokens": max(r["encoder_tokens"] for r in audit),
            "core_classifications_two_models": len(stimuli) * 2,
            "planned_classifications_two_models": len(schedule),
            "repeat_requests": 6,
            "inputs": {
                "recipes_sha256": file_digest(recipes_path),
                "accepted_manifest_sha256": file_digest(input_dir / "manifest.json"),
                "sample_review_sha256": file_digest(
                    ROOT / "data/reviews/sample_review.csv"
                ),
                "input_review_policy_sha256": file_digest(
                    ROOT / "data/reviews/input_review.json"
                ),
                "processed_manifest_sha256": file_digest(
                    ROOT / "data/processed/manifest.json"
                ),
                "pilot_exclusions_sha256": file_digest(
                    ROOT / "data/reviews/pilot_exclusions.csv"
                ),
                "builder_sha256": file_digest(Path(__file__)),
                "utilities_sha256": file_digest(Path(__file__).with_name("utils.py")),
                "preprocessing_code_sha256": file_digest(Path(prep.__file__)),
                "input_review_code_sha256": file_digest(
                    Path(__file__).with_name("finalize_inputs.py")
                ),
                "model_input_code_sha256": file_digest(Path(model_inputs.__file__)),
            },
            "outputs": {p.name: {"sha256": file_digest(p)} for p in stage.iterdir()},
        }
        write_json(stage / "manifest.json", result)
        stage.rename(output_dir)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipes", type=Path, default=RECIPES)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/accepted")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/variants/main")
    parser.add_argument("--draft", action="store_true")
    args = parser.parse_args()
    try:
        result = build(args.output_dir, args.recipes, args.input_dir, args.draft)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"Main preparation failed: {error}\n")
    print(
        f"Prepared {sum(result['stimulus_counts'].values())} {result['status']} texts; no inference performed."
    )


if __name__ == "__main__":
    main()
