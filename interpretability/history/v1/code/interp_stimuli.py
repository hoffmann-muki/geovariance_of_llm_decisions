"""Build ambiguity-ladder stimuli, system prompts and the prompt schedule.

Reads the reviewed ladders, bases and prompt parts in `interpretability/data/`,
runs the construction checks in interpretability/PLAN.md, and writes an
immutable snapshot to an unused output directory. It loads no model.
"""

import argparse
import itertools
import json
import re
import string
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .utils import (
    digest,
    file_digest,
    read_csv,
    read_json,
    require,
    write_csv,
    write_json,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "interpretability" / "data"
PRODUCER = Path(__file__).resolve()

BOUNDARIES = ("PR", "BV")
LEVELS = (1, 2, 3, 4, 5)
LEVEL_ROLES = {1: "anchor_lower", 5: "anchor_higher"}
WORDINGS = 2
NO_COUNTRY = "none"
STAGE_CALIBRATION = "A"
STAGE_COUNTRIES = "B"
SLOT_FIELDS = {"country", "evidence"}

STIMULUS_FIELDS = [
    "stimulus_id",
    "stage",
    "boundary",
    "base_id",
    "base_factors",
    "level",
    "level_role",
    "wording",
    "country",
    "evidence",
    "text",
    "text_sha256",
]
SYSTEM_PROMPT_FIELDS = [
    "task_id",
    "boundary",
    "task_version",
    "order",
    "first_class",
    "second_class",
    "lower_class",
    "higher_class",
    "system_prompt",
    "system_prompt_sha256",
]
SCHEDULE_FIELDS = ["prompt_id", "stage", "boundary", "stimulus_id", "task_id"]


# ---------------------------------------------------------------- inputs


def load_inputs(data_dir):
    ladders = read_json(data_dir / "ladders.json")
    prompts = read_json(data_dir / "prompts.json")
    require(
        ladders.get("status") == "approved_for_calibration",
        "Ladders are not approved for calibration",
    )
    require(set(ladders["boundaries"]) == set(BOUNDARIES), "Unexpected boundaries")
    require(set(prompts["boundaries"]) == set(BOUNDARIES), "Prompt boundaries differ")
    bases = {
        boundary: read_csv(data_dir / f"bases_{boundary.lower()}.csv")
        for boundary in BOUNDARIES
    }
    return ladders, prompts, bases


def frame_fields(frame):
    return {name for _, name, _, _ in string.Formatter().parse(frame) if name}


def validate_ladder(boundary, spec):
    levels = [level["level"] for level in spec["levels"]]
    require(tuple(levels) == LEVELS, f"{boundary}: levels must be 1-5 in order")
    phrases = []
    for level in spec["levels"]:
        role = LEVEL_ROLES.get(level["level"], "ambiguous")
        require(
            level["role"] == role, f"{boundary} L{level['level']}: role must be {role}"
        )
        wordings = level["wordings"]
        require(
            len(wordings) == WORDINGS, f"{boundary} L{level['level']}: two wordings"
        )
        require(
            len(set(wordings)) == WORDINGS, f"{boundary} L{level['level']}: duplicate"
        )
        for wording in wordings:
            require(
                wording.strip() == wording and wording, f"{boundary}: blank wording"
            )
            require("{" not in wording, f"{boundary}: placeholder in evidence")
        phrases += wordings
    require(
        len(set(phrases)) == len(phrases), f"{boundary}: wordings repeat across levels"
    )
    for phrase in phrases:
        require(
            phrase not in spec["frame"], f"{boundary}: evidence appears in the frame"
        )
    require(
        SLOT_FIELDS <= frame_fields(spec["frame"]),
        f"{boundary}: frame needs country and evidence slots",
    )


def validate_bases(boundary, spec, bases):
    factors = sorted(frame_fields(spec["frame"]) - SLOT_FIELDS)
    require(bases, f"{boundary}: no bases")
    for row in bases:
        require(
            set(row) == {"base_id", *factors},
            f"{boundary}: base columns must be base_id and {factors}",
        )
        require(all(row[name].strip() for name in factors), f"{boundary}: blank factor")
    ids = [row["base_id"] for row in bases]
    require(len(set(ids)) == len(ids), f"{boundary}: duplicate base IDs")
    tuples = [tuple(row[name] for name in factors) for row in bases]
    require(len(set(tuples)) == len(tuples), f"{boundary}: duplicate bases")

    crossed = spec["crossed_factors"]
    levels = [sorted({row[name] for row in bases}) for name in crossed]
    found = Counter(tuple(row[name] for name in crossed) for row in bases)
    expected = set(itertools.product(*levels))
    require(
        set(found) == expected and set(found.values()) == {1},
        f"{boundary}: {crossed} must be fully crossed once",
    )
    for name in spec["balanced_factors"]:
        counts = Counter(row[name] for row in bases)
        require(len(set(counts.values())) == 1, f"{boundary}: {name} is unbalanced")
        for other in crossed:
            per_value = defaultdict(set)
            for row in bases:
                per_value[row[other]].add(row[name])
            require(
                len({len(values) for values in per_value.values()}) == 1,
                f"{boundary}: {name} is unevenly spread across {other}",
            )
    return factors


def validate_prompt_parts(boundary, ladder, parts):
    classes = parts["order_a"]
    require(len(classes) == 2 and len(set(classes)) == 2, f"{boundary}: two classes")
    require(
        set(classes) == {ladder["lower_class"], ladder["higher_class"]},
        f"{boundary}: prompt classes differ from ladder classes",
    )
    for key in ("phrases", "definitions"):
        require(
            set(parts[key]) == set(classes), f"{boundary}: {key} must cover classes"
        )
    for name, definition in parts["definitions"].items():
        require(definition.startswith(f"{name}: "), f"{boundary}: definition label")
    first, second = (parts["definitions"][name] for name in classes)
    require(
        not (first.startswith(second) or second.startswith(first)),
        f"{boundary}: definitions must be distinguishable",
    )


# ---------------------------------------------------------------- render


def country_code(country):
    return country.lower().replace(" ", "_")


def country_conditions(ladders):
    return [(NO_COUNTRY, "")] + [
        (name, ladders["country_slot"].format(country=display))
        for name, display in ladders["countries"].items()
    ]


def render_stimuli(ladders, bases):
    stimuli = []
    for boundary in BOUNDARIES:
        spec = ladders["boundaries"][boundary]
        validate_ladder(boundary, spec)
        factors = validate_bases(boundary, spec, bases[boundary])
        for base in bases[boundary]:
            values = {name: base[name] for name in factors}
            for level in spec["levels"]:
                for wording, evidence in enumerate(level["wordings"], start=1):
                    for country, slot in country_conditions(ladders):
                        text = spec["frame"].format(
                            **values, country=slot, evidence=evidence
                        )
                        stimuli.append(
                            {
                                "stimulus_id": ":".join(
                                    [
                                        boundary,
                                        base["base_id"],
                                        f"L{level['level']}",
                                        f"W{wording}",
                                        country_code(country),
                                    ]
                                ),
                                "stage": (
                                    STAGE_CALIBRATION
                                    if country == NO_COUNTRY
                                    else STAGE_COUNTRIES
                                ),
                                "boundary": boundary,
                                "base_id": base["base_id"],
                                "base_factors": json.dumps(values, sort_keys=True),
                                "level": level["level"],
                                "level_role": level["role"],
                                "wording": wording,
                                "country": country,
                                "evidence": evidence,
                                "text": text,
                                "text_sha256": digest(text),
                            }
                        )
    return stimuli


def render_system_prompts(prompts, ladders):
    rows = []
    for boundary in BOUNDARIES:
        parts = prompts["boundaries"][boundary]
        ladder = ladders["boundaries"][boundary]
        validate_prompt_parts(boundary, ladder, parts)
        for version in prompts["task_versions"]:
            require(
                version in ("defined", "undefined"), f"Unknown task version: {version}"
            )
            for order in prompts["orders"]:
                require(order in ("A", "B"), f"Unknown prompt order: {order}")
                first, second = parts["order_a"][:: 1 if order == "A" else -1]
                paragraphs = [
                    prompts["intro_template"].format(
                        domain=parts["domain"],
                        first_phrase=parts["phrases"][first],
                        second_phrase=parts["phrases"][second],
                    )
                ]
                if version == "defined":
                    paragraphs += [
                        parts["definitions"][first],
                        parts["definitions"][second],
                    ]
                paragraphs.append(
                    parts["answer_template"].format(first=first, second=second)
                )
                text = "\n\n".join(paragraphs)
                rows.append(
                    {
                        "task_id": f"{boundary}:{version}:{order}",
                        "boundary": boundary,
                        "task_version": version,
                        "order": order,
                        "first_class": first,
                        "second_class": second,
                        "lower_class": ladder["lower_class"],
                        "higher_class": ladder["higher_class"],
                        "system_prompt": text,
                        "system_prompt_sha256": digest(text),
                    }
                )
    return rows


def build_schedule(stimuli, system_prompts):
    tasks = defaultdict(list)
    for row in system_prompts:
        tasks[row["boundary"]].append(row["task_id"])
    return [
        {
            "prompt_id": f"{stimulus['stimulus_id']}|{task_id}",
            "stage": stimulus["stage"],
            "boundary": stimulus["boundary"],
            "stimulus_id": stimulus["stimulus_id"],
            "task_id": task_id,
        }
        for stimulus in stimuli
        for task_id in tasks[stimulus["boundary"]]
    ]


# ---------------------------------------------------------------- checks


def check_stimuli(stimuli, ladders):
    """Verify the construction rules in the plan; return counts per check."""
    excluded = re.compile(
        r"\b(" + "|".join(map(re.escape, ladders["excluded_terms"])) + r")\w*",
        re.IGNORECASE,
    )
    slots = dict(country_conditions(ladders))
    names = list(ladders["countries"])
    ids = [row["stimulus_id"] for row in stimuli]
    texts = [row["text"] for row in stimuli]
    require(len(set(ids)) == len(ids), "Duplicate stimulus IDs")
    require(len(set(texts)) == len(texts), "Duplicate stimulus texts")

    skeletons = defaultdict(set)
    cells = defaultdict(set)
    for row in stimuli:
        text = row["text"]
        require(
            "{" not in text and "}" not in text, f"Unfilled slot: {row['stimulus_id']}"
        )
        require(not excluded.search(text), f"Excluded term: {row['stimulus_id']}")
        require(
            text.count(row["evidence"]) == 1,
            f"Evidence not unique: {row['stimulus_id']}",
        )
        skeleton = text.replace(row["evidence"], "{evidence}")
        mentioned = [name for name in names if name in text]
        if row["country"] == NO_COUNTRY:
            require(
                not mentioned, f"Country in no-country stimulus: {row['stimulus_id']}"
            )
        else:
            require(
                mentioned == [row["country"]],
                f"Country mention mismatch: {row['stimulus_id']}",
            )
            require(
                skeleton.count(slots[row["country"]]) == 1,
                f"Country slot not unique: {row['stimulus_id']}",
            )
            skeleton = skeleton.replace(slots[row["country"]], "")
        key = (row["boundary"], row["base_id"])
        skeletons[key].add(skeleton)
        cells[key].add((row["level"], row["wording"], row["country"]))

    expected = set(itertools.product(LEVELS, range(1, WORDINGS + 1), slots))
    for key, found in skeletons.items():
        require(len(found) == 1, f"Base text varies outside slots: {key}")
        require(cells[key] == expected, f"Incomplete design for base: {key}")
    stage_counts = Counter((row["boundary"], row["stage"]) for row in stimuli)
    return {
        "stimuli": len(stimuli),
        "bases": len(skeletons),
        "cells_per_base": len(expected),
        "by_boundary_stage": {
            f"{b}:{s}": n for (b, s), n in sorted(stage_counts.items())
        },
        "checks": [
            "unique stimulus IDs and texts",
            "no unfilled slots or excluded terms",
            "evidence sentence appears exactly once",
            "only the named country appears, once, in the country slot",
            "base text identical outside evidence and country slots",
            "every base has every level, wording and country condition",
        ],
    }


def check_system_prompts(system_prompts, prompts):
    for row in system_prompts:
        text = row["system_prompt"]
        first, second = row["first_class"], row["second_class"]
        require(
            0 <= text.find(first) < text.find(second),
            f"{row['task_id']}: classes out of order",
        )
        parts = prompts["boundaries"][row["boundary"]]
        defined = [parts["definitions"][name] in text for name in (first, second)]
        require(
            all(defined) if row["task_version"] == "defined" else not any(defined),
            f"{row['task_id']}: definitions do not match task version",
        )
        require("country" not in text.lower(), f"{row['task_id']}: mentions country")
    ids = [row["task_id"] for row in system_prompts]
    require(len(set(ids)) == len(ids), "Duplicate task IDs")
    per_boundary = Counter(row["boundary"] for row in system_prompts)
    expected = len(prompts["task_versions"]) * len(prompts["orders"])
    require(set(per_boundary.values()) == {expected}, "Missing task versions or orders")
    return {"system_prompts": len(system_prompts)}


# ---------------------------------------------------------------- build


def build(output_dir, data_dir=DATA):
    require(
        not output_dir.exists() or not any(output_dir.iterdir()),
        f"Output directory is not empty: {output_dir}",
    )
    ladders, prompts, bases = load_inputs(data_dir)
    stimuli = render_stimuli(ladders, bases)
    system_prompts = render_system_prompts(prompts, ladders)
    schedule = build_schedule(stimuli, system_prompts)
    checks = {
        "stimuli": check_stimuli(stimuli, ladders),
        "system_prompts": check_system_prompts(system_prompts, prompts),
        "schedule": dict(
            sorted(Counter(f"{r['boundary']}:{r['stage']}" for r in schedule).items())
        ),
    }
    require(
        len({row["prompt_id"] for row in schedule}) == len(schedule),
        "Duplicate prompts",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "stimuli.csv", STIMULUS_FIELDS, stimuli)
    write_csv(output_dir / "system_prompts.csv", SYSTEM_PROMPT_FIELDS, system_prompts)
    write_csv(output_dir / "schedule.csv", SCHEDULE_FIELDS, schedule)
    write_json(output_dir / "checks.json", checks)
    outputs = ["stimuli.csv", "system_prompts.csv", "schedule.csv", "checks.json"]
    sources = {
        name: data_dir / name
        for name in ["ladders.json", "prompts.json", "bases_pr.csv", "bases_bv.csv"]
    }
    write_json(
        output_dir / "manifest.json",
        {
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "producer_sha256": file_digest(PRODUCER),
            "ladders_version": ladders["version"],
            "prompts_version": prompts["version"],
            "prompts_status": prompts["status"],
            "sources": {name: file_digest(path) for name, path in sources.items()},
            "outputs": {
                name: {"sha256": file_digest(output_dir / name)} for name in outputs
            },
        },
    )
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    args = parser.parse_args(argv)
    checks = build(args.output_dir, args.data_dir)
    print(json.dumps({"stimuli": checks["stimuli"]["stimuli"], **checks["schedule"]}))


if __name__ == "__main__":
    main()
