"""Build the v4 description inputs: frames, evidence (from items.json) and bases.

Frames contain no date. The BV frames name the first group as an armed group,
because both class definitions require one, and describe the other group
neutrally, without saying who acted against whom.

Bases are built deterministically from fixed value lists, none of which
appears in any earlier base (v1–v3; checked against history/). The pilot and
the main split use disjoint value lists. Each value appears once per split:
base i takes value i of the first field and values (i * k) mod n of the
others, for k coprime with n. BV times are generic and are reused across splits.

  python -m geoflip.materials --items inputs/v4/items.json --out inputs/v4
"""

import argparse
import csv
import glob
import json
from pathlib import Path

from .utils import read_json, require, write_csv

ROOT = Path(__file__).resolve().parent.parent

FRAMES = {
    "PR": {
        "f1": "A group of {actors} rallied {place}{country} to oppose {grievance}. {evidence}",
        "f2": "A crowd of {actors} assembled {place}{country} to call attention to {grievance}. {evidence}",
    },
    "BV": {
        "f1": "Members of an armed group and members of another group were involved in a shooting {place}{country} {time}. {evidence}",
        "f2": "A shooting {time} {place}{country} involved members of an armed group and members of another group. {evidence}",
    },
}
CLASSES = {
    "PR": {"lower_class": "Protest", "higher_class": "Riot"},
    "BV": {"lower_class": "Violence against civilians", "higher_class": "Battle"},
}

PR_VALUES = {
    "actors": [
        "postal workers",
        "bakers",
        "librarians",
        "construction workers",
        "cleaners",
        "pharmacists",
        "electricians",
        "bank clerks",
        "airport staff",
        "garment workers",
        "sanitation workers",
        "cooks",
        "painters",
        "musicians",
        "farm laborers",
        "delivery riders",
        "plumbers",
        "textile workers",
        "railway workers",
        "hotel workers",
    ],
    "grievance": [
        "a ban on street trading",
        "hospital closures",
        "higher bus fares",
        "a new water tariff",
        "delayed land titles",
        "cuts to university funding",
        "a hiring freeze",
        "rising medicine prices",
        "late scholarship payments",
        "a new parking fee",
        "a plan to close a library",
        "housing evictions",
        "low crop prices",
        "rising school fees",
        "a new fishing quota",
        "job cuts at a factory",
        "a delayed bridge repair",
        "rising rice prices",
        "fewer bus routes",
        "a new tax on mobile phones",
    ],
    "place": [
        "outside a hospital",
        "near a tax office",
        "in a bus station square",
        "outside a post office",
        "near a city library",
        "outside a district office",
        "in a park near a stadium",
        "outside a mayor's office",
        "near a train depot",
        "in a shopping street",
        "outside a university gate",
        "near a port authority office",
        "outside an electricity company office",
        "on a busy boulevard",
        "near a regional parliament",
        "outside a water utility office",
        "in a market hall",
        "near a customs office",
        "outside a land registry",
        "near a bus depot",
    ],
}
BV_VALUES = {
    "place": [
        "near a cattle ranch",
        "on a forest road",
        "at a mountain pass",
        "near a fishing village",
        "on a riverbank",
        "near an irrigation canal",
        "in a rural hamlet",
        "near a gold mine",
        "on a coastal road",
        "near a sugarcane plantation",
        "in a valley settlement",
        "near a logging camp",
        "on a desert track",
        "near a dam",
        "near a coffee farm",
        "at a highway junction",
        "near an abandoned factory",
        "in a palm grove",
        "on a hilltop",
        "near a salt pan",
    ],
    "time": [
        "at dawn",
        "in the early morning",
        "around noon",
        "in the afternoon",
        "in the late afternoon",
        "at dusk",
        "in the evening",
        "at night",
        "late at night",
        "shortly after midnight",
        "before sunrise",
        "at midday",
    ],
}
SPLITS = {"pilot": 8, "main": 12}
STEPS = {8: (1, 3, 5), 12: (1, 5, 7)}  # coprime multipliers per split size


def old_values():
    seen = {}
    for path in glob.glob(str(ROOT / "history" / "**" / "*.csv"), recursive=True):
        if "bases" not in path:
            continue
        with open(path, newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                for key in ("actors", "grievance", "place", "time"):
                    if key in row:
                        seen.setdefault(key, set()).add(row[key])
    return seen


def build_bases(boundary):
    values = PR_VALUES if boundary == "PR" else BV_VALUES
    fields = list(values)
    out, start = {}, 0
    for split, n in SPLITS.items():
        steps = STEPS[n]
        rows = []
        for i in range(n):
            row = {"base_id": f"{boundary.lower()}4_{split}_{i + 1:02d}"}
            for field, step in zip(fields, steps):
                pool = values[field]
                if boundary == "BV" and field == "time":
                    row[field] = pool[(i * 5) % len(pool)]  # 5 is coprime with 12
                else:
                    row[field] = pool[start + (i * step) % n]
            rows.append(row)
        out[split] = rows
        start += n
    return out


def check_fresh():
    seen = old_values()
    for boundary, values in (("PR", PR_VALUES), ("BV", BV_VALUES)):
        for field, pool in values.items():
            require(
                len(pool) == len(set(pool)), f"{boundary}.{field}: duplicate values"
            )
            if boundary == "BV" and field == "time":
                continue
            reused = set(pool) & seen.get(field, set())
            require(
                not reused, f"{boundary}.{field}: values used before: {sorted(reused)}"
            )
            require(
                len(pool) >= sum(SPLITS.values()), f"{boundary}.{field}: too few values"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    check_fresh()
    items = read_json(args.items)
    descriptions = {
        "version": "v4",
        "notes": "Frames have no date. Evidence from geoflip/items.py (ACLED-attested acts). Bases from geoflip/materials.py.",
        "boundaries": {
            b: {**CLASSES[b], "frames": FRAMES[b], "evidence": items["evidence"][b]}
            for b in ("PR", "BV")
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "descriptions.json").write_text(
        json.dumps(descriptions, indent=2, ensure_ascii=False) + "\n"
    )
    (args.out / "bases").mkdir(exist_ok=True)
    for boundary in ("PR", "BV"):
        for split, rows in build_bases(boundary).items():
            write_csv(
                args.out / "bases" / f"{boundary}_{split}.csv", list(rows[0]), rows
            )
            for row in rows[:3]:
                print(split, row)


if __name__ == "__main__":
    main()
