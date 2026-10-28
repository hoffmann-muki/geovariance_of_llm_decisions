#!/usr/bin/env python3
"""Prepare auditable Experiment A candidates. Requires Python 3.10+; no packages.

Raw notes are never rewritten. Similarity grouping is a conservative sampling
heuristic, not an assertion that two reports describe the same historical event.
"""

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS = {
    "Violence against civilians": "V",
    "Battles": "B",
    "Explosions/Remote violence": "E",
    "Protests": "P",
    "Riots": "R",
    "Strategic developments": "S",
}
REQUIRED = {
    "event_id_cnty",
    "event_date",
    "country",
    "region",
    "event_type",
    "notes",
    "sub_event_type",
    "admin1",
    "location",
    "actor1",
    "actor2",
    "inter1",
    "inter2",
}
NONEMPTY = ("event_id_cnty", "event_date", "country", "region", "event_type", "notes")
ACTOR_GROUPS = {
    "state forces": "state",
    "rebel group": "non_state_armed",
    "political militia": "non_state_armed",
    "identity militia": "non_state_armed",
    "rioters": "rioters",
    "protesters": "protesters",
    "civilians": "civilians",
    "external/other forces": "external_other",
}
SAMPLE_FIELDS = [
    "sample_id",
    "sample_role",
    "study_group",
    "selection_rank",
    "event_id_cnty",
    "event_group_id",
    "source_file",
    "source_record",
    "country",
    "region",
    "event_date",
    "year_month",
    "event_type",
    "label",
    "sub_event_type",
    "actor1",
    "actor2",
    "inter1",
    "inter2",
    "actor_group",
    "admin1",
    "location",
    "word_count",
    "length_bin",
    "notes",
    "target_word_count",
    "word_difference_from_target",
]
ISSUE_FIELDS = [
    "source_file",
    "source_record",
    "event_id_cnty",
    "country",
    "reason",
    "retained_event_id",
]


# Initial allocation settings; accepted targets are recorded in the review adjudications.
DEFAULT_CONFIG = {
    "seed": 42,
    "main_per_cell": 20,
    "main_per_event_type": {"Riots": 16},
    "pilot_per_cell": 2,
    "reserve_per_cell": 5,
    "primary_subtypes": {
        "Protests": "Peaceful protest",
        "Riots": "Violent demonstration",
        "Battles": "Armed clash",
        "Violence against civilians": "Attack",
    },
    "match_length_bins": True,
    "pilot_countries": ["China", "India", "Libya", "Mexico"],
    "country_regions": {
        "Libya": "Northern Africa",
        "Morocco": "Northern Africa",
        "Mexico": "North America",
        "United States": "North America",
        "Brazil": "South America",
        "Colombia": "South America",
        "India": "South Asia",
        "Pakistan": "South Asia",
        "China": "East Asia",
        "South Korea": "East Asia",
    },
    "country_aliases": {"United States of America": "United States"},
    "samples": [
        {
            "name": "protests_riots",
            "countries": [
                "Libya",
                "Morocco",
                "Mexico",
                "United States",
                "Brazil",
                "Colombia",
                "India",
                "Pakistan",
                "China",
                "South Korea",
            ],
            "event_types": ["Protests", "Riots"],
        },
        {
            "name": "battles_civilian_violence",
            "countries": ["Libya", "Mexico", "Brazil", "Colombia", "India", "Pakistan"],
            "event_types": ["Battles", "Violence against civilians"],
        },
    ],
    "near_duplicate_similarity": 0.90,
    "source_token_overlap": 0.75,
    "near_duplicate_days": 7,
    "length_bins_words": [40, 80],
}


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_digest(path):
    with path.open("rb") as stream:
        return (
            hashlib.file_digest(stream, "sha256").hexdigest()
            if hasattr(hashlib, "file_digest")
            else digest_bytes(stream)
        )


def digest_bytes(stream):
    result = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        result.update(chunk)
    return result.hexdigest()


def text_key(text):
    """Comparison-only normalization; never used as model input."""
    return unicodedata.normalize("NFKC", " ".join(text.split())).casefold()


def clean(value):
    return " ".join((value or "").split())


def load_config(path=None):
    config = json.loads(path.read_text(encoding="utf-8")) if path else DEFAULT_CONFIG
    for key in (
        "main_per_cell",
        "pilot_per_cell",
        "reserve_per_cell",
        "near_duplicate_days",
    ):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if config["main_per_cell"] == 0 or type(config["seed"]) is not int:
        raise ValueError("main_per_cell must be positive and seed must be an integer")
    if not 0 < config["near_duplicate_similarity"] <= 1:
        raise ValueError("near_duplicate_similarity must be in (0, 1]")
    if not 0 <= config.get("source_token_overlap", 0.75) <= 1:
        raise ValueError("source_token_overlap must be in [0, 1]")
    bins = config["length_bins_words"]
    if (
        not bins
        or any(type(x) is not int or x <= 0 for x in bins)
        or sorted(set(bins)) != bins
    ):
        raise ValueError("length_bins_words must contain increasing positive integers")
    countries = set(config["country_regions"])
    if not set(config["pilot_countries"]) <= countries:
        raise ValueError("pilot_countries must belong to country_regions")
    cells = set()
    names = set()
    for sample in config["samples"]:
        if (
            not sample["countries"]
            or not sample["event_types"]
            or sample["name"] in names
        ):
            raise ValueError(
                "Samples need unique names and nonempty country and event lists"
            )
        names.add(sample["name"])
        for country in sample["countries"]:
            for label in sample["event_types"]:
                if (
                    country not in countries
                    or label not in LABELS
                    or (country, label) in cells
                ):
                    raise ValueError(
                        f"Unknown or overlapping sample cell: {country}/{label}"
                    )
                cells.add((country, label))
    if not cells:
        raise ValueError("At least one sampling cell is required")
    subtypes = config.get("primary_subtypes", {})
    if subtypes and (
        set(subtypes) != {event_type for _, event_type in cells}
        or any(not isinstance(s, str) or not s.strip() for s in subtypes.values())
    ):
        raise ValueError(
            "primary_subtypes must specify one subtype for every sampled event type"
        )
    if type(config.get("match_length_bins", False)) is not bool:
        raise ValueError("match_length_bins must be a boolean")
    for event_type, count in config.get("main_per_event_type", {}).items():
        if (
            event_type not in {t for _, t in cells}
            or type(count) is not int
            or count <= 0
        ):
            raise ValueError(
                "main_per_event_type needs positive quotas for sampled event types"
            )
    return config


def length_bin(count, boundaries):
    lower = 1
    for upper in boundaries:
        if count <= upper:
            return f"{lower}-{upper}"
        lower = upper + 1
    return f"{lower}+"


def issue(row, reason, retained=""):
    return {
        **{k: row.get(k, "") for k in ISSUE_FIELDS},
        "reason": reason,
        "retained_event_id": retained,
    }


def read_sources(input_dir, config):
    paths = sorted(input_dir.glob("*.csv"))
    if not paths:
        raise ValueError(f"No CSV files in {input_dir}")
    rows, inventory, exclusions = [], [], []
    first_schema = None
    for path in paths:
        missing = Counter()
        file_rows = []
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream, strict=True)
            fields = reader.fieldnames
            if (
                not fields
                or len(set(fields)) != len(fields)
                or not REQUIRED <= set(fields)
            ):
                raise ValueError(
                    f"Invalid schema in {path.name}; missing {sorted(REQUIRED - set(fields or []))}"
                )
            if first_schema is not None and set(fields) != set(first_schema):
                raise ValueError(f"Schema differs across CSVs: {path.name}")
            first_schema = fields
            for ordinal, raw in enumerate(reader, 1):
                if None in raw or any(value is None for value in raw.values()):
                    raise ValueError(f"Malformed CSV record {ordinal} in {path.name}")
                missing.update(k for k, value in raw.items() if not value.strip())
                row = {k: clean(v) for k, v in raw.items()}
                row["notes"] = raw["notes"]
                row.update(source_file=path.name, source_record=ordinal)
                row["country"] = config.get("country_aliases", {}).get(
                    row["country"], row["country"]
                )
                event_names = {name.casefold(): name for name in LABELS}
                row["event_type"] = event_names.get(
                    row["event_type"].casefold(), row["event_type"]
                )
                row["label"] = LABELS.get(row["event_type"], "")
                row["actor_group"] = ACTOR_GROUPS.get(
                    row["inter1"].casefold(), "unknown"
                )
                row["word_count"] = len(row["notes"].split())
                row["length_bin"] = length_bin(
                    row["word_count"], config["length_bins_words"]
                )
                row["_text_key"] = text_key(row["notes"])
                row["_record_hash"] = digest(
                    json.dumps(raw, sort_keys=True, ensure_ascii=False)
                )
                reasons = [f"missing_{k}" for k in NONEMPTY if not row[k].strip()]
                if row["event_type"] and not row["label"]:
                    reasons.append("unknown_event_type")
                try:
                    parsed_date = date.fromisoformat(row["event_date"])
                    if parsed_date.isoformat() != row["event_date"]:
                        raise ValueError("Noncanonical date")
                    row["_ordinal_date"] = parsed_date.toordinal()
                    row["year_month"] = parsed_date.strftime("%Y-%m")
                except ValueError:
                    reasons.append("invalid_event_date")
                    row["year_month"] = ""
                expected_region = config["country_regions"].get(row["country"])
                if expected_region and row["region"] != expected_region:
                    reasons.append("country_region_mismatch")
                row["_valid"] = not reasons
                for reason in reasons:
                    exclusions.append(issue(row, reason))
                file_rows.append(row)
        dates = [r["event_date"] for r in file_rows if r.get("_ordinal_date")]
        inventory.append(
            {
                "file": path.name,
                "sha256": file_digest(path),
                "bytes": path.stat().st_size,
                "rows": len(file_rows),
                "valid_rows": sum(r["_valid"] for r in file_rows),
                "columns": fields,
                "missing_counts": {k: missing[k] for k in fields},
                "date_min": min(dates, default=None),
                "date_max": max(dates, default=None),
                "regions": dict(
                    sorted(Counter(r["region"] for r in file_rows).items())
                ),
            }
        )
        rows.extend(file_rows)
    return rows, inventory, exclusions


def deduplicate(rows, exclusions):
    """Exclude contradictory IDs/notes; retain one deterministic exact duplicate."""
    by_id = defaultdict(list)
    for row in rows:
        if row["_valid"]:
            by_id[row["event_id_cnty"]].append(row)
    unique_ids = []
    for event_id, members in sorted(by_id.items()):
        if len({r["_record_hash"] for r in members}) > 1:
            exclusions.extend(issue(r, "conflicting_event_id") for r in members)
        else:
            unique_ids.append(members[0])
            exclusions.extend(
                issue(r, "duplicate_event_id", event_id) for r in members[1:]
            )
    by_notes = defaultdict(list)
    for row in unique_ids:
        by_notes[row["_text_key"]].append(row)
    kept = []
    for members in by_notes.values():
        members.sort(key=lambda r: r["event_id_cnty"])
        if len({r["label"] for r in members}) > 1:
            exclusions.extend(issue(r, "conflicting_note_labels") for r in members)
        else:
            kept.append(members[0])
            exclusions.extend(
                issue(r, "duplicate_notes", members[0]["event_id_cnty"])
                for r in members[1:]
            )
    return sorted(kept, key=lambda r: r["event_id_cnty"])


def group_related(rows, config):
    parent = {r["event_id_cnty"]: r["event_id_cnty"] for r in rows}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    buckets = defaultdict(list)
    for row in rows:
        if row["country"] in config["country_regions"]:
            buckets[(row["country"], text_key(row["admin1"]))].append(row)
    pairs = []
    for members in buckets.values():
        members.sort(key=lambda r: (r["_ordinal_date"], r["event_id_cnty"]))
        tokens = {
            r["event_id_cnty"]: set(re.findall(r"\w+", r["_text_key"])) for r in members
        }
        start = 0
        for index, row in enumerate(members):
            while (
                members[start]["_ordinal_date"]
                < row["_ordinal_date"] - config["near_duplicate_days"]
            ):
                start += 1
            matcher = SequenceMatcher(None, "", row["_text_key"], autojunk=False)
            for earlier in members[start:index]:
                if find(earlier["event_id_cnty"]) == find(row["event_id_cnty"]):
                    continue
                left_words, right_words = (
                    tokens[earlier["event_id_cnty"]],
                    tokens[row["event_id_cnty"]],
                )
                # Candidate retrieval only. The independent selected-record audit
                # uses full text and no token/admin/location blocking.
                if 2 * len(left_words & right_words) < config.get(
                    "source_token_overlap", 0.75
                ) * (len(left_words) + len(right_words)):
                    continue
                matcher.set_seq1(earlier["_text_key"])
                threshold = config["near_duplicate_similarity"]
                if (
                    matcher.real_quick_ratio() < threshold
                    or matcher.quick_ratio() < threshold
                ):
                    continue
                similarity = matcher.ratio()
                if similarity < threshold:
                    continue
                left, right = find(earlier["event_id_cnty"]), find(row["event_id_cnty"])
                parent[max(left, right)] = min(left, right)
                pairs.append(
                    similarity_pair(earlier, row, similarity, "source_admin_window")
                )
    for row in rows:
        row["event_group_id"] = "group_" + digest(find(row["event_id_cnty"]))[:16]
    for pair in pairs:
        pair["event_group_id"] = "group_" + digest(find(pair["event_id_1"]))[:16]
    return sorted(pairs, key=lambda p: (p["event_id_1"], p["event_id_2"]))


def similarity_pair(left, right, similarity, reason):
    return {
        "event_id_1": left["event_id_cnty"],
        "event_id_2": right["event_id_cnty"],
        "country": left["country"],
        "date_1": left["event_date"],
        "date_2": right["event_date"],
        "admin1_1": left["admin1"],
        "admin1_2": right["admin1"],
        "location_1": left["location"],
        "location_2": right["location"],
        "days_apart": abs(left["_ordinal_date"] - right["_ordinal_date"]),
        "similarity": round(similarity, 6),
        "reason": reason,
    }


def sampling_pool(rows, config):
    """Freeze one eligible representative per text family before allocating roles."""
    cells = {
        (c, t)
        for s in config["samples"]
        for c in s["countries"]
        for t in s["event_types"]
    }
    subtypes = config.get("primary_subtypes", {})
    primary = [
        r
        for r in rows
        if (r["country"], r["event_type"]) in cells
        and (not subtypes or r["sub_event_type"] == subtypes[r["event_type"]])
    ]
    representatives = {}
    for row in sorted(
        primary, key=lambda r: digest(f"{config['seed']}/pool/{r['event_id_cnty']}")
    ):
        representatives.setdefault(row["event_group_id"], row)
    return sorted(representatives.values(), key=lambda r: r["event_id_cnty"])


def audit_similarity(selected, config):
    """Independently compare ALL selected roles within country/date windows.

    No location, administrative-region, event-type or group-ID blocking is used.
    """
    countries = defaultdict(list)
    for rows in selected.values():
        for row in rows:
            countries[row["country"]].append(row)
    conflicts = []
    compared = 0
    highest = 0.0
    for rows in countries.values():
        rows.sort(key=lambda r: (r["_ordinal_date"], r["event_id_cnty"]))
        for index, right in enumerate(rows):
            for left in reversed(rows[:index]):
                if (
                    right["_ordinal_date"] - left["_ordinal_date"]
                    > config["near_duplicate_days"]
                ):
                    break
                compared += 1
                similarity = SequenceMatcher(
                    None, left["_text_key"], right["_text_key"], autojunk=False
                ).ratio()
                highest = max(highest, similarity)
                if similarity >= config["near_duplicate_similarity"]:
                    pair = similarity_pair(
                        left, right, similarity, "selected_country_window"
                    )
                    pair.update(role_1=left["sample_role"], role_2=right["sample_role"])
                    conflicts.append(pair)
    return conflicts, {
        "pairs_compared": compared,
        "flagged_pairs": len(conflicts),
        "maximum_similarity": round(highest, 6),
        "passed": not conflicts,
    }


def merge_sampling_families(rows, conflicts, pairs):
    """Add independently detected links without deleting any source records."""
    by_id = {r["event_id_cnty"]: r for r in rows}
    parent = {r["event_group_id"]: r["event_group_id"] for r in rows}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for pair in conflicts:
        left = find(by_id[pair["event_id_1"]]["event_group_id"])
        right = find(by_id[pair["event_id_2"]]["event_group_id"])
        parent[max(left, right)] = min(left, right)
        pairs.append({k: v for k, v in pair.items() if not k.startswith("role_")})
    minimum_id = {}
    for row in rows:
        root = find(row["event_group_id"])
        minimum_id[root] = min(
            minimum_id.get(root, row["event_id_cnty"]), row["event_id_cnty"]
        )
    for row in rows:
        row["event_group_id"] = (
            "group_" + digest(minimum_id[find(row["event_group_id"])])[:16]
        )
    for pair in pairs:
        pair["event_group_id"] = by_id[pair["event_id_1"]]["event_group_id"]


def proportional_quotas(capacities, total, seed, context):
    available = sum(capacities.values())
    count = min(total, available)
    if not available:
        return {key: 0 for key in capacities}
    quotas = {key: count * value // available for key, value in capacities.items()}
    order = sorted(
        capacities,
        key=lambda key: (
            -(count * capacities[key] % available),
            digest(f"{seed}/{context}/{key}"),
        ),
    )
    for key in order[: count - sum(quotas.values())]:
        quotas[key] += 1
    return quotas


def stratified_sample(rows, requested, used_groups, seed, context):
    """Proportional joint-stratum allocation; deterministic seeded tie breaking."""
    representatives = {}
    priority = lambda r: digest(f"{seed}/{context}/{r['event_id_cnty']}")
    for row in sorted(rows, key=priority):
        group = row["event_group_id"]
        if group not in used_groups:
            representatives.setdefault(group, row)
    strata = defaultdict(list)
    for row in representatives.values():
        strata[(row["sub_event_type"], row["year_month"], row["length_bin"])].append(
            row
        )
    available = len(representatives)
    count = min(requested, available)
    if not count:
        return [], available
    quotas = {key: count * len(values) // available for key, values in strata.items()}
    order = sorted(
        strata,
        key=lambda key: (
            -(count * len(strata[key]) % available),
            digest(f"{seed}/{context}/{key}"),
        ),
    )
    for key in order[: count - sum(quotas.values())]:
        quotas[key] += 1
    chosen = [
        row
        for key in sorted(strata)
        for row in sorted(strata[key], key=priority)[: quotas[key]]
    ]
    chosen.sort(key=priority)
    used_groups.update(r["event_group_id"] for r in chosen)
    return chosen, available


def nearest_length_sample(rows, targets, used_groups, seed, context):
    """Minimum-total-distance assignment to ordered shared word-count targets."""
    ordered = sorted(
        (r for r in rows if r["event_group_id"] not in used_groups),
        key=lambda r: (
            r["word_count"],
            digest(f"{seed}/{context}/{r['event_id_cnty']}"),
        ),
    )
    if len(targets) > len(ordered):
        raise ValueError("Insufficient representatives for shared word-count targets")
    width = len(ordered)
    previous = [0] * (width + 1)
    decisions = []
    for target in targets:
        current = [float("inf")] * (width + 1)
        take = bytearray(width + 1)
        for column, row in enumerate(ordered, 1):
            cost = previous[column - 1] + abs(row["word_count"] - target)
            current[column] = current[column - 1]
            if cost < current[column]:
                current[column] = cost
                take[column] = 1
        previous = current
        decisions.append(take)
    chosen = []
    column = width
    for index in reversed(range(len(targets))):
        while not decisions[index][column]:
            column -= 1
        row = ordered[column - 1]
        chosen.append(
            {
                **row,
                "target_word_count": targets[index],
                "word_difference_from_target": row["word_count"] - targets[index],
            }
        )
        column -= 1
    used_groups.update(r["event_group_id"] for r in chosen)
    return chosen


def sample_events(rows, config):
    pools = defaultdict(list)
    for row in rows:
        pools[(row["country"], row["event_type"])].append(row)
    cells = sorted(
        (sample["name"], country, event_type)
        for sample in config["samples"]
        for country in sample["countries"]
        for event_type in sample["event_types"]
    )
    selected = {"main": [], "pilot": [], "reserve": []}
    summaries = {}
    used_groups = set()
    for study, country, event_type in cells:
        pool = pools[(country, event_type)]
        summaries[(study, country, event_type)] = {
            "study_group": study,
            "country": country,
            "event_type": event_type,
            "eligible_events": len(pool),
            "eligible_groups": len({r["event_group_id"] for r in pool}),
        }
    targets = []
    matching = {}
    word_targets = {}
    # Pilot first; reserve selection is last so it cannot consume a main quota.
    for role in ("pilot", "main", "reserve"):
        if role == "main" and config.get("match_length_bins", False):
            bins = [
                length_bin(n, config["length_bins_words"])
                for n in [1, *[b + 1 for b in config["length_bins_words"]]]
            ]
            for study in config["samples"]:
                for event_type in study["event_types"]:
                    availability = {
                        country: Counter(
                            r["length_bin"]
                            for r in pools[(country, event_type)]
                            if r["event_group_id"] not in used_groups
                        )
                        for country in study["countries"]
                    }
                    capacities = {
                        key: min(counts[key] for counts in availability.values())
                        for key in bins
                    }
                    requested_main = config.get("main_per_event_type", {}).get(
                        event_type, config["main_per_cell"]
                    )
                    quotas = proportional_quotas(
                        capacities,
                        requested_main,
                        config["seed"],
                        f"match/{study['name']}/{event_type}",
                    )
                    reserve_quotas = proportional_quotas(
                        quotas,
                        config["reserve_per_cell"],
                        config["seed"],
                        f"reserve-match/{study['name']}/{event_type}",
                    )
                    matching[(study["name"], event_type, "main")] = quotas
                    matching[(study["name"], event_type, "reserve")] = reserve_quotas
                    for key in bins:
                        country_lengths = [
                            sorted(
                                r["word_count"]
                                for r in pools[(country, event_type)]
                                if r["length_bin"] == key
                                and r["event_group_id"] not in used_groups
                            )
                            for country in study["countries"]
                        ]
                        size = quotas[key]
                        anchors = [
                            int(
                                statistics.median(
                                    lengths[(2 * i + 1) * len(lengths) // (2 * size)]
                                    for lengths in country_lengths
                                )
                                + 0.5
                            )
                            for i in range(size)
                        ]
                        word_targets[(study["name"], event_type, key)] = anchors
                        targets.append(
                            {
                                "study_group": study["name"],
                                "event_type": event_type,
                                "length_bin": key,
                                "minimum_available_groups": capacities[key],
                                "limiting_countries": ";".join(
                                    sorted(
                                        c
                                        for c, counts in availability.items()
                                        if counts[key] == capacities[key]
                                    )
                                ),
                                "main_target_per_country": quotas[key],
                                "reserve_target_per_country": reserve_quotas[key],
                                "target_word_counts": json.dumps(anchors),
                            }
                        )
        for cell in cells:
            study, country, event_type = cell
            requested = config[f"{role}_per_cell"]
            if role == "main":
                requested = config.get("main_per_event_type", {}).get(
                    event_type, requested
                )
            if role == "pilot" and country not in config["pilot_countries"]:
                requested = 0
            context = f"{role}/{study}/{country}/{event_type}"
            if (study, event_type, role) in matching:
                quotas = matching[(study, event_type, role)]
                available = len(
                    {
                        r["event_group_id"]
                        for r in pools[(country, event_type)]
                        if r["event_group_id"] not in used_groups
                    }
                )
                chosen = []
                for key, count in quotas.items():
                    bin_rows = [
                        r
                        for r in pools[(country, event_type)]
                        if r["length_bin"] == key
                    ]
                    if role == "main":
                        part = nearest_length_sample(
                            bin_rows,
                            word_targets[(study, event_type, key)],
                            used_groups,
                            config["seed"],
                            context + "/" + key,
                        )
                    else:
                        part, _ = stratified_sample(
                            bin_rows,
                            count,
                            used_groups,
                            config["seed"],
                            context + "/" + key,
                        )
                    chosen.extend(part)
                chosen.sort(
                    key=lambda r: digest(
                        f"{config['seed']}/{context}/{r['event_id_cnty']}"
                    )
                )
            else:
                chosen, available = stratified_sample(
                    pools[(country, event_type)],
                    requested,
                    used_groups,
                    config["seed"],
                    context,
                )
            summaries[cell].update(
                {
                    f"{role}_requested": requested,
                    f"{role}_selected": len(chosen),
                    f"{role}_shortfall": requested - len(chosen),
                    f"{role}_available_groups": available,
                }
            )
            for rank, row in enumerate(chosen, 1):
                selected[role].append(
                    {
                        **row,
                        "sample_id": f"a_{role}_{row['event_id_cnty']}",
                        "sample_role": role,
                        "study_group": study,
                        "selection_rank": rank,
                    }
                )
    return selected, list(summaries.values()), targets


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sample_balance(eligible, selected, config, pool):
    counts = defaultdict(Counter)
    cells = {
        (c, t)
        for s in config["samples"]
        for c in s["countries"]
        for t in s["event_types"]
    }
    for role, rows in [
        ("population", eligible),
        ("sampling_pool", pool),
        *selected.items(),
    ]:
        for row in rows:
            if (row["country"], row["event_type"]) not in cells:
                continue
            for dimension in (
                "sub_event_type",
                "year_month",
                "length_bin",
                "actor_group",
            ):
                key = (row["country"], row["event_type"], dimension, row[dimension])
                counts[key][role] += 1
    return [
        {
            **dict(zip(("country", "event_type", "dimension", "value"), key)),
            **{
                f"{role}_count": values[role]
                for role in ("population", "sampling_pool", "main", "pilot", "reserve")
            },
        }
        for key, values in sorted(counts.items())
    ]


def comparability_summary(eligible, pool, selected, targets, config):
    target_map = {
        (r["study_group"], r["event_type"], r["length_bin"]): r for r in targets
    }
    summaries = []
    for sample in config["samples"]:
        for country in sample["countries"]:
            for event_type in sample["event_types"]:
                source = [
                    r
                    for r in eligible
                    if (r["country"], r["event_type"]) == (country, event_type)
                ]
                members = [
                    r
                    for r in pool
                    if (r["country"], r["event_type"]) == (country, event_type)
                ]
                main = [
                    r
                    for r in selected["main"]
                    if (r["country"], r["event_type"]) == (country, event_type)
                ]
                allowed = config.get("primary_subtypes", {}).get(event_type, "")
                subtype_ok = (
                    all(r["sub_event_type"] == allowed for r in main)
                    if allowed
                    else True
                )
                length_ok = True
                for key in [
                    t[2] for t in target_map if t[:2] == (sample["name"], event_type)
                ]:
                    target = target_map[(sample["name"], event_type, key)]
                    observed = sum(r["length_bin"] == key for r in main)
                    length_ok = (
                        length_ok and observed == target["main_target_per_country"]
                    )
                summaries.append(
                    {
                        "study_group": sample["name"],
                        "country": country,
                        "event_type": event_type,
                        "primary_subtype": allowed,
                        "source_events": len(source),
                        "sampling_pool_events": len(members),
                        "main_selected": len(main),
                        "selected_subtypes": ";".join(
                            sorted({r["sub_event_type"] for r in main})
                        ),
                        "source_median_words": (
                            statistics.median(r["word_count"] for r in source)
                            if source
                            else ""
                        ),
                        "main_median_words": (
                            statistics.median(r["word_count"] for r in main)
                            if main
                            else ""
                        ),
                        "main_min_words": min(
                            (r["word_count"] for r in main), default=""
                        ),
                        "main_max_words": max(
                            (r["word_count"] for r in main), default=""
                        ),
                        "mean_absolute_word_difference": (
                            round(
                                statistics.mean(
                                    abs(r["word_difference_from_target"]) for r in main
                                ),
                                3,
                            )
                            if main
                            and all("word_difference_from_target" in r for r in main)
                            else ""
                        ),
                        "maximum_absolute_word_difference": max(
                            (
                                abs(r["word_difference_from_target"])
                                for r in main
                                if "word_difference_from_target" in r
                            ),
                            default="",
                        ),
                        "primary_subtype_met": subtype_ok,
                        "length_targets_met": length_ok,
                    }
                )
    return summaries


def run(input_dir, output_dir, config):
    if output_dir.exists():
        raise ValueError(
            f"Output directory already exists: {output_dir}. Use a new --output-dir to preserve review work."
        )
    if output_dir.resolve().is_relative_to(input_dir.resolve()):
        raise ValueError("Output directory must be outside the raw input directory")
    print("Reading and validating source CSVs...", flush=True)
    raw, inventory, exclusions = read_sources(input_dir, config)
    eligible = deduplicate(raw, exclusions)
    print(
        f"Grouping similar reports in {len(config['country_regions'])} selected countries...",
        flush=True,
    )
    pairs = group_related(eligible, config)
    audit_history = []
    for attempt in range(1, 11):
        pool = sampling_pool(eligible, config)
        selected, sampling, targets = sample_events(pool, config)
        conflicts, audit = audit_similarity(selected, config)
        audit_history.append({"pass": attempt, **audit})
        if not conflicts or attempt == 10:
            break
        print(
            f"Similarity audit pass {attempt}: linking {len(conflicts)} additional text-family pairs and resampling...",
            flush=True,
        )
        merge_sampling_families(eligible, conflicts, pairs)
    comparable = comparability_summary(eligible, pool, selected, targets, config)
    all_selected = [
        row for role in ("main", "pilot", "reserve") for row in selected[role]
    ]
    if len({r["event_group_id"] for r in all_selected}) != len(all_selected):
        raise RuntimeError(
            "A related-event group occurs more than once across sample roles"
        )
    main_shortfalls = sum(s["main_shortfall"] for s in sampling)
    comparability_passed = all(
        r["primary_subtype_met"] and r["length_targets_met"] for r in comparable
    )
    status = "prepared"
    if main_shortfalls:
        status = "quota_shortfall"
    elif not audit["passed"] or not comparability_passed:
        status = "validation_failed"
    report = {
        "raw_rows": len(raw),
        "valid_rows": sum(r["_valid"] for r in raw),
        "eligible_rows": len(eligible),
        "excluded_rows": len(raw) - len(eligible),
        "exclusion_reasons": dict(
            sorted(Counter(e["reason"] for e in exclusions).items())
        ),
        "input_files": inventory,
        "near_duplicate_pairs": len(pairs),
        "related_groups": len({p["event_group_id"] for p in pairs}),
        "sample_counts": {role: len(values) for role, values in selected.items()},
        "main_quota_shortfall": main_shortfalls,
        "sampling_pool_rows": len(pool),
        "primary_subtypes": config.get("primary_subtypes", {}),
        "length_matching_enabled": config.get("match_length_bins", False),
        "comparability_passed": comparability_passed,
        "matching_targets": targets,
        "similarity_audit": {
            **audit,
            "passes": audit_history,
            "additional_links": sum(a["flagged_pairs"] for a in audit_history[:-1]),
        },
        "reserve_shortfall": sum(s["reserve_shortfall"] for s in sampling),
        "status": status,
        "unknown_actor_categories": dict(
            sorted(
                Counter(
                    r["inter1"] for r in eligible if r["actor_group"] == "unknown"
                ).items()
            )
        ),
        "warnings": [
            "Region membership comes from record metadata, not filenames.",
            "Text families limit repeated wording, not necessarily unique historical episodes; review is required.",
            "Matching subtypes and word-length bands does not equalize actors, timing, narrative style, or exact word counts.",
            "Country comparisons on natural reports remain observational; controlled descriptions provide the direct geography test.",
            "Preprocessing creates candidates; acceptance requires a separate label and edit-suitability review.",
        ],
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".preprocessing-", dir=output_dir.parent
    ) as temp:
        stage = Path(temp) / "result"
        stage.mkdir()
        for role, name in (
            ("main", "candidates.csv"),
            ("pilot", "pilot.csv"),
            ("reserve", "reserves.csv"),
        ):
            write_csv(stage / name, SAMPLE_FIELDS, selected[role])
        write_csv(
            stage / "exclusions.csv",
            ISSUE_FIELDS,
            sorted(
                exclusions,
                key=lambda r: (r["source_file"], r["source_record"], r["reason"]),
            ),
        )
        pair_fields = [
            "event_group_id",
            "event_id_1",
            "event_id_2",
            "country",
            "date_1",
            "date_2",
            "admin1_1",
            "admin1_2",
            "location_1",
            "location_2",
            "days_apart",
            "similarity",
            "reason",
        ]
        write_csv(
            stage / "related_event_pairs.csv",
            pair_fields,
            sorted(pairs, key=lambda p: (p["event_id_1"], p["event_id_2"])),
        )
        write_csv(
            stage / "sampling_pool.csv",
            [
                f
                for f in SAMPLE_FIELDS
                if f
                not in {
                    "notes",
                    "sample_id",
                    "sample_role",
                    "study_group",
                    "selection_rank",
                    "target_word_count",
                    "word_difference_from_target",
                }
            ],
            pool,
        )
        write_csv(stage / "sampling_summary.csv", list(sampling[0]), sampling)
        write_csv(
            stage / "sample_balance.csv",
            [
                "country",
                "event_type",
                "dimension",
                "value",
                "population_count",
                "sampling_pool_count",
                "main_count",
                "pilot_count",
                "reserve_count",
            ],
            sample_balance(eligible, selected, config, pool),
        )
        lines = [
            "# Preprocessing Quality Report",
            "",
            f"Status: **{report['status']}**",
            "",
            f"- Raw records: {len(raw):,}",
            f"- Eligible records after validation and deduplication: {len(eligible):,}",
            f"- Excluded input-text duplicates or invalid records: {report['excluded_rows']:,}",
            f"- Text-family links recorded: {len(pairs):,}",
            f"- Restricted sampling-pool representatives: {len(pool):,}",
            f"- Main candidates: {len(selected['main']):,}",
            f"- Pilot candidates: {len(selected['pilot']):,}",
            f"- Reserve candidates: {len(selected['reserve']):,}",
            f"- Unfilled main quota: {main_shortfalls}",
            "",
            "## Separation and Comparability Checks",
            "",
            f"- Independent selected-text audit: {'passed' if audit['passed'] else 'failed'}.",
            f"- Pairs compared across all selected roles: {audit['pairs_compared']:,}.",
            f"- Remaining similar pairs: {audit['flagged_pairs']} (threshold {config['near_duplicate_similarity']}, within {config['near_duplicate_days']} days in the same country).",
            f"- Maximum similarity in the final selection: {audit['maximum_similarity']:.3f}.",
            f"- Audit/selection passes: {len(audit_history)}. Additional links: {report['similarity_audit']['additional_links']}.",
            f"- Shared-subtype and length-band targets: {'met' if comparability_passed else 'not met'}.",
            f"- Length matching enabled: {config.get('match_length_bins', False)}.",
            f"- Reserve quota shortfall: {report['reserve_shortfall']}.",
            "",
            "## Source Inventory",
            "",
            "| File | Records | Date range | Regions |",
            "| --- | ---: | --- | --- |",
        ]
        for source in inventory:
            regions = "; ".join(
                f"{name}: {count:,}" for name, count in source["regions"].items()
            )
            lines.append(
                f"| {source['file']} | {source['rows']:,} | {source['date_min']} to {source['date_max']} | {regions} |"
            )
        lines += [
            "",
            "## Sampling",
            "",
            "| Country | Event type | Eligible primary groups | Main | Pilot | Reserve |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for cell in sorted(sampling, key=lambda s: (s["country"], s["event_type"])):
            lines.append(
                f"| {cell['country']} | {cell['event_type']} | {cell['eligible_groups']} | {cell['main_selected']} | {cell['pilot_selected']} | {cell['reserve_selected']} |"
            )
        lines += [
            "",
            "## Primary Subtypes and Description Length",
            "",
            "| Country | Event type | Selected subtype | Main median words | Main word range | Mean absolute word difference from target | Length bands met |",
            "| --- | --- | --- | ---: | --- | ---: | --- |",
        ]
        for row in sorted(comparable, key=lambda r: (r["country"], r["event_type"])):
            lines.append(
                f"| {row['country']} | {row['event_type']} | {row['selected_subtypes']} | {row['main_median_words']} | {row['main_min_words']}-{row['main_max_words']} | {row['mean_absolute_word_difference']} | {row['length_targets_met']} |"
            )
        if targets:
            lines += [
                "",
                "Each event class uses identical main length-band quotas across its countries. The manifest records quotas, limiting countries, and word-count targets; `sample_balance.csv` records observed composition.",
                "",
            ]
        lines += ["", "## Review and Limitations", ""] + [
            f"- {warning}" for warning in report["warnings"]
        ]
        lines += [
            "- Initial text-family grouping spans locations within country/admin1 over "
            + str(config["near_duplicate_days"])
            + " days. An independent audit then checks all selected records within country, across administrative regions and event classes. Additional links trigger deterministic resampling.",
            "- At most one example per detected group is selected across main, pilot, and reserve roles.",
            "- Non-primary subtypes remain in source coverage; they are not relabeled or treated as invalid records.",
            "- Notes retain their original text. Unicode, case, and whitespace normalization are used only for duplicate comparisons.",
            "- The source inventory covers all records. Sampling balance tables cover the planned country/class cells.",
            "",
        ]
        (stage / "quality_report.md").write_text("\n".join(lines), encoding="utf-8")
        manifest = {
            "format_version": 2,
            "status": report["status"],
            "config": config,
            "label_map": LABELS,
            "actor_groups": ACTOR_GROUPS,
            "preprocessor_sha256": file_digest(Path(__file__)),
            "python_version": sys.version.split()[0],
            "inputs": inventory,
            "sample_counts": report["sample_counts"],
            "sampling": sampling,
            "similarity_audit": report["similarity_audit"],
            "matching_targets": targets,
            "method": {
                "notes": "Unmodified raw notes; metadata whitespace trimmed/collapsed.",
                "exact_duplicates": "NFKC + casefold + whitespace collapse for comparisons; lowest event ID retained. Conflicting IDs or note labels excluded.",
                "related_events": "Initial country/admin1 date-window text families span locations, with configured word-set overlap retrieval followed by full-text similarity. Source links form a spanning forest. Independent selected-country audit has no token/admin1/class blocking; conflicts link families and trigger resampling (up to 10 passes).",
                "sampling": "Configured shared subtypes; one frozen representative per family. Equal word-length-band quotas per class/country weighted by minimum remaining group capacity after pilot allocation. Shared within-band targets are medians of country quantiles; ordered dynamic programming minimizes total absolute word-count distance. Ties use seeded SHA256 ordering. Pilot/reserves use proportional month strata.",
                "roles": "Pilot first, then main, then reserves; all selected groups disjoint.",
                "source_record": "1-based CSV data-record ordinal, excluding header; not a physical line number.",
            },
            "outputs": {
                p.name: {"sha256": file_digest(p), "bytes": p.stat().st_size}
                for p in sorted(stage.iterdir())
            },
        }
        write_json(stage / "manifest.json", manifest)
        stage.rename(output_dir)
    print(
        f"Wrote {output_dir}: main={len(selected['main'])}, pilot={len(selected['pilot'])}, reserve={len(selected['reserve'])}; {report['status']}",
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON override for the built-in study settings",
    )
    parser.add_argument("--input-dir", type=Path, default=ROOT / "conflict_data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    args = parser.parse_args()
    try:
        report = run(args.input_dir, args.output_dir, load_config(args.config))
    except (ValueError, OSError, csv.Error, KeyError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")
    if report["main_quota_shortfall"]:
        return 2
    return 3 if report["status"] == "validation_failed" else 0


if __name__ == "__main__":
    sys.exit(main())
