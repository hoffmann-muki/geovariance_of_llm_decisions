"""The v4 evidence items: one sentence per ACLED-attested act (PLAN.md § 3).

Inclusion rule (fixed before any model scoring): every act in the ACLED lexicon
(geoflip/acled.py) that names a concrete act (not in acled.GENERIC) and is
mentioned in at least acled.MIN_ATTESTATIONS notes across both sides of the
boundary becomes one item. Acts used for the anchors are anchors, not items.

Wording rule. W1 is the canonical sentence. W2 applies the fixed
transformations in W2_RULES: the subject swap for the boundary, and at most
one substitution from the verb table. Where a sentence would echo a class
definition, a neutral stand-in is used and noted in STAND_INS.

  python -m geoflip.items --acts inputs/v4/acled_acts.csv --out inputs/v4/items.json
"""

import argparse
import json
from pathlib import Path

from .acled import GENERIC, MIN_ATTESTATIONS
from .utils import read_csv, require

W2_RULES = {
    "PR": {
        "subject": ["Some participants", "A few demonstrators"],
        "anchor_subject": ["Participants", "The demonstrators"],
    },
    "BV": {
        "subject": ["The other group", "Members of the other group"],
        "one_member": ["One member of the other group", "A member of the other group"],
    },
    "verbs": {
        "threw": "hurled",
        "burned": "set fire to",
        "set off": "lit",
        "blocked": "obstructed",
        "built": "put up",
        "locked": "shut",
        "vandalised": "defaced",
        "smashed": "shattered",
        "tore down": "pulled down",
        "spray-painted": "painted",
        "pried open": "levered open",
        "pushed over": "knocked over",
        "occupied": "took over",
        "stormed": "rushed into",
        "scuffled with": "jostled with",
        "beat": "hit",
        "carried": "had",
        "fired back at": "returned fire at",
        "fled": "ran away",
        "chanted slogans and held": "chanted slogans and carried",
        "exchanged fire": "traded gunfire",
        "was": "belonged to",
    },
}

STAND_INS = {
    "self_defense_group": "'self-defense' shares its stem with the definitions' 'defend'; written as a village night patrol ('committee' shares its stem with 'commit')",
    "break_gate": "'forced open' shares its stem with the definitions' 'force'; written as 'pried open'",
}

# act -> W1 predicate (after the subject). Wordings are built by build_sentence.
PR_ITEMS = {
    "throw_stones": "threw stones at a nearby building.",
    "throw_eggs": "threw eggs at a nearby building.",
    "throw_bottles": "threw plastic bottles at a nearby building.",
    "throw_water": "threw water at a group of officials.",
    "throw_paint_ink": "threw paint at a nearby building.",
    "throw_fireworks": "threw firecrackers at a nearby building.",
    "throw_objects": "threw objects at a nearby building.",
    "throw_petrol_bomb": "threw a petrol bomb at a nearby building.",
    "burn_tires": "burned tires in the road.",
    "burn_effigy": "burned an effigy in the street.",
    "burn_flag": "burned a flag in the street.",
    "burn_documents": "burned copies of a new regulation.",
    "burn_vehicle": "burned a parked car.",
    "block_road": "blocked the main road.",
    "barricade": "built barricades across the road.",
    "block_entrance": "blocked the entrance of a nearby building.",
    "lock_in": "locked officials inside a nearby building.",
    "vandalise": "vandalised a nearby building.",
    "break_windows": "smashed the windows of a nearby building.",
    "tear_down": "tore down a sign on a nearby building.",
    "deface_paint": "spray-painted slogans on a nearby building.",
    "break_gate": "pried open the gate of a nearby building.",
    "push_barrier": "pushed over a metal barrier.",
    "set_off_fireworks": "set off fireworks in the street.",
    "block_rail": "blocked the railway tracks.",
    "occupy": "occupied a nearby building.",
    "storm": "stormed a nearby building.",
    "scuffle": "scuffled with security staff.",
    "assault": "beat a man who was filming the crowd.",
}
PR_ANCHORS = {
    "anchor_lo": (
        "chant_placards",
        "chanted slogans and held placards the whole time.",
    ),
    "anchor_hi": (
        ["throw_stones", "burn_vehicle"],
        "threw stones and burned parked cars.",
    ),
}

# (subject key, predicate)
BV_ITEMS = {
    "returned_fire": ("one_member", "fired back at the armed group."),
    "self_defense_group": ("subject", "was a village night patrol."),
    "security_guard": ("subject", "included a night watchman."),
    "militia": ("subject", "was a local militia."),
    "villagers": ("subject", "were villagers from a nearby settlement."),
    "farmers": ("subject", "were farmers working in nearby fields."),
    "drivers": ("subject", "were truck drivers waiting by the road."),
    "unarmed": ("subject", "carried no weapons."),
    "armed_residents": ("subject", "were local residents who carried guns."),
    "stones": ("subject", "threw stones at the armed group."),
    "sticks_machetes": ("subject", "carried sticks and machetes."),
    "pistol": ("one_member", "carried a pistol."),
    "rifle": ("one_member", "carried a rifle."),
    "shotgun": ("one_member", "carried a shotgun."),
    "miners": ("subject", "were miners leaving a nearby mine."),
    "traders": ("subject", "were traders from a nearby market."),
    "students": ("subject", "were students walking home."),
    "fishermen": ("subject", "were fishermen returning from the river."),
    "bodyguards": ("subject", "were bodyguards escorting a local businessman."),
    "grenade": ("one_member", "threw a grenade at the armed group."),
    "fled": ("subject", "fled when the shooting started."),
    "former_members": ("subject", "were former members of a rival militia."),
}
BV_ANCHORS = {
    "anchor_lo": (["farmers", "unarmed"], "were farmers who carried no weapons."),
    "anchor_hi": ("exchanged_fire", "The two groups exchanged fire for an hour."),
}


def swap_verb(predicate):
    """Apply the first matching verb substitution (longest key first)."""
    for old in sorted(W2_RULES["verbs"], key=len, reverse=True):
        if predicate.startswith(old + " ") or predicate.startswith(old + "."):
            return W2_RULES["verbs"][old] + predicate[len(old) :]
    return predicate


def build_sentence(boundary, subject_key, predicate):
    if predicate[0].isupper():  # a full sentence (BV anchor_hi)
        return [predicate, predicate.replace("exchanged fire", "traded gunfire")]
    one, two = W2_RULES[boundary][subject_key]
    w2 = swap_verb(predicate)
    if boundary == "BV" and w2 == predicate:
        w2 = (
            predicate  # BV sentences without a verb substitution differ by subject only
        )
    return [f"{one} {predicate}", f"{two} {w2}"]


def provenance(table, boundary, acts):
    acts = acts if isinstance(acts, list) else [acts]
    rows = [table[(boundary, act)] for act in acts]
    return {
        "acled_acts": acts,
        "dimensions": [row["dimensions"] for row in rows],
        "n_higher": [int(row["n_higher"]) for row in rows],
        "n_lower": [int(row["n_lower"]) for row in rows],
        "share_higher": [float(row["share_higher"]) for row in rows],
    }


def build(acts_csv):
    table = {(row["boundary"], row["act"]): row for row in read_csv(acts_csv)}
    out = {}
    for boundary, items, anchors in (
        ("PR", PR_ITEMS, PR_ANCHORS),
        ("BV", BV_ITEMS, BV_ANCHORS),
    ):
        anchor_acts = {
            acts for acts, _ in anchors.values() if not isinstance(acts, list)
        }
        eligible = {
            act
            for (b, act), row in table.items()
            if b == boundary
            and act not in GENERIC
            and act not in anchor_acts
            and int(row["n_higher"]) + int(row["n_lower"]) >= MIN_ATTESTATIONS
        }
        require(
            set(items) == eligible,
            f"{boundary}: items {sorted(set(items) ^ eligible)} differ from the inclusion rule",
        )
        evidence = {}
        for role, (acts, text) in anchors.items():
            key = "anchor_subject" if boundary == "PR" else "subject"
            evidence[role] = {
                "role": "anchor_lower" if role == "anchor_lo" else "anchor_higher",
                "wordings": build_sentence(boundary, key, text),
                **provenance(table, boundary, acts),
            }
        for act, spec in items.items():
            key, predicate = ("subject", spec) if boundary == "PR" else spec
            evidence[act] = {
                "role": "item",
                "wordings": build_sentence(boundary, key, predicate),
                **provenance(table, boundary, act),
                **({"stand_in": STAND_INS[act]} if act in STAND_INS else {}),
            }
        out[boundary] = evidence
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--acts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    items = build(args.acts)
    payload = {
        "rule": f"every concrete ACLED-lexicon act with >= {MIN_ATTESTATIONS} attestations; anchors from attested acts",
        "w2_rules": W2_RULES,
        "stand_ins": STAND_INS,
        "evidence": items,
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    for boundary, evidence in items.items():
        print(
            boundary,
            len([e for e in evidence.values() if e["role"] == "item"]),
            "items",
        )
        for act, e in evidence.items():
            print(
                f"  {act:18} {e['share_higher']} | {e['wordings'][0]} || {e['wordings'][1]}"
            )


if __name__ == "__main__":
    main()
