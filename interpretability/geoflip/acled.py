"""How human coders (ACLED) code events that mention a given act.

For each boundary, the notes of two groups of ACLED sub-event types are
compared:

  PR  higher: Violent demonstration
      lower:  Peaceful protest, Protest with intervention
  BV  higher: Armed clash
      lower:  Attack

For every act pattern in a fixed lexicon (ACT_LEXICON, versioned below), the
module counts the notes that mention it in each group and reports
P(higher | act) = n_higher / (n_higher + n_lower). An act whose share is near 0
or 1 is coded consistently; a share in between means coders put events with
that act on both sides of the boundary. Mob violence is left out of PR because
it is not a demonstration.

Only counts are written; notes are never copied (ACLED terms of use).

  python -m geoflip.acled --out inputs/v4/acled_acts.csv
"""

import argparse
import re
from collections import Counter
from pathlib import Path

from .utils import read_csv, write_csv

ROOT = Path(__file__).resolve().parent.parent
CONFLICT_DATA = ROOT.parent / "conflict_data"
LEXICON_VERSION = "1"

GROUPS = {
    "PR": {
        "higher": ("Violent demonstration",),
        "lower": ("Peaceful protest", "Protest with intervention"),
    },
    "BV": {"higher": ("Armed clash",), "lower": ("Attack",)},
}

THROW = r"(?:threw|throw|throwing|hurled|hurling|pelted|pelting|tossed|lobbed)"
BURN = r"(?:burn(?:ed|t|ing)?|set (?:on )?fire(?: to)?|set ablaze|set alight|torch(?:ed|ing))"
NEAR = r"\W+(?:\w+\W+){0,4}?"  # up to 4 words between verb and object
AGENT = r"\b(?:protesters|demonstrators|rioters|participants|students|workers|residents|supporters|members|villagers|activists)"


def _near(verb, obj):
    return rf"\b{verb}{NEAR}{obj}\b"


# Patterns that do not name a concrete act (they are counted but never become items).
GENERIC = {"set_fire_generic", "damage"}
MIN_ATTESTATIONS = 20  # an act becomes an item if n_higher + n_lower >= this

# act id -> (dimension levels, regex). Dimensions follow the class definitions
# (MATERIALS.md § 3.2): PR act type x target; BV means x use x organisation.
ACT_LEXICON = {
    "PR": {
        # projectiles
        "throw_stones": (
            "projectile",
            "unspecified",
            _near(THROW, r"(?:stones?|rocks?|bricks?)"),
        ),
        "throw_eggs": ("projectile", "unspecified", _near(THROW, r"eggs?")),
        "throw_tomatoes": ("projectile", "unspecified", _near(THROW, r"tomato(?:es)?")),
        "throw_shoes": (
            "projectile",
            "unspecified",
            _near(THROW, r"(?:shoes?|slippers?|sandals?)"),
        ),
        "throw_bottles": ("projectile", "unspecified", _near(THROW, r"bottles?")),
        "throw_water": ("projectile", "unspecified", _near(THROW, r"water")),
        "throw_paint_ink": (
            "projectile",
            "unspecified",
            _near(THROW, r"(?:paint|ink)"),
        ),
        "throw_fireworks": (
            "projectile",
            "unspecified",
            _near(THROW, r"(?:fireworks?|firecrackers?|crackers?)"),
        ),
        "throw_objects": ("projectile", "unspecified", _near(THROW, r"objects?")),
        "throw_petrol_bomb": (
            "projectile",
            "unspecified",
            _near(THROW, r"(?:petrol bombs?|molotov\w*)"),
        ),
        # fire
        "burn_tires": ("fire", "property", _near(BURN, r"(?:tires|tyres)")),
        "burn_effigy": ("fire", "symbolic", _near(BURN, r"effig(?:y|ies)")),
        "burn_flag": ("fire", "symbolic", _near(BURN, r"flags?")),
        "burn_documents": (
            "fire",
            "symbolic",
            _near(BURN, r"(?:copies|documents?|posters?|banners?|books?)"),
        ),
        "burn_vehicle": (
            "fire",
            "property",
            _near(BURN, r"(?:vehicles?|cars?|bus(?:es)?|trucks?|motorcycles?)"),
        ),
        "burn_garbage": (
            "fire",
            "property",
            _near(BURN, r"(?:garbage|trash|rubbish|waste|bins?)"),
        ),
        "set_fire_generic": ("fire", "property", rf"\b{BURN}\b"),
        # obstruction
        "block_road": (
            "obstruction",
            "none",
            r"\bblock(?:ed|ing|ade[sd]?)\W+(?:\w+\W+){0,3}?(?:roads?|highways?|streets?|traffic)\b",
        ),
        "barricade": ("obstruction", "none", r"\bbarricad\w*"),
        "block_entrance": (
            "obstruction",
            "none",
            r"\bblock(?:ed|ing)\W+(?:\w+\W+){0,3}?(?:entrances?|gates?|doors?)\b",
        ),
        "lock_in": (
            "obstruction",
            "person",
            r"\b(?:locked|gheraoed|gherao|confined)\b",
        ),
        # property contact
        "vandalise": ("property", "property", r"\bvandali[sz]\w*"),
        "break_windows": (
            "property",
            "property",
            r"\b(?:broke|smashed|shattered|breaking|smashing)\W+(?:\w+\W+){0,3}?(?:windows?|glass|windshields?)\b",
        ),
        "damage": ("property", "property", r"\bdamag(?:ed|ing)\b"),
        "tear_down": (
            "property",
            "symbolic",
            r"\b(?:tore|torn|pulled|ripped)\s+(?:down|off)\b",
        ),
        "deface_paint": (
            "property",
            "symbolic",
            r"\b(?:graffiti|spray[- ]?paint\w*|painted slogans|blackened|smeared)\b",
        ),
        "break_gate": (
            "property",
            "property",
            r"\b(?:broke|smashed|forced|breaking)\W+(?:\w+\W+){0,3}?(?:gates?|doors?|barriers?|locks?)\b",
        ),
        "push_barrier": (
            "property",
            "property",
            r"\b(?:pushed|toppled|knocked down|pulled down|removed)\W+(?:\w+\W+){0,3}?(?:barriers?|barricades?|fences?)\b",
        ),
        # entering
        "set_off_fireworks": (
            "fire",
            "none",
            r"\b(?:set off|lit|launched|burst)\W+(?:\w+\W+){0,2}?(?:fireworks?|firecrackers?|crackers?|flares?|smoke bombs?)\b",
        ),
        "burn_photos": (
            "fire",
            "symbolic",
            _near(BURN, r"(?:photos?|photographs?|pictures?|portraits?|images?)"),
        ),
        "block_rail": (
            "obstruction",
            "none",
            r"\bblock(?:ed|ing)\W+(?:\w+\W+){0,3}?(?:rail\w*|train\w*|tracks?)\b",
        ),
        "occupy": ("entry", "property", r"\boccup(?:ied|y|ying)\b"),
        "storm": (
            "entry",
            "property",
            r"\b(?:storm(?:ed|ing)|forced (?:their )?way|barged)\b",
        ),
        # person contact
        "scuffle": (
            "person",
            "person",
            r"\b(?:scuffl\w*|jostl\w*|push(?:ed|ing)?\s+and\s+shov\w*|manhandl\w*)\b",
        ),
        "assault": (
            "person",
            "person",
            AGENT
            + r"\W+(?:\w+\W+){0,3}?(?:assault\w*|beat|thrash\w*|slapp\w*|attack\w*)\b",
        ),
        "chant_placards": (
            "none",
            "none",
            r"\b(?:chant\w*|placards?|banners?|slogans?)\b",
        ),
    },
    "BV": {
        # what the other side has and does (first-party armed group is given)
        "exchanged_fire": (
            "firearm",
            "used_at",
            "unspecified",
            r"\b(?:exchanged (?:fire|gunfire|shots)|exchange of fire|gunfight|shootout|fired at each other)\b",
        ),
        "returned_fire": (
            "firearm",
            "used_at",
            "unspecified",
            r"\b(?:returned fire|fired back|retaliat\w*)\b",
        ),
        "self_defense_group": (
            "unspecified",
            "unspecified",
            "standing",
            r"\b(?:self[- ]defen[cs]e (?:group|force|committee)s?|village defen[cs]e|community guards?|ronda\w*|autodefensa\w*)\b",
        ),
        "vigilante": ("unspecified", "unspecified", "informal", r"\bvigilante\w*\b"),
        "security_guard": (
            "unspecified",
            "unspecified",
            "standing",
            r"\b(?:security guards?|watchm[ae]n|night guards?|private guards?)\b",
        ),
        "militia": ("unspecified", "unspecified", "standing", r"\bmilitia\w*\b"),
        "villagers": ("none", "none", "individuals", r"\bvillagers\b"),
        "farmers": (
            "none",
            "none",
            "individuals",
            r"\b(?:farmers?|herders?|shepherds?)\b",
        ),
        "hunters": ("firearm", "unspecified", "informal", r"\bhunters?\b"),
        "drivers": ("none", "none", "individuals", r"\b(?:drivers?|truckers?)\b"),
        "unarmed": ("none", "none", "unspecified", r"\bunarmed\b"),
        "armed_residents": (
            "firearm",
            "unspecified",
            "individuals",
            r"\barmed (?:civilians|residents|villagers|locals)\b",
        ),
        "stones": ("improvised", "used_at", "unspecified", r"\b(?:stones?|rocks?)\b"),
        "sticks_machetes": (
            "improvised",
            "unspecified",
            "unspecified",
            r"\b(?:sticks?|machetes?|knives|knife|clubs?|batons?|rods?)\b",
        ),
        "pistol": (
            "firearm",
            "unspecified",
            "unspecified",
            r"\b(?:pistols?|handguns?|revolvers?)\b",
        ),
        "rifle": (
            "firearm",
            "unspecified",
            "unspecified",
            r"\b(?:rifles?|ak-47s?|assault rifles?)\b",
        ),
        "shotgun": ("firearm", "unspecified", "unspecified", r"\bshotguns?\b"),
        "warning_shot": (
            "firearm",
            "warning",
            "unspecified",
            r"\b(?:warning shots?|fired (?:in|into) the air|aerial (?:fire|firing|shots))\b",
        ),
        "miners": ("none", "none", "individuals", r"\bminers?\b"),
        "traders": (
            "none",
            "none",
            "individuals",
            r"\b(?:traders?|vendors?|merchants?|shopkeepers?)\b",
        ),
        "students": ("none", "none", "individuals", r"\bstudents?\b"),
        "fishermen": ("none", "none", "individuals", r"\bfisherm[ae]n\b"),
        "worshippers": (
            "none",
            "none",
            "individuals",
            r"\b(?:worshipp?ers?|pilgrims?)\b",
        ),
        "bodyguards": (
            "firearm",
            "unspecified",
            "standing",
            r"\b(?:bodyguards?|escorts?|gunmen guarding)\b",
        ),
        "bows_spears": (
            "improvised",
            "unspecified",
            "unspecified",
            r"\b(?:bows?|arrows?|spears?|axes)\b",
        ),
        "grenade": ("explosive", "used_at", "unspecified", r"\bgrenades?\b"),
        "fled": (
            "unspecified",
            "none",
            "unspecified",
            r"\b(?:fled|escaped|ran away)\b",
        ),
        "former_members": (
            "unspecified",
            "none",
            "unspecified",
            r"\b(?:former|ex-)\s*(?:members?|fighters?|combatants?|militants?|rebels?)\b",
        ),
    },
}


def load_notes(directory=CONFLICT_DATA):
    rows = []
    for path in sorted(directory.glob("*_conflict_data.csv")):
        rows += read_csv(path)
    return rows


def act_table(rows):
    out = []
    for boundary, groups in GROUPS.items():
        side = {}
        for name, types in groups.items():
            side[name] = [
                r["notes"].lower() for r in rows if r["sub_event_type"] in types
            ]
        for act, spec in ACT_LEXICON[boundary].items():
            pattern = re.compile(spec[-1])
            counts = Counter(
                {
                    name: sum(1 for n in notes if pattern.search(n))
                    for name, notes in side.items()
                }
            )
            total = counts["higher"] + counts["lower"]
            out.append(
                {
                    "boundary": boundary,
                    "act": act,
                    "dimensions": "/".join(spec[:-1]),
                    "n_higher": counts["higher"],
                    "n_lower": counts["lower"],
                    "share_higher": round(counts["higher"] / total, 4) if total else "",
                    "notes_higher": len(side["higher"]),
                    "notes_lower": len(side["lower"]),
                    "pattern": spec[-1],
                    "lexicon_version": LEXICON_VERSION,
                }
            )
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=CONFLICT_DATA)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    table = act_table(load_notes(args.data))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, list(table[0]), table)
    for row in table:
        print(
            f"{row['boundary']} {row['act']:20} hi {row['n_higher']:5} lo {row['n_lower']:6} P(hi|act) {row['share_higher']}"
        )


if __name__ == "__main__":
    main()
