"""Render an event description from a frame, a context, an evidence sentence and a country.

The country enters only through the slot phrase (for example " in {country}"),
so a description with a country differs from the no-country description by
that phrase alone.
"""

import string

NONE = "none"


def country_conditions(countries, kinds=("none", "real", "fictional")):
    """(country id, display name or None, kind) for the requested kinds."""
    out = [(NONE, None, "none")] if "none" in kinds else []
    for kind in ("real", "fictional"):
        if kind in kinds:
            out += [(cid, name, kind) for cid, name in countries[kind].items()]
    return out


def country_phrase(countries, display):
    return "" if display is None else countries["slot"].format(country=display)


def render(frame, base, evidence, phrase):
    fields = {key: value for key, value in base.items() if key != "base_id"}
    return frame.format(**fields, country=phrase, evidence=evidence)


def frame_fields(frame):
    return [field for _, field, _, _ in string.Formatter().parse(frame) if field]
