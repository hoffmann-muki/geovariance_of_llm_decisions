"""Resolve the token positions of the name, evidence and read-out in a rendered prompt.

Positions come from character offsets on the full rendered chat string, never
from hardcoded indices: the option order changes the system-prompt length,
frames place the country differently, and Llama BPE tokens carry their leading
space (" Kenya" is not "Kenya"). Every span is checked by decoding it back to
the exact substring it covers.

Spans (lists of token positions; contiguous unless noted):
  name      " " + rendered name (None for the no-country prompt)
  ev_group  the target-group phrase inside the evidence sentence
  ev_rest   the rest of the evidence sentence (may be split around ev_group)
  ev_end    the last evidence token (its period), part of ev_rest
  bridge    every other position after the name slot, before the read-out
  readout   the final prompt token ("\\n\\n" after the assistant header)
"""

from dataclasses import dataclass, field

from .scoring import render_prompt
from .utils import require

READOUT_TEXT = "\n\n"
ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>"


@dataclass
class PromptSpans:
    text: str
    ids: list
    spans: dict
    pieces: dict = field(default_factory=dict)  # span -> [(start, end)] runs

    @property
    def readout(self):
        return self.spans["readout"][0]


def _runs(positions):
    runs, start = [], None
    for i, p in enumerate(positions):
        if start is None:
            start = p
        if i + 1 == len(positions) or positions[i + 1] != p + 1:
            runs.append((start, p + 1))
            start = None
    return runs


def _tokens_covering(offsets, char_start, char_end, label):
    """Token positions whose characters exactly tile [char_start, char_end)."""
    inside = [i for i, (a, b) in enumerate(offsets) if a < char_end and b > char_start]
    require(inside, f"{label}: no tokens cover characters {char_start}-{char_end}")
    require(
        offsets[inside[0]][0] == char_start and offsets[inside[-1]][1] == char_end,
        f"{label}: token boundaries do not align with characters "
        f"{char_start}-{char_end} (got {offsets[inside[0]][0]}-{offsets[inside[-1]][1]})",
    )
    return inside


def _with_space(text, start):
    """Include the preceding space, which Llama BPE attaches to the next word."""
    return start - 1 if start > 0 and text[start - 1] == " " else start


def resolve(
    tokenizer,
    spec,
    system_prompt,
    template,
    slot,
    display,
    evidence,
    ev_group,
    expected_name_tokens=None,
):
    """Render one prompt and resolve its spans.

    template: description text with "{COUNTRY}"; slot: e.g. " in {country}";
    display: rendered name or None; evidence: the evidence sentence, which ends
    the description; ev_group: the target-group phrase, unique in evidence."""
    phrase = "" if display is None else slot.format(country=display)
    user_text = template.replace("{COUNTRY}", phrase)
    text, ids = render_prompt(tokenizer, spec, system_prompt, user_text)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    require(
        list(encoded["input_ids"]) == ids,
        "Offset tokenization differs from geoflip.scoring token ids",
    )
    offsets = [tuple(o) for o in encoded["offset_mapping"]]
    user_start = text.index(user_text)
    require(text.count(user_text) == 1, "User text occurs more than once")
    slot_char = user_start + template.index("{COUNTRY}")
    spans, expected = {}, {}

    if display is None:
        spans["name"] = None
        after_name = slot_char
    else:
        name_start = slot_char + phrase.index(display)
        name_start = _with_space(text, name_start)
        name_end = slot_char + phrase.index(display) + len(display)
        spans["name"] = _tokens_covering(offsets, name_start, name_end, "name")
        expected["name"] = text[name_start:name_end]
        require(
            expected["name"] == " " + display, "Name span must start with one space"
        )
        if expected_name_tokens is not None:
            require(
                len(spans["name"]) == expected_name_tokens,
                f"{display!r}: {len(spans['name'])} tokens, expected {expected_name_tokens}",
            )
        after_name = name_end

    require(user_text.endswith(evidence), "Evidence must end the description")
    ev_start = user_start + len(user_text) - len(evidence)
    ev_end = ev_start + len(evidence)
    require(evidence.count(ev_group) == 1, f"{ev_group!r} not unique in {evidence!r}")
    group_start = ev_start + evidence.index(ev_group)
    group_end = group_start + len(ev_group)
    ev_all = _tokens_covering(offsets, _with_space(text, ev_start), ev_end, "evidence")
    group = _tokens_covering(
        offsets, _with_space(text, group_start), group_end, "ev_group"
    )
    require(set(group) <= set(ev_all), "ev_group lies outside the evidence")
    spans["ev_group"] = group
    spans["ev_rest"] = [p for p in ev_all if p not in set(group)]
    spans["ev_end"] = [ev_all[-1]]
    expected["ev_group"] = text[_with_space(text, group_start) : group_end]

    readout = len(ids) - 1
    require(
        tokenizer.decode(ids[readout:]) == READOUT_TEXT,
        "Read-out token is not '\\n\\n'",
    )
    require(
        text.endswith(ASSISTANT_HEADER + READOUT_TEXT),
        "Read-out does not follow the assistant header",
    )
    spans["readout"] = [readout]
    first_after = next(i for i, (a, _) in enumerate(offsets) if a >= after_name)
    taken = set(ev_all) | {readout}
    spans["bridge"] = [p for p in range(first_after, readout) if p not in taken]

    pieces = {k: _runs(v) for k, v in spans.items() if v}
    # Round trip: every contiguous run decodes to the characters it covers.
    for key, runs in pieces.items():
        for a, b in runs:
            chunk = text[offsets[a][0] : offsets[b - 1][1]]
            require(
                tokenizer.decode(ids[a:b]) == chunk,
                f"{key}: tokens {a}-{b} decode to {tokenizer.decode(ids[a:b])!r}, "
                f"not {chunk!r}",
            )
    for key, value in expected.items():
        decoded = tokenizer.decode([ids[p] for p in spans[key]])
        require(decoded == value, f"{key}: decodes to {decoded!r}, not {value!r}")
    names = [k for k in spans if spans[k] and k != "ev_end"]
    flat = [p for k in names for p in spans[k]]
    require(len(flat) == len(set(flat)), "Spans overlap")
    return PromptSpans(text=text, ids=ids, spans=spans, pieces=pieces)


def check_matched(base, target):
    """Patching pairs: equal length, identical tokens outside the name span."""
    require(len(base.ids) == len(target.ids), "Patch pair differs in length")
    require(base.spans["name"] == target.spans["name"], "Name spans are not aligned")
    name = set(base.spans["name"] or [])
    for key in ("ev_group", "ev_rest", "bridge", "readout"):
        require(base.spans[key] == target.spans[key], f"{key} spans not aligned")
    diff = [i for i, (a, b) in enumerate(zip(base.ids, target.ids)) if a != b]
    require(set(diff) <= name, f"Pair differs outside the name span at {diff}")
