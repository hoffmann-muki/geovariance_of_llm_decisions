"""Part 2 invariants: span resolution, token matching, splits, and model parity.

  cd interpretability && python -m pytest geoflip/tests/test_part2_invariants.py

Tokenizer tests run anywhere the tokenizer is cached; model tests need CUDA.
"""

import csv
import os
from pathlib import Path

import pytest

from country_effect.part2_attribution import common as C
from geoflip import scoring
from geoflip.spans import check_matched, resolve
from geoflip.utils import load_secret, read_json

SPEC = scoring.MODELS[C.MODEL_KEY]


@pytest.fixture(scope="module")
def tokenizer():
    import transformers

    if not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = load_secret("HF_TOKEN", scoring.ENV_FILE)
    return transformers.AutoTokenizer.from_pretrained(
        SPEC["hf_id"], revision=SPEC["revision"]
    )


@pytest.fixture(scope="module")
def inputs():
    return C.part1()


def spans_for(tokenizer, d, names, display, task, n=None):
    return resolve(
        tokenizer,
        SPEC,
        task["system_prompt"],
        d["template"],
        names["slot"],
        display,
        d["evidence_sentence"],
        C.ev_group(d["boundary"], d["item"], d["wording"]),
        n,
    )


# ---------------------------------------------------------------- splits


def test_study_rebuild_matches_part1(inputs):
    descriptions, _, _ = inputs
    rebuilt = C.build_descriptions("PR", "study") + C.build_descriptions("BV", "study")
    by_id = {r["description_id"]: r for r in rebuilt}
    assert len(by_id) == len(descriptions) == 3520
    for d in descriptions:
        r = by_id[d["description_id"]]
        for key, value in d.items():
            assert str(r[key]) == value, (d["description_id"], key)


def test_splits_use_disjoint_values():
    for b in ("PR", "BV"):
        fields = ("actors", "grievance", "place") if b == "PR" else ("place",)
        values = {
            split: {f: {c[f] for c in C.contexts(b, split)} for f in fields}
            for split in ("pilot", "study", "heldout")
        }
        for f in fields:
            assert not values["pilot"][f] & values["study"][f]
            assert not values["heldout"][f] & values["study"][f]
            assert not values["heldout"][f] & values["pilot"][f]
            assert len(values["heldout"][f]) == 8 and len(values["pilot"][f]) == 8


# ---------------------------------------------------------------- spans


def test_answer_letters_single_token(tokenizer, inputs):
    descriptions, names, tasks = inputs
    d = descriptions[0]
    for order in "AB":
        text, ids = scoring.render_prompt(
            tokenizer,
            SPEC,
            tasks[f"{d['boundary']}:{order}"]["system_prompt"],
            d["text_none"],
        )
        for letter in "AB":
            assert len(scoring.answer_token_ids(tokenizer, text, ids, letter)) == 1


def test_ev_group_defined_and_resolves_for_every_item(tokenizer, inputs):
    """Every (item, wording) of BV, and every PR panel item, on the pilot split."""
    _, names, tasks = inputs
    cfg = C.panel()
    for b in ("BV", "PR"):
        items = cfg["boundaries"][b]["ev_group"]
        rows = [r for r in C.build_descriptions(b, "pilot") if r["item"] in items]
        if b == "BV":
            assert {r["item"] for r in rows} == {
                r["item"] for r in C.build_descriptions(b, "pilot")
            }
        for r in rows:
            if r["context_id"].endswith("p0"):
                for order in "AB":
                    p = spans_for(tokenizer, r, names, None, tasks[f"{b}:{order}"])
                    assert p.spans["name"] is None
                    phrase = C.ev_group(b, r["item"], r["wording"])
                    got = tokenizer.decode([p.ids[i] for i in p.spans["ev_group"]])
                    assert got.strip() == phrase
                    assert tokenizer.decode([p.ids[p.spans["ev_end"][0]]]) == "."


def test_name_spans_all_names_panel(tokenizer, inputs):
    """Every Part 1 name in both orders and both frames: token counts, round trip,
    coverage of every position the name can affect."""
    descriptions, names, tasks = inputs
    for b in ("BV", "PR"):
        sample = [
            d for d in C.panel_descriptions(b) if d["context_id"].endswith("c00")
        ][:4]
        for d in sample:
            for order in "AB":
                task = tasks[f"{b}:{order}"]
                none = spans_for(tokenizer, d, names, None, task)
                for display in list(names["real"].values()) + list(names["fictional"]):
                    p = spans_for(
                        tokenizer, d, names, display, task, names["tokens"][display]
                    )
                    assert (
                        tokenizer.decode([p.ids[i] for i in p.spans["name"]])
                        == " " + display
                    )
                    covered = sorted(
                        i
                        for k in ("name", "ev_group", "ev_rest", "bridge", "readout")
                        for i in p.spans[k]
                    )
                    assert covered == list(range(p.spans["name"][0], len(p.ids)))
                    slot = p.spans["name"][0] - 1  # the " in" token of the slot phrase
                    assert tokenizer.decode([p.ids[slot]]) == " in"
                    assert p.ids[:slot] == none.ids[:slot]
                    assert p.readout == len(p.ids) - 1


def test_frames_shift_indices(tokenizer, inputs):
    """Frames place the country differently, so name indices must differ; the
    two option orders happen to have equal system-prompt lengths, and their
    spans are resolved independently all the same."""
    _, names, tasks = inputs
    rows = {
        d["frame"]: d for d in C.panel_descriptions("BV") if d["item"] == "anchor_lo"
    }
    a = spans_for(tokenizer, rows["f1"], names, "Kenya", tasks["BV:A"], 1)
    b = spans_for(tokenizer, rows["f2"], names, "Kenya", tasks["BV:A"], 1)
    assert a.spans["name"] != b.spans["name"]
    for order in "AB":
        p = spans_for(tokenizer, rows["f1"], names, "Kenya", tasks[f"BV:{order}"], 1)
        assert p.readout == len(p.ids) - 1


def test_matched_pairs(tokenizer, inputs):
    _, names, tasks = inputs
    d = C.panel_descriptions("BV")[0]
    task = tasks["BV:A"]
    base = spans_for(tokenizer, d, names, "Atlantis", task, 1)
    target = spans_for(tokenizer, d, names, "Kenya", task, 1)
    check_matched(base, target)
    three = spans_for(tokenizer, d, names, "the United States", task, 3)
    with pytest.raises(ValueError):
        check_matched(base, three)
    region = spans_for(tokenizer, d, names, "the Central Region", task, 3)
    check_matched(region, three)


def test_controls_token_counts(tokenizer, inputs):
    path = C.OUTPUTS / "01_controls" / "controls.json"
    if not path.exists():
        pytest.skip("controls not built yet")
    _, names, tasks = inputs
    d = C.panel_descriptions("BV")[0]
    for name, info in read_json(path)["names"].items():
        spans_for(tokenizer, d, names, name, tasks["BV:A"], info["tokens"])


# ---------------------------------------------------------------- model


cuda = pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="needs CUDA"
)


@pytest.fixture(scope="module")
def runner():
    return C.load_runner()[0]


@cuda
def test_hooks_transparent_and_identity_patch(runner, inputs):
    import torch

    from geoflip.hooks import span_patch

    descriptions, names, tasks = inputs
    d = C.panel_descriptions("BV")[0]
    p = C.resolve_prompt(runner, d, names, "Kenya", tasks["BV:A"], 1)
    plain, _ = runner.run(p.ids)
    captured, states = runner.run(p.ids, capture=True)
    assert torch.equal(plain, captured)
    for layer in (0, 13, 27):
        patched, _ = runner.run(
            p.ids, patch=span_patch(states, layer, p.spans["readout"])
        )
        assert torch.equal(plain, patched)
    ref = scoring.score_first_token(
        runner.model,
        runner.tokenizer,
        runner.spec,
        tasks["BV:A"]["system_prompt"],
        C.text_for(d, names, "Kenya"),
        ("A", "B"),
        runner.device,
    )
    ids = runner.letter_ids(p.text, p.ids)
    assert float(plain[ids["A"]]) == ref["lp"]["A"]


@cuda
def test_inputs_embeds_parity(runner, inputs):
    import torch

    descriptions, names, tasks = inputs
    d = C.panel_descriptions("BV")[0]
    p = C.resolve_prompt(runner, d, names, "Kenya", tasks["BV:B"], 1)
    by_ids, _ = runner.run(p.ids)
    with torch.inference_mode():
        embeds = runner.model.model.embed_tokens(
            torch.tensor([p.ids], device=runner.device)
        )
    by_embeds, _ = runner.run(p.ids, embeds=embeds)
    assert torch.equal(by_ids, by_embeds)
