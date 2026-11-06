"""Build and verify event descriptions whose coded label changes with the country.

Implements PLAN.md § 4 (the verifier loop) for one model:

  check      V0 static input checks, V1 scorer reproduces reference scores, V2 task prompts
  ambiguity  V3 every item without a country (explore bases)
  screen     V4 ambiguous items x every country (screen bases, wording 1)
  confirm    V5 chosen countries on confirm bases: systematic pairs and flips
  test       V6 final pairs on the new test bases (allowed once per iteration)
  matrix     every country condition on every description with a verified flip
  transfer   re-score another run's matrix rows with a different model
  report     markdown report, flip list and matrices from the stage summaries
  probe      score draft evidence sentences (explore bases, no country)

Every prompt is scored with scoring.score_prompt, one prompt per forward pass
and no batching. Scores are cached in
<work>/cache/, keyed by model, system prompt, description and classes.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import statistics
import string
import sys
import time
from collections import defaultdict
from itertools import product
from pathlib import Path

from . import scoring as S
from .utils import file_digest, read_csv, read_json, read_jsonl, require, write_json

ROOT = Path(__file__).resolve().parent.parent
GEO = ROOT / "inputs"
RESULTS = ROOT / "results"
WORK = Path(os.environ.get("GEOFLIP_WORK", "/fs/clip-scratch/pramu/geoflip_work"))
PRODUCER = Path(__file__).resolve()

BOUNDARIES = ("PR", "BV")
NONE = "none"
ROLES = ("anchor_lower", "anchor_higher", "borderline")
INPUT_FILES = (
    "descriptions.json",
    "task_prompts.json",
    "countries.json",
    "splits.json",
)

# Study parameters (PLAN.md § Terms and § Verifier Loop).
MARGIN = 0.5  # |s| needed for a robust label
MIN_AGREE = 3  # paraphrases that must share the sign of s
AMBIGUITY_D = 1.5  # |median s| for an ambiguous item (p in 0.18-0.82)
ANCHOR_ACCURACY = 0.9
MIN_MASS = 0.9
REPRODUCE_TOLERANCE = 0.01  # V1: max |Δ log P| against inputs/scorer_reference.jsonl
TOP_COUNTRIES = 4
SCREEN_CONSISTENCY = 3  # of the screen bases
SYSTEMATIC = 0.75
MAX_SCREEN_ITEMS = 16  # per boundary: ambiguous items closest to s = 0 first
MAX_CONFIRM_ITEMS = 10  # per boundary: screened items with most flip bases first
MAX_MATRIX_ROWS = 250  # per boundary: test-split descriptions first, then largest gap
ITEM_FIELDS = ("item", "boundary", "frame", "evidence", "version")
STIMULUS_KEY = ("boundary", "frame", "evidence", "base_id", "wording", "country")


# ---------------------------------------------------------------- inputs


def load_inputs(geo_dir=GEO):
    """Every input file, its sha256, and the explore / confirm / test bases."""
    inputs = {name.split(".")[0]: read_json(geo_dir / name) for name in INPUT_FILES}
    inputs["digests"] = {name: file_digest(geo_dir / name) for name in INPUT_FILES}
    inputs["bases"] = {}
    for boundary in BOUNDARIES:
        files = inputs["splits"][boundary]
        inputs["bases"][boundary] = {
            split: read_csv(geo_dir / files[split])
            for split in ("explore", "confirm", "test")
        }
        for split in ("explore", "confirm", "test"):
            inputs["digests"][files[split]] = file_digest(geo_dir / files[split])
    inputs["dir"] = str(geo_dir)
    return inputs


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


def render_tasks(prompts, descriptions):
    """Every task prompt: boundary x version x paraphrase x order."""
    tasks = []
    for boundary in BOUNDARIES:
        spec = prompts["boundaries"][boundary]
        classes = descriptions["boundaries"][boundary]
        lower, higher = classes["lower_class"], classes["higher_class"]
        require(
            set(spec["order_a"]) == {lower, higher},
            f"{boundary}: task classes differ from description classes",
        )
        for version in prompts["versions"]:
            require(version in ("defined", "undefined"), f"Unknown version {version}")
            for index, para in enumerate(spec["paraphrases"]):
                for order in prompts["orders"]:
                    require(order in ("A", "B"), f"Unknown order {order}")
                    first, second = spec["order_a"][:: 1 if order == "A" else -1]
                    parts = [
                        para["intro"].format(
                            first_phrase=spec["phrases"][first],
                            second_phrase=spec["phrases"][second],
                        )
                    ]
                    if version == "defined":
                        parts += [
                            para["definitions"][first],
                            para["definitions"][second],
                        ]
                    parts.append(para["answer"].format(first=first, second=second))
                    tasks.append(
                        {
                            "task_id": f"{boundary}:{version}:p{index}:{order}",
                            "boundary": boundary,
                            "version": version,
                            "paraphrase": index,
                            "order": order,
                            "system_prompt": "\n\n".join(parts),
                            "lower": lower,
                            "higher": higher,
                        }
                    )
    return tasks


def paraphrase_count(prompts, boundary):
    return len(prompts["boundaries"][boundary]["paraphrases"])


# ---------------------------------------------------------------- V0 static checks


def static_findings(inputs):
    """Every problem with the inputs, as readable strings (empty when clean)."""
    findings = []
    descriptions, countries = inputs["descriptions"], inputs["countries"]
    terms = [term.lower() for term in descriptions["excluded_terms"]]

    def scan(label, text):
        lowered = text.lower()
        findings.extend(
            f"{label}: contains excluded term {term!r}"
            for term in terms
            if term in lowered
        )

    for boundary in BOUNDARIES:
        spec = descriptions["boundaries"][boundary]
        splits = inputs["bases"][boundary]
        for frame_id, frame in spec["frames"].items():
            label = f"{boundary}/{frame_id}"
            fields = frame_fields(frame)
            if fields.count("country") != 1 or fields.count("evidence") != 1:
                findings.append(
                    f"{label}: needs exactly one {{country}} and {{evidence}}"
                )
            if "{place}{country}" not in frame:
                findings.append(f"{label}: the country slot must follow {{place}}")
            scan(label, re.sub(r"\{\w+\}", " ", frame))
            needed = set(fields) - {"country", "evidence"}
            for rows in splits.values():
                for row in rows:
                    missing = needed - set(row)
                    if missing:
                        findings.append(
                            f"{boundary}/{row['base_id']}: missing {sorted(missing)}"
                        )
                    for key in needed & set(row):
                        scan(f"{boundary}/{row['base_id']}.{key}", row[key])
        roles = [entry["role"] for entry in spec["evidence"].values()]
        for role in ("anchor_lower", "anchor_higher"):
            if roles.count(role) != 1:
                findings.append(f"{boundary}: needs exactly one {role} evidence entry")
        seen = set()
        for evidence_id, entry in spec["evidence"].items():
            label = f"{boundary}/{evidence_id}"
            if entry["role"] not in ROLES:
                findings.append(f"{label}: unknown role {entry['role']!r}")
            wordings = entry["wordings"]
            if len(wordings) != 2 or wordings[0] == wordings[1]:
                findings.append(f"{label}: needs two different wordings")
            for index, text in enumerate(wordings, start=1):
                scan(f"{label}/W{index}", text)
                if text in seen:
                    findings.append(f"{label}/W{index}: wording reused")
                seen.add(text)
        ids = {
            split: {row["base_id"] for row in rows} for split, rows in splits.items()
        }
        for one, two in (
            ("explore", "confirm"),
            ("explore", "test"),
            ("confirm", "test"),
        ):
            if ids[one] & ids[two]:
                findings.append(f"{boundary}: {one} and {two} bases overlap")
        if inputs["splits"]["screen_bases"] > len(ids["explore"]):
            findings.append(f"{boundary}: more screen bases than explore bases")
        findings += slot_findings(boundary, spec, splits["explore"][0], countries)

    for kind in ("real", "fictional"):
        for name in countries[kind].values():
            scan(f"country {name}", name)
    for boundary, spec in inputs["task_prompts"]["boundaries"].items():
        for index, para in enumerate(spec["paraphrases"]):
            text = " ".join(
                [para["intro"], para["answer"], *para["definitions"].values()]
            ).lower()
            for word in (
                "country",
                "countries",
                "location",
                "region",
                "unclear",
                "uncertain",
            ):
                if word in text:
                    findings.append(f"task {boundary}/p{index}: mentions {word!r}")
    return findings


def slot_findings(boundary, spec, base, countries):
    """The country phrase must be the only difference from the no-country text."""
    findings = []
    evidence = next(iter(spec["evidence"].values()))["wordings"][0]
    for frame_id, frame in spec["frames"].items():
        plain = render(frame, base, evidence, "")
        for cid, display, _ in country_conditions(countries, ("real", "fictional")):
            phrase = country_phrase(countries, display)
            text = render(frame, base, evidence, phrase)
            if text.count(display) != 1 or text.replace(phrase, "", 1) != plain:
                findings.append(f"{boundary}/{frame_id}: slot error for {cid}")
    return findings


# ---------------------------------------------------------------- scoring


class Scorer:
    """scoring.score_prompt, one prompt per forward pass, with a cache."""

    def __init__(self, model_key, work, device="cuda", dtype=None):
        require(model_key in S.MODELS, f"Unknown model {model_key}")
        self.spec = dict(S.MODELS[model_key])
        if dtype:
            self.spec["dtype"] = dtype
        self.tokenizer, self.model, self.revision = S.load_model(self.spec, device)
        self.model_key, self.device = model_key, device
        name = f"{model_key}_{self.spec['dtype']}_{self.revision[:10]}.jsonl"
        self.cache_path = Path(work) / "cache" / name
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        if self.cache_path.exists():
            with self.cache_path.open(encoding="utf-8") as stream:
                for line in stream:
                    record = json.loads(line)
                    self.cache[record["key"]] = record

    def describe(self):
        return {
            "model_key": self.model_key,
            "hf_id": self.spec["hf_id"],
            "dtype": self.spec["dtype"],
            "revision": self.revision,
        }

    @staticmethod
    def key(prompt):
        payload = [
            prompt["system_prompt"],
            prompt["text"],
            prompt["lower"],
            prompt["higher"],
        ]
        return hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()

    def score_fresh(self, prompt):
        """Score one prompt without the cache (used by V1)."""
        result = S.score_prompt(
            self.model,
            self.tokenizer,
            self.spec,
            prompt["system_prompt"],
            prompt["text"],
            [prompt["lower"], prompt["higher"]],
            self.device,
        )
        return {n: e["logprob"] for n, e in result["answers"].items()}

    def score(self, prompts):
        """Per prompt: log-probability of each class answer, and their mass."""
        keys = [self.key(prompt) for prompt in prompts]
        pending = {}
        for key, prompt in zip(keys, prompts):
            if key not in self.cache:
                pending.setdefault(key, prompt)
        started = time.monotonic()
        with self.cache_path.open("a", encoding="utf-8") as stream:
            for count, (key, prompt) in enumerate(pending.items(), start=1):
                result = S.score_prompt(
                    self.model,
                    self.tokenizer,
                    self.spec,
                    prompt["system_prompt"],
                    prompt["text"],
                    [prompt["lower"], prompt["higher"]],
                    self.device,
                )
                record = {
                    "key": key,
                    "lp": {n: e["logprob"] for n, e in result["answers"].items()},
                    "mass": result["answer_mass"],
                }
                self.cache[key] = record
                stream.write(json.dumps(record) + "\n")
                if count % 500 == 0:
                    rate = count / (time.monotonic() - started)
                    print(f"  {count}/{len(pending)} at {rate:.1f}/s", file=sys.stderr)
        if pending:
            rate = len(pending) / max(time.monotonic() - started, 1e-9)
            print(
                f"  scored {len(pending)} new prompts at {rate:.1f}/s", file=sys.stderr
            )
        return [self.cache[key] for key in keys]


# ---------------------------------------------------------------- stimuli and aggregation


def build_stimuli(
    inputs, boundary, evidence_ids, bases, wordings, countries, frame_ids=None
):
    spec = inputs["descriptions"]["boundaries"][boundary]
    frames = frame_ids or list(spec["frames"])
    out = []
    for frame_id, evidence_id, base, wording, (cid, display, kind) in product(
        frames, evidence_ids, bases, wordings, countries
    ):
        entry = spec["evidence"][evidence_id]
        evidence = entry["wordings"][wording - 1]
        frame = spec["frames"][frame_id]
        out.append(
            {
                "boundary": boundary,
                "frame": frame_id,
                "evidence": evidence_id,
                "role": entry["role"],
                "base_id": base["base_id"],
                "wording": wording,
                "country": cid,
                "country_kind": kind,
                "text": render(
                    frame, base, evidence, country_phrase(inputs["countries"], display)
                ),
                "template": render(frame, base, evidence, " in [COUNTRY]"),
            }
        )
    return out


def build_prompts(stimuli, tasks, versions):
    by_boundary = defaultdict(list)
    for task in tasks:
        if task["version"] in versions:
            by_boundary[task["boundary"]].append(task)
    return [
        {**stimulus, **task}
        for stimulus in stimuli
        for task in by_boundary[stimulus["boundary"]]
    ]


def scored_rows(scorer, prompts):
    return [
        {
            **prompt,
            "d": record["lp"][prompt["higher"]] - record["lp"][prompt["lower"]],
            "mass": record["mass"],
        }
        for prompt, record in zip(prompts, scorer.score(prompts))
    ]


def robust_label(dbars, margin=MARGIN, min_agree=MIN_AGREE):
    """+1 / -1 for a robust higher / lower label, 0 when unstable."""
    s = statistics.fmean(dbars)
    if abs(s) < margin:
        return 0
    agree = sum(1 for value in dbars if value * s > 0)
    return (1 if s > 0 else -1) if agree >= min_agree else 0


def collapse(rows, n_paraphrases):
    """Per stimulus x version: d-bar per paraphrase (order-averaged), s and label."""
    cells = defaultdict(dict)
    for row in rows:
        key = tuple(row[field] for field in STIMULUS_KEY)
        cells[key + (row["version"], row["paraphrase"])][row["order"]] = (
            row["d"],
            row["mass"],
        )
    grouped = defaultdict(dict)
    for key, orders in cells.items():
        require(set(orders) == {"A", "B"}, f"Missing prompt order for {key}")
        (d_a, m_a), (d_b, m_b) = orders["A"], orders["B"]
        grouped[key[:-1]][key[-1]] = {
            "dbar": (d_a + d_b) / 2,
            "order_effect": d_a - d_b,
            "mass": min(m_a, m_b),
        }
    out = {}
    for key, paras in grouped.items():
        require(len(paras) == n_paraphrases, f"Missing paraphrases for {key}")
        dbars = [paras[index]["dbar"] for index in sorted(paras)]
        out[key] = {
            "dbar": dbars,
            "s": statistics.fmean(dbars),
            "label": robust_label(dbars),
            "order_effect": statistics.fmean(p["order_effect"] for p in paras.values()),
            "mass_min": min(p["mass"] for p in paras.values()),
        }
    return out


def screen_countries(shifts, consistency=SCREEN_CONSISTENCY, top=TOP_COUNTRIES):
    """Countries consistently above / below the item's mean over real countries.

    shifts: {country: [s minus the real-country mean, per base]}."""
    up, down = [], []
    for country, values in shifts.items():
        mean = statistics.fmean(values)
        if sum(value > 0 for value in values) >= consistency and mean > 0:
            up.append((mean, country))
        if sum(value < 0 for value in values) >= consistency and mean < 0:
            down.append((mean, country))
    return (
        [c for _, c in sorted(up, reverse=True)[:top]],
        [c for _, c in sorted(down)[:top]],
    )


def pair_stats(high, low):
    """high, low: {cell: s}. Share and mean of s(high) - s(low) over shared cells."""
    shared = sorted(set(high) & set(low))
    require(shared, "No shared descriptions for a country pair")
    gaps = [high[cell] - low[cell] for cell in shared]
    return {
        "n": len(gaps),
        "share_positive": sum(gap > 0 for gap in gaps) / len(gaps),
        "mean_gap": statistics.fmean(gaps),
    }


# ---------------------------------------------------------------- stages


def reproduce_reference(scorer, path=GEO / "scorer_reference.jsonl"):
    """V1: re-score stored reference prompts, bypassing the cache, and compare.

    The reference prompts are 8 v1 Stage A prompts scored by the same code; they
    apply only to the same model, dtype and revision."""
    if not path.exists():
        return {"available": False, "pass": True}
    reference = [
        r
        for r in read_jsonl(path)
        if r["model_key"] == scorer.model_key
        and r["dtype"] == scorer.spec["dtype"]
        and r["revision"] == scorer.revision
    ]
    if not reference:
        return {"available": False, "pass": True}
    delta = max(
        abs(scorer.score_fresh(r)[name] - r["logprob"][name])
        for r in reference
        for name in (r["lower"], r["higher"])
    )
    return {
        "available": True,
        "sample": len(reference),
        "max_abs_delta": delta,
        "pass": delta <= REPRODUCE_TOLERANCE,
    }


def anchor_summary(rows, boundary, version, index):
    """V2 for one task paraphrase: anchor accuracy (order-averaged) and answer mass."""
    expected = {"anchor_lower": -1, "anchor_higher": 1}
    mine = [r for r in rows if r["version"] == version and r["paraphrase"] == index]
    cells = defaultdict(dict)
    for r in mine:
        cells[(r["role"], r["base_id"], r["wording"])][r["order"]] = r["d"]
    accuracy = {
        role: statistics.fmean(
            (o["A"] + o["B"]) / 2 * sign > 0 for k, o in cells.items() if k[0] == role
        )
        for role, sign in expected.items()
    }
    mass = {
        order: statistics.median(r["mass"] for r in mine if r["order"] == order)
        for order in ("A", "B")
    }
    return {
        "boundary": boundary,
        "version": version,
        "paraphrase": index,
        "anchor_accuracy_lower": accuracy["anchor_lower"],
        "anchor_accuracy_higher": accuracy["anchor_higher"],
        "mass_median_A": mass["A"],
        "mass_median_B": mass["B"],
        "pass": min(accuracy.values()) >= ANCHOR_ACCURACY
        and min(mass.values()) >= MIN_MASS,
    }


def stage_check(inputs, scorer, boundaries):
    tasks = render_tasks(inputs["task_prompts"], inputs["descriptions"])
    summary = {"static_findings": static_findings(inputs)}
    require(not summary["static_findings"], "V0 static checks failed")
    summary["reproduce"] = reproduce_reference(
        scorer, Path(inputs["dir"]) / "scorer_reference.jsonl"
    )
    summary["task_prompts"] = []
    for boundary in boundaries:
        spec = inputs["descriptions"]["boundaries"][boundary]
        anchors = [e for e, x in spec["evidence"].items() if x["role"] != "borderline"]
        stimuli = build_stimuli(
            inputs,
            boundary,
            anchors,
            inputs["bases"][boundary]["explore"],
            (1, 2),
            country_conditions(inputs["countries"], ("none",)),
        )
        versions = inputs["task_prompts"]["versions"]
        rows = scored_rows(scorer, build_prompts(stimuli, tasks, versions))
        for version in versions:
            for index in range(paraphrase_count(inputs["task_prompts"], boundary)):
                summary["task_prompts"].append(
                    anchor_summary(rows, boundary, version, index)
                )
    summary["pass"] = summary["reproduce"]["pass"] and all(
        entry["pass"] for entry in summary["task_prompts"]
    )
    return summary


def stage_ambiguity(inputs, scorer, boundaries):
    tasks = render_tasks(inputs["task_prompts"], inputs["descriptions"])
    items = []
    for boundary in boundaries:
        spec = inputs["descriptions"]["boundaries"][boundary]
        stimuli = build_stimuli(
            inputs,
            boundary,
            list(spec["evidence"]),
            inputs["bases"][boundary]["explore"],
            (1, 2),
            country_conditions(inputs["countries"], ("none",)),
        )
        versions = inputs["task_prompts"]["versions"]
        rows = scored_rows(scorer, build_prompts(stimuli, tasks, versions))
        cells = collapse(rows, paraphrase_count(inputs["task_prompts"], boundary))
        groups = defaultdict(list)
        for key, cell in cells.items():
            groups[(key[1], key[2], key[6])].append((key, cell))
        for (frame, evidence, version), members in sorted(groups.items()):
            values = [cell["s"] for _, cell in members]
            by_base = defaultdict(dict)
            for key, cell in members:
                by_base[key[3]][key[4]] = cell["s"]
            role = spec["evidence"][evidence]["role"]
            median = statistics.median(values)
            items.append(
                {
                    "item": f"{boundary}:{frame}:{evidence}",
                    "boundary": boundary,
                    "frame": frame,
                    "evidence": evidence,
                    "role": role,
                    "version": version,
                    "median_s": median,
                    "min_s": min(values),
                    "max_s": max(values),
                    "wording_gap": statistics.median(
                        abs(w[1] - w[2]) for w in by_base.values()
                    ),
                    "order_effect": statistics.fmean(
                        cell["order_effect"] for _, cell in members
                    ),
                    "share_robust": statistics.fmean(
                        cell["label"] != 0 for _, cell in members
                    ),
                    "ambiguous": role == "borderline" and abs(median) <= AMBIGUITY_D,
                }
            )
    return {"items": items, "n_ambiguous": sum(item["ambiguous"] for item in items)}


def item_cells(inputs, scorer, item, bases, wordings, countries):
    """{(base, wording, country): cell} and the stimuli, for one item x version."""
    tasks = render_tasks(inputs["task_prompts"], inputs["descriptions"])
    stimuli = build_stimuli(
        inputs,
        item["boundary"],
        [item["evidence"]],
        bases,
        wordings,
        countries,
        [item["frame"]],
    )
    rows = scored_rows(scorer, build_prompts(stimuli, tasks, [item["version"]]))
    cells = collapse(rows, paraphrase_count(inputs["task_prompts"], item["boundary"]))
    texts = {(s["base_id"], s["wording"], s["country"]): s for s in stimuli}
    return {(k[3], k[4], k[5]): cell for k, cell in cells.items()}, texts


def stage_screen(inputs, scorer, boundaries, ambiguity):
    countries = country_conditions(inputs["countries"])
    real = [cid for cid, _, kind in countries if kind == "real"]
    n_screen = inputs["splits"]["screen_bases"]
    chosen = []
    for boundary in boundaries:
        candidates = [
            i
            for i in ambiguity["items"]
            if i["ambiguous"] and i["boundary"] == boundary
        ]
        candidates.sort(key=lambda i: abs(i["median_s"]))
        chosen += candidates[:MAX_SCREEN_ITEMS]
    out = []
    for item in chosen:
        bases = inputs["bases"][item["boundary"]]["explore"][:n_screen]
        cells, _ = item_cells(inputs, scorer, item, bases, (1,), countries)
        base_ids = [base["base_id"] for base in bases]
        centre = {
            b: statistics.fmean(cells[(b, 1, c)]["s"] for c in real) for b in base_ids
        }
        shifts = {
            c: [cells[(b, 1, c)]["s"] - centre[b] for b in base_ids]
            for c, _, _ in countries
        }
        mean_shift = {c: statistics.fmean(v) for c, v in shifts.items()}
        up, down = screen_countries({c: shifts[c] for c in real})
        flips = {
            b: {
                "high": [c for c in real if cells[(b, 1, c)]["label"] == 1],
                "low": [c for c in real if cells[(b, 1, c)]["label"] == -1],
            }
            for b in base_ids
        }
        out.append(
            {
                **{k: item[k] for k in ITEM_FIELDS},
                "up": up,
                "down": down,
                "mean_shift": mean_shift,
                "spread": max(mean_shift[c] for c in real)
                - min(mean_shift[c] for c in real),
                "flip_bases": sum(1 for f in flips.values() if f["high"] and f["low"]),
                "flips": flips,
                "selected": bool(up and down),
            }
        )
    out.sort(key=lambda entry: (-entry["flip_bases"], -entry["spread"]))
    return {"items": out, "n_selected": sum(entry["selected"] for entry in out)}


def flip_record(entry, split, cell_of, text_of, base_id, wording, high, low, fictional):
    h, l, none = (cell_of(base_id, wording, c) for c in (high, low, NONE))
    return {
        **{k: entry[k] for k in ITEM_FIELDS},
        "split": split,
        "base_id": base_id,
        "wording": wording,
        "template": text_of(base_id, wording, high)["template"],
        "text_high": text_of(base_id, wording, high)["text"],
        "text_low": text_of(base_id, wording, low)["text"],
        "country_high": high,
        "country_low": low,
        "s_high": h["s"],
        "s_low": l["s"],
        "dbar_high": h["dbar"],
        "dbar_low": l["dbar"],
        "s_none": none["s"],
        "label_none": none["label"],
        "s_fictional": {c: cell_of(base_id, wording, c)["s"] for c in fictional},
        "labels_fictional": {
            c: cell_of(base_id, wording, c)["label"] for c in fictional
        },
        "mass_min": min(h["mass_min"], l["mass_min"]),
    }


def evaluate_pairs(inputs, scorer, entry, split):
    """Systematic country pairs and their flips for one item x version on a split."""
    chosen = set(entry["up"]) | set(entry["down"])
    countries = [
        c
        for c in country_conditions(inputs["countries"])
        if c[0] in chosen or c[2] != "real"
    ]
    fictional = [c for c, _, kind in countries if kind == "fictional"]
    bases = inputs["bases"][entry["boundary"]][split]
    cells, texts = item_cells(inputs, scorer, entry, bases, (1, 2), countries)
    base_ids = [base["base_id"] for base in bases]
    reference = statistics.median(
        abs(cells[(b, 1, NONE)]["s"] - cells[(b, 2, NONE)]["s"]) for b in base_ids
    )
    descriptions = [(b, w) for b in base_ids for w in (1, 2)]
    pairs, flips = [], []
    for high in entry["up"]:
        for low in entry["down"]:
            stats = pair_stats(
                {c: cells[(*c, high)]["s"] for c in descriptions},
                {c: cells[(*c, low)]["s"] for c in descriptions},
            )
            found = [
                c
                for c in descriptions
                if cells[(*c, high)]["label"] == 1 and cells[(*c, low)]["label"] == -1
            ]
            stats.update(
                high=high,
                low=low,
                flips=len(found),
                systematic=stats["share_positive"] >= SYSTEMATIC
                and stats["mean_gap"] > reference,
            )
            pairs.append(stats)
            if stats["systematic"]:
                flips += [
                    flip_record(
                        entry,
                        split,
                        lambda b, w, c: cells[(b, w, c)],
                        lambda b, w, c: texts[(b, w, c)],
                        base_id,
                        wording,
                        high,
                        low,
                        fictional,
                    )
                    for base_id, wording in found
                ]
    return {
        **{k: entry[k] for k in ITEM_FIELDS},
        "split": split,
        "wording_reference": reference,
        "pairs": pairs,
        "flips": flips,
    }


def stage_confirm(inputs, scorer, boundaries, screen):
    chosen = []
    for boundary in boundaries:
        selected = [
            e for e in screen["items"] if e["selected"] and e["boundary"] == boundary
        ]
        chosen += selected[:MAX_CONFIRM_ITEMS]  # screen items are sorted by yield
    results = [evaluate_pairs(inputs, scorer, entry, "confirm") for entry in chosen]
    return {"items": results, "n_flips": sum(len(r["flips"]) for r in results)}


def stage_test(inputs, scorer, boundaries, confirm):
    """Only pairs that were systematic and produced flips on the confirm split."""
    results = []
    for result in confirm["items"]:
        keep = {(f["country_high"], f["country_low"]) for f in result["flips"]}
        if result["boundary"] not in boundaries or not keep:
            continue
        entry = {
            **{k: result[k] for k in ITEM_FIELDS},
            "up": sorted({high for high, _ in keep}),
            "down": sorted({low for _, low in keep}),
        }
        outcome = evaluate_pairs(inputs, scorer, entry, "test")
        outcome["pairs"] = [
            p for p in outcome["pairs"] if (p["high"], p["low"]) in keep
        ]
        outcome["flips"] = [
            f for f in outcome["flips"] if (f["country_high"], f["country_low"]) in keep
        ]
        results.append(outcome)
    return {"items": results, "n_flips": sum(len(r["flips"]) for r in results)}


def probe(inputs, scorer, boundary, sentences):
    """Score candidate evidence sentences (explore bases, no country, all tasks).

    For drafting inputs between iterations; nothing is written but the cache."""
    tasks = render_tasks(inputs["task_prompts"], inputs["descriptions"])
    spec = inputs["descriptions"]["boundaries"][boundary]
    out = []
    for index, sentence in enumerate(sentences):
        frame = spec["frames"]["f1"]
        stimuli = [
            {
                "boundary": boundary,
                "frame": "f1",
                "evidence": f"probe{index}",
                "base_id": base["base_id"],
                "wording": 1,
                "country": NONE,
                "text": render(frame, base, sentence, ""),
            }
            for base in inputs["bases"][boundary]["explore"]
        ]
        versions = inputs["task_prompts"]["versions"]
        rows = scored_rows(scorer, build_prompts(stimuli, tasks, versions))
        cells = collapse(rows, paraphrase_count(inputs["task_prompts"], boundary))
        for version in versions:
            mine = [c for k, c in cells.items() if k[6] == version]
            per_para = list(zip(*(c["dbar"] for c in mine)))
            out.append(
                {
                    "sentence": sentence,
                    "version": version,
                    "median_s": statistics.median(c["s"] for c in mine),
                    "min_s": min(c["s"] for c in mine),
                    "paraphrase_medians": [statistics.median(v) for v in per_para],
                }
            )
    return out


def stage_matrix(inputs, scorer, boundaries, confirm, test=None):
    """Every country on every description that flipped (rows x countries)."""
    countries = country_conditions(inputs["countries"])
    flips = [
        f
        for summary in (test, confirm)
        if summary
        for r in summary["items"]
        for f in r["flips"]
        if f["boundary"] in boundaries
    ]
    flips.sort(key=lambda f: (f["split"] != "test", -(f["s_high"] - f["s_low"])))
    descriptions = {}
    for f in flips:
        key = (f["item"], f["version"], f["split"], f["base_id"], f["wording"])
        descriptions.setdefault(key, f)
    chosen = []
    for boundary in boundaries:
        mine = [f for f in descriptions.values() if f["boundary"] == boundary]
        chosen += mine[:MAX_MATRIX_ROWS]
    return {
        "countries": [c for c, _, _ in countries],
        "rows": score_matrix_rows(inputs, scorer, chosen),
        "n_rows": len(chosen),
        "n_descriptions_with_flips": len(descriptions),
    }


def score_matrix_rows(inputs, scorer, descriptions):
    """One matrix row per description: every country condition's cell."""
    countries = country_conditions(inputs["countries"])
    rows = []
    for f in descriptions:
        bases = inputs["bases"][f["boundary"]][f["split"]]
        base = next(b for b in bases if b["base_id"] == f["base_id"])
        cells, _ = item_cells(inputs, scorer, f, [base], (f["wording"],), countries)
        cell = {c: cells[(f["base_id"], f["wording"], c)] for c, _, _ in countries}
        rows.append(
            {
                **{k: f[k] for k in ITEM_FIELDS},
                "split": f["split"],
                "base_id": f["base_id"],
                "wording": f["wording"],
                "template": f["template"],
                "s": {c: x["s"] for c, x in cell.items()},
                "label": {c: x["label"] for c, x in cell.items()},
                "dbar": {c: x["dbar"] for c, x in cell.items()},
                "mass_min": min(x["mass_min"] for x in cell.values()),
            }
        )
    return rows


def stage_transfer(inputs, scorer, source):
    """Re-score another run's matrix rows with this scorer's model, and compare."""
    rows = score_matrix_rows(inputs, scorer, source["rows"])
    real = [c for c in source["countries"] if c in REAL_IDS()]
    same = [
        r["label"][c] == old["label"][c]
        for r, old in zip(rows, source["rows"])
        for c in source["countries"]
    ]
    gaps = [
        (r["s"][c] - statistics.fmean(r["s"][x] for x in real))
        * (old["s"][c] - statistics.fmean(old["s"][x] for x in real))
        for r, old in zip(rows, source["rows"])
        for c in real
    ]
    still_flip = [
        1 in [r["label"][c] for c in real] and -1 in [r["label"][c] for c in real]
        for r in rows
    ]
    return {
        "countries": source["countries"],
        "rows": rows,
        "n_rows": len(rows),
        "n_descriptions_with_flips": sum(still_flip),
        "label_agreement": statistics.fmean(same),
        "same_direction_share": statistics.fmean(g > 0 for g in gaps),
        "rows_still_flipping": sum(still_flip),
    }


# ---------------------------------------------------------------- report


def fmt(value, digits=2):
    return f"{value:+.{digits}f}"


def report_checks(check):
    lines = [
        "## V0–V2 Checks",
        "",
        f"- V0 static findings: {len(check['static_findings'])}",
    ]
    rep = check["reproduce"]
    if rep["available"]:
        lines.append(
            f"- V1 reproduces the reference scores: max |Δ log P| = {rep['max_abs_delta']:.4f} over "
            f"{rep['sample']} prompts (limit {REPRODUCE_TOLERANCE})"
        )
    else:
        lines.append("- V1: no reference scores for this model, dtype and revision")
    lines += [
        "",
        "| Boundary | Version | Paraphrase | Anchor acc. (lower) | Anchor acc. (higher) "
        "| Mass A | Mass B | Pass |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for e in check["task_prompts"]:
        lines.append(
            f"| {e['boundary']} | {e['version']} | p{e['paraphrase']} | "
            f"{e['anchor_accuracy_lower']:.2f} | {e['anchor_accuracy_higher']:.2f} | "
            f"{e['mass_median_A']:.3f} | {e['mass_median_B']:.3f} | "
            f"{'yes' if e['pass'] else '**no**'} |"
        )
    return lines


def report_ambiguity(ambiguity):
    lines = [
        "",
        "## V3 Ambiguity (No Country, Explore Bases)",
        "",
        "s is the task-robust score: higher-class log odds averaged over orders and "
        f"paraphrases. Ambiguous if |median s| ≤ {AMBIGUITY_D}.",
        "",
        "| Item | Role | Version | Median s | Range | Wording gap | Order effect "
        "| Robust share | Ambiguous |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for e in ambiguity["items"]:
        lines.append(
            f"| {e['item']} | {e['role']} | {e['version']} | {fmt(e['median_s'])} | "
            f"{fmt(e['min_s'])} to {fmt(e['max_s'])} | {e['wording_gap']:.2f} | "
            f"{fmt(e['order_effect'])} | {e['share_robust']:.2f} | "
            f"{'**yes**' if e['ambiguous'] else 'no'} |"
        )
    return lines


def report_screen(screen, fictional):
    lines = [
        "",
        "## V4 Screen (Screen Bases, Wording 1, All Countries)",
        "",
        "Shifts are relative to the item's mean over real countries.",
        "",
        "| Item | Version | Up | Down | Spread | Bases with a flip | Fictional shifts "
        "| Selected |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for e in screen["items"]:
        fic = ", ".join(f"{c} {fmt(e['mean_shift'][c])}" for c in fictional)
        lines.append(
            f"| {e['item']} | {e['version']} | {', '.join(e['up']) or '–'} | "
            f"{', '.join(e['down']) or '–'} | {e['spread']:.2f} | {e['flip_bases']} | "
            f"{fic} | {'yes' if e['selected'] else 'no'} |"
        )
    return lines


def report_pairs(summary, title):
    lines = [
        "",
        f"## {title} (Both Wordings)",
        "",
        "| Item | Version | High | Low | Share s(high) > s(low) | Mean gap "
        "| Wording ref. | Systematic | Flips |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in summary["items"]:
        for p in r["pairs"]:
            lines.append(
                f"| {r['item']} | {r['version']} | {p['high']} | {p['low']} | "
                f"{p['share_positive']:.2f} | {fmt(p['mean_gap'])} | "
                f"{r['wording_reference']:.2f} | "
                f"{'**yes**' if p['systematic'] else 'no'} | {p['flips']} |"
            )
    lines += ["", f"Flips from systematic pairs: {summary['n_flips']}"]
    return lines


def report_examples(flips):
    lines = [
        "",
        "## Example Flips",
        "",
        "One description; only the country differs. s > 0 favours the higher-evidence "
        "class (Riot or Battle). Brackets give each paraphrase's order-averaged d. "
        "One example per item, version, country pair and split.",
        "",
    ]
    seen = set()
    for f in sorted(flips, key=lambda f: -(f["s_high"] - f["s_low"])):
        key = (f["item"], f["version"], f["country_high"], f["country_low"], f["split"])
        if key in seen:
            continue
        seen.add(key)
        fic = ", ".join(f"{c} {fmt(v)}" for c, v in f["s_fictional"].items())
        lines += [
            f"- **{f['item']} · {f['version']} · {f['split']} · {f['base_id']} "
            f"W{f['wording']}**",
            f"  - {f['template']}",
            f"  - {f['country_high']}: s = {fmt(f['s_high'])} "
            f"[{', '.join(fmt(x, 1) for x in f['dbar_high'])}]",
            f"  - {f['country_low']}: s = {fmt(f['s_low'])} "
            f"[{', '.join(fmt(x, 1) for x in f['dbar_low'])}]",
            f"  - no country: s = {fmt(f['s_none'])}; fictional: {fic}",
        ]
    return lines


CLASS_NAMES = {
    "PR": {1: "Riot", -1: "Protest", 0: "unstable"},
    "BV": {1: "Battle", -1: "Violence against civilians", 0: "unstable"},
}


def write_matrix(name, matrix, out_dir):
    """Rows = descriptions, columns = countries; one CSV of labels, one of s."""
    countries = matrix["countries"]
    head = ["boundary", "item", "version", "split", "base_id", "wording", "description"]
    for kind in ("labels", "s"):
        with (out_dir / f"{name}_matrix_{kind}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(head + countries)
            for r in matrix["rows"]:
                values = [
                    (
                        CLASS_NAMES[r["boundary"]][r["label"][c]]
                        if kind == "labels"
                        else f"{r['s'][c]:.3f}"
                    )
                    for c in countries
                ]
                writer.writerow([r[k] for k in head[:-1]] + [r["template"]] + values)
    real = [c for c in countries if c in REAL_IDS()]
    lines = [
        "",
        "## Country Matrix",
        "",
        f"{matrix['n_rows']} descriptions (of {matrix['n_descriptions_with_flips']} with "
        f"a verified flip) scored on all {len(countries)} country conditions: "
        f"`{name}_matrix_labels.csv` (robust label per cell, 'unstable' if |s| < "
        f"{MARGIN} or fewer than {MIN_AGREE} paraphrases agree) and "
        f"`{name}_matrix_s.csv` (s per cell).",
        "",
    ]
    for r in sorted(
        matrix["rows"], key=lambda r: -statistics.pstdev(r["s"][c] for c in real)
    )[:6]:
        names = CLASS_NAMES[r["boundary"]]
        groups = {v: [c for c in real if r["label"][c] == v] for v in (1, -1, 0)}
        lines += [
            f"- **{r['item']} · {r['version']} · {r['split']} · {r['base_id']} W{r['wording']}**: {r['template']}",
            f"  - {names[1]} ({len(groups[1])}): {', '.join(groups[1]) or '–'}",
            f"  - {names[-1]} ({len(groups[-1])}): {', '.join(groups[-1]) or '–'}",
            f"  - unstable: {len(groups[0])}; no country: {names[r['label'][NONE]]} ({fmt(r['s'][NONE])})",
        ]
    return lines


def REAL_IDS():
    return set(read_json(GEO / "countries.json")["real"])


def write_report(name, work_dir, out_dir):
    summaries = {
        stage: read_json(work_dir / f"{stage}.json")
        for stage in ("check", "ambiguity", "screen", "confirm", "test")
        if (work_dir / f"{stage}.json").exists()
    }
    require("check" in summaries, "No check stage to report")
    scorer = summaries["check"]["meta"]["scorer"]
    lines = [
        f"# Geographic Flip Dataset: {name}",
        "",
        f"Generated by `geoflip/pipeline.py` from `{work_dir}`. Model `{scorer['hf_id']}` "
        f"({scorer['dtype']}, revision `{scorer['revision'][:10]}`), scored with "
        "`scoring.score_prompt`, one prompt per forward pass. Items were selected "
        "because they show country effects (PLAN.md § 1); counts are not "
        "prevalence estimates.",
        "",
    ]
    lines += report_checks(summaries["check"])
    if "ambiguity" in summaries:
        lines += report_ambiguity(summaries["ambiguity"])
    if "screen" in summaries:
        fictional = list(read_json(GEO / "countries.json")["fictional"])
        lines += report_screen(summaries["screen"], fictional)
    flips = []
    for stage, title in (("confirm", "V5 Confirm"), ("test", "V6 Test")):
        if stage in summaries:
            lines += report_pairs(summaries[stage], title)
            flips += [f for r in summaries[stage]["items"] for f in r["flips"]]
    if flips:
        lines += report_examples(flips)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (work_dir / "matrix.json").exists():
        lines += write_matrix(name, read_json(work_dir / "matrix.json"), out_dir)
    (out_dir / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (out_dir / f"{name}_flips.jsonl").open("w", encoding="utf-8") as stream:
        for f in flips:
            stream.write(json.dumps(f) + "\n")
    stages = out_dir / "stages" / work_dir.name
    stages.mkdir(parents=True, exist_ok=True)
    for path in sorted(work_dir.glob("*.json")):
        shutil.copy2(path, stages / path.name)
    return out_dir / f"{name}.md"


# ---------------------------------------------------------------- CLI

STAGES = (
    "check",
    "ambiguity",
    "screen",
    "confirm",
    "test",
    "matrix",
    "transfer",
    "report",
    "probe",
)
PREVIOUS = {
    "ambiguity": "check",
    "screen": "ambiguity",
    "confirm": "screen",
    "test": "confirm",
    "matrix": "confirm",
}


def run_stage(stage, inputs, scorer, boundaries, work_dir):
    if stage == "check":
        return stage_check(inputs, scorer, boundaries)
    if stage == "ambiguity":
        return stage_ambiguity(inputs, scorer, boundaries)
    previous = read_json(work_dir / f"{PREVIOUS[stage]}.json")
    if stage == "matrix":
        test = work_dir / "test.json"
        return stage_matrix(
            inputs,
            scorer,
            boundaries,
            previous,
            read_json(test) if test.exists() else None,
        )
    run = {"screen": stage_screen, "confirm": stage_confirm, "test": stage_test}
    return run[stage](inputs, scorer, boundaries, previous)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--iteration", required=True)
    parser.add_argument("--model", default="llama3.2-3b", choices=sorted(S.MODELS))
    parser.add_argument("--dtype", default=None, help="exploratory dtype override")
    parser.add_argument("--boundary", choices=BOUNDARIES, action="append")
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument(
        "--inputs", type=Path, default=GEO, help="this iteration's input directory"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--sentence", action="append", help="probe: candidate evidence sentence"
    )
    parser.add_argument(
        "--source", type=Path, help="transfer: work directory holding matrix.json"
    )
    parser.add_argument(
        "--max-rows", type=int, help="transfer: first N source rows (test first)"
    )
    args = parser.parse_args(argv)
    boundaries = tuple(args.boundary or BOUNDARIES)
    name = f"{args.iteration}_{'-'.join(boundaries)}"
    work_dir = args.work / args.iteration / "-".join(boundaries)
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "report":
        print(write_report(name, work_dir, RESULTS / args.iteration))
        return 0
    inputs = load_inputs(args.inputs)
    findings = static_findings(inputs)
    if findings:
        print("V0 static checks failed:\n" + "\n".join(findings), file=sys.stderr)
        return 1
    if args.stage in PREVIOUS:
        require(
            (work_dir / f"{PREVIOUS[args.stage]}.json").exists(),
            f"Run {PREVIOUS[args.stage]} first",
        )
        require(
            read_json(work_dir / "check.json")["meta"]["digests"] == inputs["digests"],
            "Inputs changed since check: start a new iteration",
        )
    if args.stage == "test":
        require(
            not (work_dir / "test.json").exists(),
            "The test split is scored once per iteration",
        )
    scorer = Scorer(args.model, args.work, args.device, args.dtype)
    if args.stage == "probe":
        require(
            args.sentence and len(boundaries) == 1, "probe: one --boundary, --sentence"
        )
        for row in probe(inputs, scorer, boundaries[0], args.sentence):
            paras = " ".join(f"{v:+.1f}" for v in row["paraphrase_medians"])
            print(
                f"{row['median_s']:+6.2f} (min {row['min_s']:+.2f}) "
                f"[{paras}] {row['version']:9} {row['sentence']}"
            )
        return 0
    started = time.monotonic()
    if args.stage == "transfer":
        require(args.source, "transfer needs --source")
        source = read_json(args.source / "matrix.json")
        if args.max_rows:
            source["rows"] = source["rows"][: args.max_rows]
        summary = stage_transfer(inputs, scorer, source)
        summary["source"] = str(args.source)
        write_matrix(name, summary, work_dir)
    else:
        summary = run_stage(args.stage, inputs, scorer, boundaries, work_dir)
    summary["meta"] = {
        "stage": args.stage,
        "iteration": args.iteration,
        "boundaries": boundaries,
        "digests": inputs["digests"],
        "producer_sha256": file_digest(PRODUCER),
        "scorer": scorer.describe(),
        "seconds": round(time.monotonic() - started, 1),
    }
    write_json(work_dir / f"{args.stage}.json", summary)
    keys = (
        "pass",
        "n_ambiguous",
        "n_selected",
        "n_flips",
        "n_rows",
        "label_agreement",
        "same_direction_share",
        "rows_still_flipping",
    )
    print(json.dumps({k: v for k, v in summary.items() if k in keys}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
