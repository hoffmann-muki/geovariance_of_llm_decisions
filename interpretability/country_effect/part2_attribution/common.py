"""Shared Part 2 helpers: paths, splits, description building, prompts, manifests.

Part 1 artifacts are read, never written. Descriptions for the pilot and
held-out splits are rebuilt with the Part 1 rules (frames, wordings, items,
rendering); rebuilding the study split reproduces part1_behavioral's
descriptions.csv exactly, which the tests check.
"""

import csv
import functools
import gzip
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

from geoflip import study as S
from geoflip.materials import BV_VALUES, PR_VALUES
from geoflip.render import render
from geoflip.utils import read_json, require, write_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # interpretability/
PART1 = HERE.parent / "part1_behavioral"
CONFIGS = HERE / "configs"
OUTPUTS = HERE / "outputs"
WORK = Path(
    os.environ.get("CE2_WORK", "/fs/clip-scratch/pramu/country_effect/part2/work")
)
MODEL_KEY = "llama3.2-3b"
SEED = 20261121
N_BOOT = 2000
LEVELS = 4
BV_TIME_INDICES = (0, 3, 6, 9)  # Part 1 build.py
SPLIT_START = {"pilot": 0, "study": 8, "heldout": 12}


# ---------------------------------------------------------------- inputs


def part1():
    """Part 1 descriptions, names (slot, real, fictional, tokens) and prompts."""
    with (PART1 / "dataset" / "descriptions.csv").open(encoding="utf-8") as f:
        descriptions = list(csv.DictReader(f))
    names = read_json(PART1 / "inputs" / "names.json")
    tasks = read_json(PART1 / "inputs" / "prompts.json")
    return descriptions, names, tasks


def part1_scores(boundary=None):
    """{(description_id, name): (d_A, d_B, s)} from labels.csv.gz."""
    out = {}
    with gzip.open(PART1 / "dataset" / "labels.csv.gz", "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if boundary and row["boundary"] != boundary:
                continue
            out[(row["description_id"], row["name"])] = (
                float(row["d_A"]),
                float(row["d_B"]),
                float(row["s"]),
                row["kind"],
            )
    return out


@functools.lru_cache(maxsize=1)
def panel():
    return read_json(CONFIGS / "mechanistic_panel.json")


def controls():
    path = OUTPUTS / "01_controls" / "controls.json"
    require(path.exists(), "Run scripts/01_build_controls.py first")
    return read_json(path)


def display_name(names, cid):
    """Rendered form of a Part 1 name id ("United States" -> "the United States")."""
    if cid in names["real"]:
        return names["real"][cid]
    if cid in names["fictional"]:
        return names["fictional"][cid]
    raise KeyError(cid)


# ---------------------------------------------------------------- splits


def contexts(boundary, split):
    """Context rows for a split (PREREGISTRATION.md § 1)."""
    start = SPLIT_START[split]
    rows = []
    if split == "study":
        # Part 1 build.py: 4 x 4 designs over indices 8-11.
        for i, j in product(range(LEVELS), range(LEVELS)):
            if boundary == "PR":
                values = {
                    "actors": PR_VALUES["actors"][start + i],
                    "grievance": PR_VALUES["grievance"][start + j],
                    "place": PR_VALUES["place"][start + (i + j) % LEVELS],
                    "time": "",
                }
            else:
                values = {
                    "actors": "",
                    "grievance": "",
                    "place": BV_VALUES["place"][start + i],
                    "time": BV_VALUES["time"][BV_TIME_INDICES[j]],
                }
            rows.append(
                {"context_id": f"{boundary}_c{i}{j}", "half": (i + j) % 2, **values}
            )
        return rows
    for i in range(8):
        if boundary == "PR":
            values = {
                "actors": PR_VALUES["actors"][start + i],
                "grievance": PR_VALUES["grievance"][start + (3 * i) % 8],
                "place": PR_VALUES["place"][start + (5 * i) % 8],
                "time": "",
            }
        else:
            values = {
                "actors": "",
                "grievance": "",
                "place": BV_VALUES["place"][start + i],
                "time": BV_VALUES["time"][BV_TIME_INDICES[i % 4]],
            }
        tag = "p" if split == "pilot" else "h"
        rows.append({"context_id": f"{boundary}_{tag}{i}", "half": i % 2, **values})
    return rows


def build_descriptions(boundary, split):
    """Description rows with the Part 1 columns, for one boundary and split."""
    spec = S.load_inputs()["descriptions"]["boundaries"][boundary]
    rows = []
    for (eid, entry), fid, wording, c in product(
        spec["evidence"].items(), spec["frames"], (1, 2), contexts(boundary, split)
    ):
        base = {k: c[k] for k in ("actors", "grievance", "place", "time") if c[k]}
        sentence = entry["wordings"][wording - 1]
        template = render(spec["frames"][fid], base, sentence, "{COUNTRY}")
        n_hi, n_lo = sum(entry["n_higher"]), sum(entry["n_lower"])
        if entry["role"] == "anchor_lower":
            p_higher = min(entry["share_higher"])
        elif entry["role"] == "anchor_higher":
            p_higher = max(entry["share_higher"])
        else:
            p_higher = entry["share_higher"][0]
        rows.append(
            {
                "description_id": f"{boundary}|{eid}|{fid}|W{wording}|{c['context_id']}",
                "boundary": boundary,
                "item": eid,
                "role": entry["role"],
                "acled_acts": "+".join(entry["acled_acts"]),
                "acled_n": n_hi + n_lo,
                "acled_p_higher": round(p_higher, 4),
                "frame": fid,
                "wording": f"W{wording}",
                "context_id": c["context_id"],
                "actors": c["actors"],
                "grievance": c["grievance"],
                "place": c["place"],
                "time": c["time"],
                "half": c["half"],
                "evidence_sentence": sentence,
                "template": template,
                "text_none": template.replace("{COUNTRY}", ""),
                "split": split,
            }
        )
    return rows


def panel_descriptions(boundary, contexts_tags=None):
    """Study-split panel descriptions in a fixed order (item, frame, wording, context)."""
    cfg = panel()
    tags = contexts_tags or cfg["contexts"]
    descriptions, _, _ = part1()
    by_id = {d["description_id"]: d for d in descriptions}
    out = []
    for item, fid, wording, tag in product(
        cfg["boundaries"][boundary]["items"], ("f1", "f2"), ("W1", "W2"), tags
    ):
        did = f"{boundary}|{item}|{fid}|{wording}|{boundary}_{tag}"
        require(did in by_id, f"Panel description {did} not in Part 1")
        out.append(by_id[did])
    return out


def ev_group(boundary, item, wording):
    return panel()["boundaries"][boundary]["ev_group"][item][wording]


def text_for(d, names, display):
    phrase = "" if display is None else names["slot"].format(country=display)
    return d["template"].replace("{COUNTRY}", phrase)


# ---------------------------------------------------------------- model


def load_runner(device="cuda"):
    from geoflip import scoring
    from geoflip.hooks import Runner

    spec = dict(scoring.MODELS[MODEL_KEY])
    tokenizer, model, revision = scoring.load_model(spec, device)
    require(revision == spec["revision"], "Model revision differs from the spec")
    cfg = model.config
    require(
        cfg.num_hidden_layers == 28 and cfg.hidden_size == 3072,
        "Unexpected model shape",
    )
    return Runner(model, tokenizer, spec, device), revision


def resolve_prompt(runner, d, names, display, task, n_tokens=None):
    from geoflip.spans import resolve

    return resolve(
        runner.tokenizer,
        runner.spec,
        task["system_prompt"],
        d["template"],
        names["slot"],
        display,
        d["evidence_sentence"],
        ev_group(d["boundary"], d["item"], d["wording"]),
        n_tokens,
    )


def require_parity():
    path = OUTPUTS / "00_parity" / "manifest.json"
    require(path.exists(), "Run scripts/00_verify_parity.py first")
    require(read_json(path).get("all_passed") is True, "Parity checks did not pass")


# ---------------------------------------------------------------- manifests


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_manifest(out_dir, script, inputs, outputs, extra=None):
    """manifest.json: code and input hashes, environment, outputs, parameters."""
    code = [
        Path(script),
        HERE / "common.py",
        ROOT / "geoflip" / "scoring.py",
        ROOT / "geoflip" / "spans.py",
        ROOT / "geoflip" / "hooks.py",
        ROOT / "geoflip" / "attribution.py",
        ROOT / "geoflip" / "render.py",
        ROOT / "geoflip" / "study.py",
        CONFIGS / "mechanistic_panel.json",
        CONFIGS / "stimuli_controls.json",
    ]
    env = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch
        import transformers

        env.update(
            torch=torch.__version__,
            transformers=transformers.__version__,
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        )
    except ImportError:
        pass

    def rel(p):
        p = Path(p).resolve()
        return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)

    manifest = {
        "script": rel(script),
        "written": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "git_commit": git_commit(),
        "environment": env,
        "code": {rel(p): sha256(p) for p in code if Path(p).exists()},
        "inputs": {rel(p): sha256(p) for p in inputs},
        "outputs": {rel(p): sha256(p) for p in outputs},
        **(extra or {}),
    }
    write_json(Path(out_dir) / "manifest.json", manifest)
    return manifest


class JsonlStore:
    """Append-only results keyed by a string, so long runs resume."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self.rows[row["key"]] = row
        self.stream = self.path.open("a", encoding="utf-8")

    def __contains__(self, key):
        return key in self.rows

    def add(self, row):
        self.rows[row["key"]] = row
        self.stream.write(json.dumps(row) + "\n")
        self.stream.flush()


# ---------------------------------------------------------------- panel pairs


def country_group(boundary, cid):
    groups = panel()["boundaries"][boundary]["countries"]
    return next(g for g, members in groups.items() if cid in members)


def item_group(boundary, item):
    groups = panel()["boundaries"][boundary]["item_groups"]
    return next(g for g, members in groups.items() if item in members)


def panel_pairs(boundary, countries=None, contexts_tags=None):
    """Token-length-matched (description, country, baseline) pairs: a country
    of n tokens is paired with baseline (description index mod 3) of n tokens
    (PREREGISTRATION.md § 5)."""
    _, names, _ = part1()
    cfg = panel()["boundaries"][boundary]
    baselines = controls()["baselines"]
    chosen = countries or [c for members in cfg["countries"].values() for c in members]
    all_desc = panel_descriptions(boundary)
    index = {d["description_id"]: i for i, d in enumerate(all_desc)}
    out = []
    for d in panel_descriptions(boundary, contexts_tags):
        i = index[d["description_id"]]
        for cid in chosen:
            display = display_name(names, cid)
            n = int(names["tokens"][display])
            base = baselines[str(n)][i % len(baselines[str(n)])]
            out.append(
                {
                    "d": d,
                    "country": cid,
                    "display": display,
                    "baseline": base,
                    "tokens": n,
                    "country_group": country_group(boundary, cid),
                    "item_group": item_group(boundary, d["item"]),
                }
            )
    return out
