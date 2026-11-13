"""Part 1: build the factorial dataset of event descriptions (DESIGN.md § 3).

Reuses the v4 materials (ACLED-attested items, frames, wordings, the 37 real
countries, the defined task prompts) and adds a balanced design over the
context fields and 20 placebo names. Writes:

  inputs/contexts.csv      the context cells (PR: 4x4 Latin square; BV: 4x4 crossing)
  inputs/names.json        real + placebo names and the country slot
  inputs/prompts.json      the two system prompts (orders A and B) per boundary
  inputs/manifest.json     sha256 of every input this design was built from
  dataset/descriptions.csv one row per description (country slot left open)

  cd interpretability && python -m country_effect.part1_behavioral.build
"""

import csv
import hashlib
import json
import re
from itertools import product
from pathlib import Path

from geoflip import study as S
from geoflip.materials import BV_VALUES, PR_VALUES
from geoflip.render import country_conditions, country_phrase, render

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # interpretability/
V4 = ROOT / "inputs" / "v4"

LAYOUT = "options_first"
VERSION = "defined"
PARAPHRASE = {"PR": 0, "BV": 3}  # both passed every pilot anchor gate (DESIGN.md § 4)
LEVELS = 4
# Context values come from the v4 *main* pools (indices 8-19); the first four of
# each are used here, so indices 12-19 stay unseen for a held-out replication.
MAIN_START = 8
BV_TIME_INDICES = (0, 3, 6, 9)  # evenly spaced over the 12 generic times


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contexts():
    rows = []
    pr = {
        k: PR_VALUES[k][MAIN_START : MAIN_START + LEVELS]
        for k in ("actors", "grievance", "place")
    }
    # Latin square: every actor x grievance pair once; place = (i + j) mod 4, so
    # every actor x place and grievance x place pair also occurs exactly once.
    for i, j in product(range(LEVELS), range(LEVELS)):
        rows.append(
            {
                "boundary": "PR",
                "context_id": f"PR_c{i}{j}",
                "actors": pr["actors"][i],
                "grievance": pr["grievance"][j],
                "place": pr["place"][(i + j) % LEVELS],
                "time": "",
                "half": (i + j) % 2,  # split-half for replication
            }
        )
    places = BV_VALUES["place"][MAIN_START : MAIN_START + LEVELS]
    times = [BV_VALUES["time"][k] for k in BV_TIME_INDICES]
    for i, j in product(range(LEVELS), range(LEVELS)):
        rows.append(
            {
                "boundary": "BV",
                "context_id": f"BV_c{i}{j}",
                "actors": "",
                "grievance": "",
                "place": places[i],
                "time": times[j],
                "half": (i + j) % 2,
            }
        )
    return rows


def names():
    real = S.read_json(V4 / "names.json")
    placebos = S.read_json(HERE / "inputs" / "placebos.json")
    return {
        "slot": real["slot"],
        "real": real["real"],
        "fictional": {n: n for n in placebos["placebos"]},
        "tokens": {**placebos["real_tokens"], **{n: v["tokens"] for n, v in placebos["placebos"].items()}},
    }


def prompts(task_prompts):
    out = {}
    for t in S.render_tasks(task_prompts, LAYOUT):
        if t["version"] == VERSION and t["paraphrase"] == PARAPHRASE[t["boundary"]]:
            out[f"{t['boundary']}:{t['order']}"] = t
    return out


def check(descriptions, ctx, nm, tasks):
    """V0 rules on the new parts: context fields, the country slot, the prompts."""
    banned = S.definition_stems(tasks)
    findings = []
    for row in ctx:
        for key in ("actors", "grievance", "place", "time"):
            if row[key]:
                findings += S.text_findings(f"{row['context_id']}.{key}", row[key], banned, "field")
    for b in S.BOUNDARIES:
        spec = descriptions["boundaries"][b]
        base = next(
            {k: r[k] for k in ("actors", "grievance", "place", "time") if r[k]}
            for r in ctx
            if r["boundary"] == b
        )
        findings += S.slot_findings(b, spec, {"base_id": "x", **base}, nm)
    return findings


def main():
    inputs = S.load_inputs()
    desc = inputs["descriptions"]
    ctx = contexts()
    nm = names()
    tasks = prompts(inputs["task_prompts"])
    findings = check(desc, ctx, nm, inputs["task_prompts"]) + S.static_findings(inputs)
    if findings:
        raise SystemExit("V0 findings:\n" + "\n".join(findings))

    (HERE / "inputs" / "names.json").write_text(json.dumps(nm, indent=2) + "\n")
    (HERE / "inputs" / "prompts.json").write_text(json.dumps(tasks, indent=2) + "\n")
    with (HERE / "inputs" / "contexts.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ctx[0]))
        w.writeheader()
        w.writerows(ctx)

    rows = []
    for b in S.BOUNDARIES:
        spec = desc["boundaries"][b]
        for (eid, entry), fid, wording, c in product(
            spec["evidence"].items(),
            spec["frames"],
            (1, 2),
            [r for r in ctx if r["boundary"] == b],
        ):
            base = {k: c[k] for k in ("actors", "grievance", "place", "time") if c[k]}
            sentence = entry["wordings"][wording - 1]
            template = render(spec["frames"][fid], base, sentence, "{COUNTRY}")
            n_hi, n_lo = sum(entry["n_higher"]), sum(entry["n_lower"])
            rows.append(
                {
                    "description_id": f"{b}|{eid}|{fid}|W{wording}|{c['context_id']}",
                    "boundary": b,
                    "item": eid,
                    "role": entry["role"],
                    "acled_acts": "+".join(entry["acled_acts"]),
                    "acled_n": n_hi + n_lo,
                    "acled_p_higher": round(min(entry["share_higher"]) if entry["role"] == "anchor_lower" else (max(entry["share_higher"]) if entry["role"] == "anchor_higher" else entry["share_higher"][0]), 4),
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
                }
            )
    # Every rendered text must differ from text_none only by the country phrase.
    for r in rows:
        for cid, display, _ in country_conditions(nm, ("real", "fictional")):
            phrase = country_phrase(nm, display)
            text = r["template"].replace("{COUNTRY}", phrase)
            assert text.replace(phrase, "", 1) == r["text_none"], (r["description_id"], cid)
    with (HERE / "dataset" / "descriptions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    sources = {
        str(p.relative_to(ROOT)): sha256(p)
        for p in [
            V4 / "descriptions.json",
            V4 / "task_prompts.json",
            V4 / "names.json",
            V4 / "items.json",
            V4 / "acled_acts.csv",
            HERE / "inputs" / "placebos.json",
            ROOT / "geoflip" / "materials.py",
            ROOT / "geoflip" / "render.py",
            ROOT / "geoflip" / "scoring.py",
            ROOT / "geoflip" / "study.py",
            Path(__file__),
        ]
    }
    outputs = {
        str(p.relative_to(ROOT)): sha256(p)
        for p in [
            HERE / "inputs" / "contexts.csv",
            HERE / "inputs" / "names.json",
            HERE / "inputs" / "prompts.json",
            HERE / "dataset" / "descriptions.csv",
        ]
    }
    (HERE / "inputs" / "manifest.json").write_text(
        json.dumps({"sources": sources, "outputs": outputs}, indent=2) + "\n"
    )
    by = {}
    for r in rows:
        by[r["boundary"]] = by.get(r["boundary"], 0) + 1
    n_names = 1 + len(nm["real"]) + len(nm["fictional"])
    print("descriptions:", by, "names (incl. none):", n_names,
          "prompts:", sum(by.values()) * n_names * 2)


if __name__ == "__main__":
    main()
