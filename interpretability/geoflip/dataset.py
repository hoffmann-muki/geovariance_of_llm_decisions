"""Export the released dataset from one iteration's stage summaries.

  python -m geoflip.dataset --iteration final

reads <work>/<iteration>/{PR,BV}/{matrix,confirm,test}.json and writes dataset/:
descriptions.csv, cells.csv, system_prompts.csv, pairs.csv, flips.jsonl,
matrix_<B>_{labels,s}.csv and manifest.json (see dataset/README.md).
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from . import pipeline as P
from .utils import file_digest, read_json, require, write_csv, write_json

DATASET = P.ROOT / "dataset"


def label_name(boundary, label):
    return P.CLASS_NAMES[boundary][label]


def export(iteration, work=P.WORK, inputs_dir=P.GEO, out=DATASET):
    inputs = P.load_inputs(inputs_dir)
    require(not P.static_findings(inputs), "V0 static checks fail on the inputs")
    countries = P.country_conditions(inputs["countries"])
    tasks = P.render_tasks(inputs["task_prompts"], inputs["descriptions"])
    descriptions, cells, pairs, flips, meta = [], [], [], [], {}
    for boundary in P.BOUNDARIES:
        work_dir = Path(work) / iteration / boundary
        stage = {
            name: read_json(work_dir / f"{name}.json")
            for name in ("check", "confirm", "test", "matrix")
        }
        require(
            stage["check"]["meta"]["digests"] == inputs["digests"],
            f"{boundary}: the run's inputs differ from {inputs_dir}",
        )
        meta[boundary] = {k: stage["matrix"]["meta"][k] for k in ("scorer", "digests")}
        spec = inputs["descriptions"]["boundaries"][boundary]
        for split in ("confirm", "test"):
            for item in stage[split]["items"]:
                for p in item["pairs"]:
                    pairs.append(
                        {
                            "boundary": boundary,
                            "item": item["item"],
                            "version": item["version"],
                            "split": split,
                            "country_high": p["high"],
                            "country_low": p["low"],
                            "n_descriptions": p["n"],
                            "share_high_above_low": round(p["share_positive"], 4),
                            "mean_gap": round(p["mean_gap"], 4),
                            "wording_reference": round(item["wording_reference"], 4),
                            "systematic": p["systematic"],
                            "flips": p["flips"],
                        }
                    )
                flips += item["flips"]
        for index, row in enumerate(stage["matrix"]["rows"], start=1):
            row_id = f"{boundary}-{index:03d}"
            bases = inputs["bases"][boundary][row["split"]]
            base = next(b for b in bases if b["base_id"] == row["base_id"])
            frame = spec["frames"][row["frame"]]
            evidence = spec["evidence"][row["evidence"]]["wordings"][row["wording"] - 1]
            require(
                P.render(frame, base, evidence, " in [COUNTRY]") == row["template"],
                f"{row_id}: template does not re-render",
            )
            descriptions.append(
                {
                    "row_id": row_id,
                    "boundary": boundary,
                    "lower_class": spec["lower_class"],
                    "higher_class": spec["higher_class"],
                    "item": row["item"],
                    "frame": row["frame"],
                    "evidence": row["evidence"],
                    "version": row["version"],
                    "split": row["split"],
                    "base_id": row["base_id"],
                    "wording": row["wording"],
                    "evidence_sentence": evidence,
                    "template": row["template"],
                }
            )
            for cid, display, kind in countries:
                cell = {
                    "row_id": row_id,
                    "country": cid,
                    "country_kind": kind,
                    "text": P.render(
                        frame,
                        base,
                        evidence,
                        P.country_phrase(inputs["countries"], display),
                    ),
                    "s": round(row["s"][cid], 4),
                    "label": label_name(boundary, row["label"][cid]),
                }
                for k, value in enumerate(row["dbar"][cid]):
                    cell[f"dbar_p{k}"] = round(value, 4)
                cells.append(cell)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "descriptions.csv", list(descriptions[0]), descriptions)
    write_csv(out / "cells.csv", list(cells[0]), cells)
    write_csv(out / "pairs.csv", list(pairs[0]), pairs)
    write_csv(
        out / "system_prompts.csv",
        [
            "task_id",
            "boundary",
            "version",
            "paraphrase",
            "order",
            "lower",
            "higher",
            "system_prompt",
        ],
        tasks,
    )
    with (out / "flips.jsonl").open("w", encoding="utf-8") as stream:
        for f in flips:
            stream.write(json.dumps(f, ensure_ascii=False) + "\n")
    names = [cid for cid, _, _ in countries]
    for boundary in P.BOUNDARIES:
        mine = [d for d in descriptions if d["boundary"] == boundary]
        by_row = {}
        for c in cells:
            by_row.setdefault(c["row_id"], {})[c["country"]] = c
        for kind, field in (("labels", "label"), ("s", "s")):
            rows = [
                {
                    **{
                        k: d[k]
                        for k in (
                            "row_id",
                            "item",
                            "version",
                            "split",
                            "base_id",
                            "wording",
                            "template",
                        )
                    },
                    **{n: by_row[d["row_id"]][n][field] for n in names},
                }
                for d in mine
            ]
            write_csv(out / f"matrix_{boundary}_{kind}.csv", list(rows[0]), rows)
    files = sorted(p.name for p in out.iterdir() if p.suffix in (".csv", ".jsonl"))
    write_json(
        out / "manifest.json",
        {
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "iteration": iteration,
            "producer_sha256": {
                "pipeline.py": file_digest(Path(P.__file__)),
                "dataset.py": file_digest(Path(__file__)),
            },
            "runs": meta,
            "counts": {
                "descriptions": {
                    b: sum(d["boundary"] == b for d in descriptions)
                    for b in P.BOUNDARIES
                },
                "country_conditions": len(countries),
                "cells": len(cells),
                "flips": {
                    b: sum(f["boundary"] == b for f in flips) for b in P.BOUNDARIES
                },
            },
            "files": {name: file_digest(out / name) for name in files},
        },
    )
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--iteration", required=True)
    parser.add_argument("--work", type=Path, default=P.WORK)
    parser.add_argument("--inputs", type=Path, default=P.GEO)
    parser.add_argument("--out", type=Path, default=DATASET)
    args = parser.parse_args(argv)
    print(export(args.iteration, args.work, args.inputs, args.out))


if __name__ == "__main__":
    main()
