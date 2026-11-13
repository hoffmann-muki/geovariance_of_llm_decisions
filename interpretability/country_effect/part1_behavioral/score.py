"""Part 1: score every description x name x option order (DESIGN.md § 4).

The reference scorer (geoflip.study.Scorer -> scoring.score_first_token): one
prompt per forward pass, no batching, a JSONL cache keyed by (system prompt,
text), so an interrupted run resumes where it stopped. Descriptions are scored
in a fixed shuffled order, all names and both orders of one description
together, so a partial cache covers the whole design evenly.

  python -m country_effect.part1_behavioral.score [--limit N] [--show ID]
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path

from geoflip import study as S
from geoflip.render import country_conditions, country_phrase

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("CE_WORK", "/fs/clip-scratch/pramu/country_effect/work"))
SEED = 20261107
CHUNK = 25  # descriptions per call to the scorer


def load():
    with (HERE / "dataset" / "descriptions.csv").open() as f:
        descriptions = list(csv.DictReader(f))
    names = json.loads((HERE / "inputs" / "names.json").read_text())
    tasks = json.loads((HERE / "inputs" / "prompts.json").read_text())
    return descriptions, names, tasks


def pairs_for(d, names, tasks):
    out = []
    for cid, display, kind in country_conditions(names, ("none", "real", "fictional")):
        text = d["template"].replace("{COUNTRY}", country_phrase(names, display))
        for order in S.ORDERS:
            t = tasks[f"{d['boundary']}:{order}"]
            out.append(((cid, kind, order, t), (t["system_prompt"], text)))
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="score only the first N shuffled descriptions")
    p.add_argument("--show", default=None, help="print every number for one description_id")
    args = p.parse_args(argv)
    descriptions, names, tasks = load()
    scorer = S.Scorer("llama3.2-3b", WORK)
    print(json.dumps(scorer.describe()), flush=True)

    if args.show:
        d = next(x for x in descriptions if x["description_id"] == args.show)
        items = pairs_for(d, names, tasks)
        for order in S.ORDERS:
            print(f"--- system prompt, order {order} ---\n{tasks[d['boundary'] + ':' + order]['system_prompt']}\n")
        print("--- user text (no country) ---\n" + d["text_none"] + "\n")
        recs = scorer.score([pair for _, pair in items])
        for ((cid, kind, order, t), _), r in zip(items, recs):
            dd = r["lp"][t["letter_higher"]] - r["lp"][t["letter_lower"]]
            print(f"{cid:18s} {kind:9s} order {order}: lpA={r['lp']['A']:+.2f} lpB={r['lp']['B']:+.2f} "
                  f"mass={r['mass']:.3f} top1={r['top'][0][0]!r} d={dd:+.2f} -> {t['higher'] if dd > 0 else t['lower']}")
        return

    order = list(range(len(descriptions)))
    random.Random(SEED).shuffle(order)
    if args.limit:
        order = order[: args.limit]
    for start in range(0, len(order), CHUNK):
        chunk = [descriptions[i] for i in order[start : start + CHUNK]]
        scorer.score([pair for d in chunk for _, pair in pairs_for(d, names, tasks)], progress_every=10**9)
        print(f"descriptions {min(start + CHUNK, len(order))}/{len(order)} "
              f"rate {scorer.last_rate or 0:.1f}/s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
