"""The v4 study: one pre-specified pipeline over a fully crossed population.

  check   V0 static rules on the inputs (PLAN.md § 3.2), and the task prompts
  pilot   layout choice, anchors, first-token validity, fp32 audit, throughput (pilot bases)
  score   every cell of the main population, scored once (needs a frozen prereg)
  filter  the single flip rule and its placebo null (PLAN.md § 6)
  export  dataset/<version>/

Answers are single tokens: the options are listed as "A. <class>" and
"B. <class>", and a prompt's score is read from the first answer position.
One prompt per forward pass, no batching.
"""

import re
from itertools import product
from pathlib import Path

from .pipeline import country_conditions, country_phrase, frame_fields, render
from .utils import read_csv, read_json, require

ROOT = Path(__file__).resolve().parent.parent
INPUTS = ROOT / "inputs" / "v4"
BOUNDARIES = ("PR", "BV")
VERSIONS = ("defined", "undefined")
ORDERS = ("A", "B")  # A: order_a[0] is option A; B: reversed
LETTERS = ("A", "B")

# ---------------------------------------------------------------- V0 rules (PLAN.md § 3.2)

MIN_WORDS, MAX_WORDS = 5, 16
HEDGES = {
    "unclear",
    "uncertain",
    "reportedly",
    "allegedly",
    "possibly",
    "perhaps",
    "maybe",
    "may",
    "might",
    "could",
    "apparently",
    "seemingly",
    "likely",
    "unconfirmed",
    "rumored",
    "rumoured",
    "suspected",
    "appeared",
    "seemed",
    "probably",
}
OUTCOME_STEMS = (
    "kill",
    "dead",
    "died",
    "death",
    "injur",
    "wound",
    "casualt",
    "fatal",
    "arrest",
    "detain",
    "hospitali",
)
IDENTITY_STEMS = (
    "police",
    "army",
    "military",
    "soldier",
    "troop",
    "ethnic",
    "relig",
    "tribe",
    "tribal",
    "party",
    "parties",
)
STOPWORDS = set(
    """a an the and or of in on at to for from by with without into onto over under near as is are was were be
    been being that which who whom whose this these those it its their them they each other others both either
    any all some few one two even if such example including not no can cannot able unable who's than then""".split()
)
# Role terms the frames and evidence need; removed before the definition-word check.
ROLE_PHRASES = (
    "the armed group",
    "an armed group",
    "armed group",
    "participants",
    "demonstrators",
    "demonstrate",
    "members",
    "member",
    "groups",
    "group",
)


def words(text):
    return re.findall(r"[A-Za-z][A-Za-z'-]*", text)


def stem(word):
    word = word.lower()
    return word[:5] if len(word) >= 5 else word


def definition_stems(task_prompts):
    """Stems of every content word in the class names and all definition paraphrases."""
    stems = set()
    for spec in task_prompts["boundaries"].values():
        texts = list(spec["order_a"])
        for para in spec["paraphrases"]:
            texts += list(para["definitions"].values())
        for text in texts:
            stems |= {stem(w) for w in words(text) if w.lower() not in STOPWORDS}
    return stems


def strip_roles(text):
    text = text.lower()
    for phrase in ROLE_PHRASES:
        text = re.sub(rf"\b{re.escape(phrase)}\b", " ", text)
    return text


def text_findings(label, text, banned, kind):
    """Rules 2–5 on one evidence sentence ("evidence") or base field / frame ("field")."""
    out = []
    tokens = words(text)
    lowered = [w.lower() for w in tokens]
    if kind == "evidence":
        if not MIN_WORDS <= len(tokens) <= MAX_WORDS:
            out.append(
                f"{label}: {len(tokens)} words (allowed {MIN_WORDS}-{MAX_WORDS})"
            )
        if not text.endswith(".") or re.search(r"[.!?;]", text[:-1]):
            out.append(f"{label}: must be exactly one sentence")
        out += [f"{label}: hedge {w!r}" for w in lowered if w in HEDGES]
        out += [
            f"{label}: outcome word {w!r}"
            for w in lowered
            if w.startswith(OUTCOME_STEMS)
        ]
        capitals = [w for w in tokens[1:] if w[0].isupper()]
    else:
        capitals = [w for w in tokens if w[0].isupper()] if kind == "field" else []
    out += [f"{label}: capitalised word {w!r} (named entity?)" for w in capitals]
    out += [
        f"{label}: identity or state-force word {w!r}"
        for w in lowered
        if w.startswith(IDENTITY_STEMS)
    ]
    for w in words(strip_roles(text)):
        if w.lower() not in STOPWORDS and stem(w) in banned:
            out.append(
                f"{label}: {w!r} shares its stem with a class name or definition"
            )
    return out


def static_findings(inputs):
    """Every V0 problem, as readable strings (empty when clean)."""
    findings = []
    banned = definition_stems(inputs["task_prompts"])
    seen = set()
    for boundary in BOUNDARIES:
        spec = inputs["descriptions"]["boundaries"][boundary]
        roles = [e["role"] for e in spec["evidence"].values()]
        for role in ("anchor_lower", "anchor_higher"):
            if roles.count(role) != 1:
                findings.append(f"{boundary}: needs exactly one {role}")
        for evidence_id, entry in spec["evidence"].items():
            wordings = entry["wordings"]
            if len(wordings) != 2 or wordings[0] == wordings[1]:
                findings.append(
                    f"{boundary}/{evidence_id}: needs two different wordings"
                )
            for index, text in enumerate(wordings, start=1):
                findings += text_findings(
                    f"{boundary}/{evidence_id}/W{index}", text, banned, "evidence"
                )
                if text in seen:
                    findings.append(
                        f"{boundary}/{evidence_id}/W{index}: wording reused"
                    )
                seen.add(text)
        for frame_id, frame in spec["frames"].items():
            fields = frame_fields(frame)
            if fields.count("country") != 1 or fields.count("evidence") != 1:
                findings.append(
                    f"{boundary}/{frame_id}: needs exactly one {{country}} and {{evidence}}"
                )
            if "{place}{country}" not in frame:
                findings.append(
                    f"{boundary}/{frame_id}: the country slot must follow {{place}}"
                )
            findings += text_findings(
                f"{boundary}/{frame_id}",
                re.sub(r"\{\w+\}", " ", frame),
                banned,
                "frame",
            )
        splits = inputs["bases"][boundary]
        ids = {
            split: {row["base_id"] for row in rows} for split, rows in splits.items()
        }
        if ids["pilot"] & ids["main"]:
            findings.append(f"{boundary}: pilot and main bases overlap")
        for split, rows in splits.items():
            for row in rows:
                for key, value in row.items():
                    if key != "base_id":
                        findings += text_findings(
                            f"{boundary}/{row['base_id']}.{key}", value, banned, "field"
                        )
        findings += slot_findings(boundary, spec, splits["main"][0], inputs["names"])
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
                "place",
                "unclear",
                "uncertain",
            ):
                if re.search(rf"\b{word}\b", text):
                    findings.append(f"task {boundary}/p{index}: mentions {word!r}")
    return findings


def slot_findings(boundary, spec, base, names):
    """The country phrase must be the only difference from the no-country text."""
    findings = []
    evidence = next(iter(spec["evidence"].values()))["wordings"][0]
    for frame_id, frame in spec["frames"].items():
        plain = render(frame, base, evidence, "")
        for cid, display, _ in country_conditions(names, ("real", "fictional")):
            phrase = country_phrase(names, display)
            text = render(frame, base, evidence, phrase)
            if text.count(display) != 1 or text.replace(phrase, "", 1) != plain:
                findings.append(f"{boundary}/{frame_id}: slot error for {cid}")
    return findings


# ---------------------------------------------------------------- inputs


def load_inputs(directory=INPUTS):
    inputs = {
        "descriptions": read_json(directory / "descriptions.json"),
        "task_prompts": read_json(directory / "task_prompts.json"),
        "names": read_json(directory / "names.json"),
        "bases": {
            b: {
                split: read_csv(directory / "bases" / f"{b}_{split}.csv")
                for split in ("pilot", "main")
            }
            for b in BOUNDARIES
        },
        "dir": str(directory),
    }
    return inputs


def render_tasks(prompts, layout):
    """Every system prompt: boundary x version x paraphrase x order, for one layout.

    Order A lists order_a[0] as option A; order B swaps the letters. The
    definitions follow the option order. Layout "options_last" puts the
    option lines after the answer instruction; "options_first" before it."""
    tasks = []
    for boundary in BOUNDARIES:
        spec = prompts["boundaries"][boundary]
        lower, higher = spec["lower"], spec["higher"]
        for version, (index, para), order in product(
            VERSIONS, enumerate(spec["paraphrases"]), ORDERS
        ):
            first, second = spec["order_a"][:: 1 if order == "A" else -1]
            letter = {first: "A", second: "B"}
            parts = [para["intro"]]
            if version == "defined":
                parts += [para["definitions"][first], para["definitions"][second]]
            options = f"A. {first}\nB. {second}"
            answer = para["answer"]
            parts += (
                [options, answer] if layout == "options_first" else [answer, options]
            )
            tasks.append(
                {
                    "task_id": f"{boundary}:{version}:p{index}:{order}",
                    "boundary": boundary,
                    "version": version,
                    "paraphrase": index,
                    "order": order,
                    "layout": layout,
                    "system_prompt": "\n\n".join(parts),
                    "lower": lower,
                    "higher": higher,
                    "letter_lower": letter[lower],
                    "letter_higher": letter[higher],
                }
            )
    return tasks


# ---------------------------------------------------------------- scoring


class Scorer:
    """scoring.score_first_token, one prompt per forward pass, with a JSONL cache."""

    def __init__(self, model_key, work, device="cuda", dtype=None):
        import hashlib  # noqa: F401  (used in key)
        import json

        from . import scoring as S

        self.S = S
        self.spec = dict(S.MODELS[model_key])
        if dtype:
            self.spec["dtype"] = dtype
        self.tokenizer, self.model, self.revision = S.load_model(self.spec, device)
        self.model_key, self.device = model_key, device
        name = (
            f"{model_key}_{self.spec['dtype']}_{self.revision[:10]}_first_token.jsonl"
        )
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
    def key(system_prompt, text):
        import hashlib
        import json

        payload = ["first_token_v1", system_prompt, text, list(LETTERS)]
        return hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()

    def fresh(self, system_prompt, text):
        return self.S.score_first_token(
            self.model,
            self.tokenizer,
            self.spec,
            system_prompt,
            text,
            LETTERS,
            self.device,
        )

    def score(self, pairs, progress_every=2000):
        """pairs: [(system_prompt, text)] -> cache records, in order."""
        import json
        import sys
        import time

        keys = [self.key(s, t) for s, t in pairs]
        pending = {}
        for key, pair in zip(keys, pairs):
            if key not in self.cache:
                pending.setdefault(key, pair)
        started = time.monotonic()
        with self.cache_path.open("a", encoding="utf-8") as stream:
            for count, (key, (system_prompt, text)) in enumerate(
                pending.items(), start=1
            ):
                result = self.fresh(system_prompt, text)
                record = {
                    "key": key,
                    "lp": result["lp"],
                    "mass": result["mass"],
                    "top": result["top"],
                }
                self.cache[key] = record
                stream.write(json.dumps(record) + "\n")
                if count % progress_every == 0:
                    stream.flush()
                    rate = count / (time.monotonic() - started)
                    print(
                        f"  {count}/{len(pending)} at {rate:.1f}/s",
                        file=sys.stderr,
                        flush=True,
                    )
        self.last_rate = (
            len(pending) / max(time.monotonic() - started, 1e-9) if pending else None
        )
        return [self.cache[key] for key in keys]


# ---------------------------------------------------------------- stimuli


def stimuli(
    inputs,
    boundary,
    split,
    names_kinds=("none", "real", "fictional"),
    evidence_ids=None,
):
    """Every description of one boundary and base split, under each name condition."""
    spec = inputs["descriptions"]["boundaries"][boundary]
    names = country_conditions(inputs["names"], names_kinds)
    out = []
    evidence_ids = evidence_ids or list(spec["evidence"])
    for frame_id, evidence_id, base, wording, (cid, display, kind) in product(
        spec["frames"], evidence_ids, inputs["bases"][boundary][split], (1, 2), names
    ):
        entry = spec["evidence"][evidence_id]
        sentence = entry["wordings"][wording - 1]
        frame = spec["frames"][frame_id]
        out.append(
            {
                "boundary": boundary,
                "frame": frame_id,
                "evidence": evidence_id,
                "role": entry["role"],
                "base_id": base["base_id"],
                "wording": wording,
                "name": cid,
                "name_kind": kind,
                "text": render(
                    frame, base, sentence, country_phrase(inputs["names"], display)
                ),
            }
        )
    return out


def scored(scorer, stims, tasks):
    """One row per stimulus x task: d = log P(higher letter) - log P(lower letter)."""
    by_boundary = {}
    for task in tasks:
        by_boundary.setdefault(task["boundary"], []).append(task)
    pairs, meta = [], []
    for s in stims:
        for t in by_boundary[s["boundary"]]:
            pairs.append((t["system_prompt"], s["text"]))
            meta.append((s, t))
    rows = []
    for (s, t), record in zip(meta, scorer.score(pairs)):
        lp = record["lp"]
        rows.append(
            {
                **{
                    k: s[k]
                    for k in (
                        "boundary",
                        "frame",
                        "evidence",
                        "role",
                        "base_id",
                        "wording",
                        "name",
                        "name_kind",
                    )
                },
                **{k: t[k] for k in ("version", "paraphrase", "order", "layout")},
                "d": lp[t["letter_higher"]] - lp[t["letter_lower"]],
                "mass": record["mass"],
                "top1": record["top"][0][0],
            }
        )
    return rows


# ---------------------------------------------------------------- pilot (PLAN.md § 5)

ANCHOR_ACCURACY = 0.9  # per concrete prompt and frame, both ends
MASS_P05 = 0.9  # 5th percentile of P(A) + P(B) per concrete prompt
TOP1_LETTER = 0.99  # share of pilot prompts whose most likely first token is A or B
FP32_SAMPLE = 200
FP32_TOLERANCE = 0.05  # max |d(bf16) - d(fp32)| for bf16 to be used in the main run


def quantile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * (len(values) - 1) + 0.5))]


def pilot_summary(rows):
    """Gates per layout and the layout choice, from no-country pilot rows."""
    import statistics

    out = {"layouts": {}}
    layouts = sorted({r["layout"] for r in rows})
    for layout in layouts:
        mine = [r for r in rows if r["layout"] == layout]
        prompts = {}
        for r in mine:
            prompts.setdefault(
                (r["boundary"], r["version"], r["paraphrase"], r["order"]), []
            ).append(r)
        gates = []
        for key, prs in sorted(prompts.items()):
            entry = {
                "prompt": ":".join(map(str, key)),
                "mass_p05": quantile([r["mass"] for r in prs], 0.05),
            }
            for frame in sorted({r["frame"] for r in prs}):
                for role, sign in (("anchor_lower", -1), ("anchor_higher", 1)):
                    vals = [
                        r["d"] * sign > 0
                        for r in prs
                        if r["frame"] == frame and r["role"] == role
                    ]
                    entry[f"{frame}_{role}"] = statistics.fmean(vals)
            entry["pass"] = entry["mass_p05"] >= MASS_P05 and all(
                v >= ANCHOR_ACCURACY
                for k, v in entry.items()
                if k.endswith(("anchor_lower", "anchor_higher"))
            )
            gates.append(entry)
        pairs = {}
        for r in mine:
            pairs.setdefault(
                (
                    r["boundary"],
                    r["frame"],
                    r["evidence"],
                    r["base_id"],
                    r["wording"],
                    r["version"],
                    r["paraphrase"],
                ),
                {},
            )[r["order"]] = r["d"]
        order_effects = [abs(v["A"] - v["B"]) for v in pairs.values() if len(v) == 2]
        out["layouts"][layout] = {
            "gates": gates,
            "all_pass": all(g["pass"] for g in gates),
            "mean_abs_order_effect": statistics.fmean(order_effects),
            "top1_is_letter": statistics.fmean(r["top1"] in LETTERS for r in mine),
            "n_prompts": len(mine),
        }
        info = out["layouts"][layout]
        info["all_pass"] = info["all_pass"] and info["top1_is_letter"] >= TOP1_LETTER
    passing = [l for l in layouts if out["layouts"][l]["all_pass"]]
    out["chosen_layout"] = (
        min(passing, key=lambda l: out["layouts"][l]["mean_abs_order_effect"])
        if passing
        else None
    )
    return out


def stage_pilot(inputs, scorer):
    rows = []
    for layout in read_json(Path(inputs["dir"]) / "task_prompts.json")["layouts"]:
        tasks = render_tasks(inputs["task_prompts"], layout)
        for boundary in BOUNDARIES:
            rows += scored(
                scorer,
                stimuli(inputs, boundary, "pilot", ("none",)),
                [t for t in tasks if t["boundary"] == boundary],
            )
    summary = pilot_summary(rows)
    summary["throughput_prompts_per_s"] = scorer.last_rate
    return summary, rows


def stage_fp32_audit(inputs, scorer_bf16_cache, scorer_fp32, layout, seed=7):
    """Re-score a fixed random sample of pilot prompts in fp32; compare d with the bf16 cache."""
    import json
    import random

    tasks = render_tasks(inputs["task_prompts"], layout)
    pairs = []
    for boundary in BOUNDARIES:
        for s in stimuli(inputs, boundary, "pilot", ("none",)):
            pairs += [(t, s) for t in tasks if t["boundary"] == boundary]
    sample = random.Random(seed).sample(pairs, FP32_SAMPLE)
    bf16 = {}
    with open(scorer_bf16_cache, encoding="utf-8") as stream:
        for line in stream:
            r = json.loads(line)
            bf16[r["key"]] = r
    deltas = []
    for t, s in sample:
        old = bf16[Scorer.key(t["system_prompt"], s["text"])]["lp"]
        new = scorer_fp32.fresh(t["system_prompt"], s["text"])["lp"]
        d_old = old[t["letter_higher"]] - old[t["letter_lower"]]
        d_new = new[t["letter_higher"]] - new[t["letter_lower"]]
        deltas.append(abs(d_old - d_new))
    return {
        "sample": len(deltas),
        "max_abs_delta_d": max(deltas),
        "median_abs_delta_d": quantile(deltas, 0.5),
        "use_bf16": max(deltas) <= FP32_TOLERANCE,
    }


def main(argv=None):
    import argparse
    import json
    import os
    import sys

    from .utils import write_json

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("stage", choices=("check", "pilot", "fp32_audit"))
    parser.add_argument("--inputs", type=Path, default=INPUTS)
    parser.add_argument(
        "--work",
        type=Path,
        default=Path(
            os.environ.get("GEOFLIP_WORK", "/fs/clip-scratch/pramu/geoflip_work")
        )
        / "v4",
    )
    parser.add_argument("--model", default="llama3.2-3b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layout", help="fp32_audit: the chosen layout")
    args = parser.parse_args(argv)
    inputs = load_inputs(args.inputs)
    findings = static_findings(inputs)
    if findings:
        print("V0 static checks failed:\n" + "\n".join(findings), file=sys.stderr)
        return 1
    args.work.mkdir(parents=True, exist_ok=True)
    if args.stage == "check":
        print("V0: no findings")
        return 0
    if args.stage == "pilot":
        scorer = Scorer(args.model, args.work, args.device)
        summary, rows = stage_pilot(inputs, scorer)
        summary["scorer"] = scorer.describe()
        write_json(args.work / "pilot.json", summary)
        with (args.work / "pilot_rows.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        print(json.dumps({k: v for k, v in summary.items() if k != "layouts"}))
        for layout, info in summary["layouts"].items():
            print(layout, {k: v for k, v in info.items() if k != "gates"})
        return 0
    if args.stage == "fp32_audit":
        scorer32 = Scorer(args.model, args.work, args.device, dtype="float32")
        from . import scoring as S

        rev = S.MODELS[args.model]["revision"][:10]
        cache = args.work / "cache" / f"{args.model}_bfloat16_{rev}_first_token.jsonl"
        result = stage_fp32_audit(inputs, cache, scorer32, args.layout)
        write_json(args.work / "fp32_audit.json", result)
        print(json.dumps(result))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
