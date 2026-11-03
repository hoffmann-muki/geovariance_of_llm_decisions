"""A bounded development/test exercise for E5 classification, never main inference."""

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

from . import inference, model_inputs, prepare_data
from .finalize_inputs import similarity_hits
from .utils import (
    append_jsonl,
    digest,
    file_digest,
    load_secret,
    read_csv,
    read_json,
    require,
    verified_snapshot,
    write_csv,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
CLASSES = list(model_inputs.CLASSES)
SHORT_INSTRUCTION = "Given a conflict-event description, retrieve the event category that best describes its focal action."
METHODS = ["baseline", "short_definitions", "example_centroids"]
DEFINITIONS = {
    "B": "Battles: organized armed parties engage in direct armed conflict with one another.",
    "P": "Protests: participants demonstrate peacefully, including when security forces intervene against them.",
    "R": "Riots: demonstrators or mobs use violence, damage property, or erect destructive barricades.",
    "E": "Explosions/Remote violence: bombs, mines, shelling, air strikes or drone strikes cause explosive or remotely delivered violence.",
    "V": "Violence against civilians: organized armed actors directly attack unarmed civilians or defenseless captives.",
    "S": "Strategic developments: politically relevant nonviolent agreements, surrender, recruitment, force movements or security changes occur.",
}
EXAMPLES = {
    "B": [
        "Soldiers and an organized rebel unit exchanged gunfire during a ground clash.",
        "Two rival armed militias fought with rifles over control of a checkpoint.",
        "An organized armed group fired on a rival armed patrol, which withdrew from the encounter.",
    ],
    "P": [
        "Residents peacefully rallied outside a government office to demand better public services.",
        "Workers held a nonviolent demonstration; police dispersed them with batons while the participants remained peaceful.",
        "Students staged a peaceful sit-in on a road to oppose a government policy.",
    ],
    "R": [
        "Demonstrators smashed windows and threw stones at police during a protest.",
        "A violent mob burned shops and destroyed parked vehicles.",
        "Demonstrators set tires alight and erected burning barricades across a road.",
    ],
    "E": [
        "An armed group fired a mortar shell into a residential neighborhood.",
        "A remotely detonated roadside bomb struck a passenger vehicle.",
        "A military aircraft dropped a bomb on a warehouse.",
    ],
    "V": [
        "Militia members shot unarmed farmers working in a field.",
        "Soldiers executed defenseless prisoners held in their custody.",
        "An organized gang abducted and beat an unarmed shopkeeper.",
    ],
    "S": [
        "An armed faction signed a ceasefire agreement with the government.",
        "Members of an armed organization voluntarily surrendered their weapons to authorities.",
        "An armed organization announced recruitment of new members.",
    ],
}
FIELDS = [
    "input_id",
    "kind",
    "split",
    "label",
    "source_event_id",
    "source_family_id",
    "source_file",
    "source_record",
    "country",
    "text",
    "text_sha256",
]
BATCH_SIZE = 24
BUDGET = 0.05


def reference_rows():
    original = read_json(inference.MAIN / "protocol.json")
    rows = []
    for label in CLASSES:
        for method, texts in (
            ("baseline", [original["class_definitions"][label]]),
            ("short_definitions", [DEFINITIONS[label]]),
            ("example_centroids", EXAMPLES[label]),
        ):
            for index, text in enumerate(texts):
                rows.append(
                    {
                        "input_id": f"reference:{method}:{label}:{index}",
                        "kind": method,
                        "split": "reference",
                        "label": label,
                        "text": text,
                        "text_sha256": digest(text),
                    }
                )
    return rows


def draft_review(directory, supplement_from=None):
    """Seeded selection followed by semantic review; no model predictions are used."""
    require(not directory.exists(), "Review directory already exists")
    processed = verified_snapshot(ROOT / "data/processed")
    verified_snapshot(ROOT / "data/accepted")
    verified_snapshot(inference.MAIN)
    family = {}
    for pair in read_csv(ROOT / "data/processed/related_event_pairs.csv"):
        for key in ("event_id_1", "event_id_2"):
            family[pair[key]] = pair["event_group_id"]
    allocations = [
        r
        for name in ("candidates.csv", "pilot.csv", "reserves.csv")
        for r in read_csv(ROOT / "data/processed" / name)
    ]
    main = read_csv(inference.MAIN / "stimuli.csv")
    excluded = read_csv(ROOT / "data/reviews/pilot_exclusions.csv")
    blocked = {r["event_group_id"] for r in allocations} | {
        r["source_family_id"] for r in main + excluded
    }
    blocked_ids = {r["event_id_cnty"] for r in allocations} | {
        r["source_event_id"] for r in main + excluded
    }
    invalid = {
        r["event_id_cnty"] for r in read_csv(ROOT / "data/processed/exclusions.csv")
    }
    previous = read_csv(supplement_from) if supplement_from else []
    previous_events = [r for r in previous if r["kind"] == "event"]
    raw, protected = [], [
        {"event_id_cnty": r["input_id"], "notes": r["text"]} for r in previous_events
    ]
    for item in processed["inputs"]:
        path = ROOT / "conflict_data" / item["file"]
        require(file_digest(path) == item["sha256"], "Raw source changed")
        for index, row in enumerate(read_csv(path), 1):
            row.update(source_file=path.name, source_record=str(index))
            if row["event_id_cnty"] in blocked_ids:
                protected.append(row)
            country = {"United States of America": "United States"}.get(
                row["country"], row["country"]
            )
            group = family.get(
                row["event_id_cnty"], "group_" + digest(row["event_id_cnty"])[:16]
            )
            if (
                group in blocked
                or row["event_id_cnty"] in invalid
                or country not in prepare_data.DEFAULT_CONFIG["country_regions"]
            ):
                continue
            label = prepare_data.LABELS.get(row["event_type"])
            if supplement_from and (
                (label == "V" and row["actor1"].casefold().startswith("unidentified"))
                or (
                    label == "S"
                    and row["sub_event_type"]
                    not in {
                        "Agreement",
                        "Change to group/activity",
                        "Non-violent transfer of territory",
                    }
                )
            ):
                continue
            primary = {
                "B": "Armed clash",
                "P": "Peaceful protest",
                "R": "Violent demonstration",
                "V": "Attack",
            }
            if (
                label not in CLASSES
                or (label in primary and row["sub_event_type"] != primary[label])
                or not 25 <= len(row["notes"].split()) <= 100
            ):
                continue
            row.update(label=label, country=country, event_group_id=group)
            raw.append(row)
    chosen, skipped, used = [], [], {r["source_family_id"] for r in previous_events}
    for label in (["V", "S"] if supplement_from else CLASSES):
        candidates = sorted(
            (r for r in raw if r["label"] == label),
            key=lambda r: digest("e5-validation-42/" + r["event_id_cnty"]),
        )
        count = 0
        for row in candidates:
            if row["event_group_id"] in used:
                continue
            _, hits = similarity_hits([row], protected + chosen, within=False)
            if hits:
                skipped.append(
                    {
                        "event_id": row["event_id_cnty"],
                        "reason": "near_duplicate",
                        "hits": hits,
                    }
                )
                continue
            chosen.append(row)
            used.add(row["event_group_id"])
            count += 1
            if count == 15:
                break
        require(count == 15, f"Insufficient independent candidates for {label}")
    rows = [
        {
            "input_id": r["event_id_cnty"],
            "kind": "event",
            "split": "candidate",
            "label": r["label"],
            "source_event_id": r["event_id_cnty"],
            "source_family_id": r["event_group_id"],
            "source_file": r["source_file"],
            "source_record": r["source_record"],
            "country": r["country"],
            "text": r["notes"],
            "text_sha256": digest(r["notes"]),
        }
        for r in chosen
    ] + ([] if supplement_from else reference_rows())
    directory.mkdir(parents=True)
    write_csv(directory / "candidates.csv", FIELDS, rows)
    write_json(
        directory / "blind.json",
        [
            {"input_id": r["input_id"], "text": r["text"]}
            for r in rows
            if r["kind"] == "event"
        ],
    )
    write_json(
        directory / "selection_audit.json",
        {
            "seed": 42,
            "candidates_per_class": 15,
            "supplement_from_sha256": (
                file_digest(supplement_from) if supplement_from else None
            ),
            "supplement_rule": (
                "After label-only review shortfalls, supplement V from named source actor1 (excluding unidentified actors) and S from Agreement, Change to group/activity or Non-violent transfer of territory; retain original seed, quotas and semantic review"
                if supplement_from
                else None
            ),
            "blocked_family_count": len(blocked),
            "source_note_comparisons": "Both SequenceMatcher directions >=0.90 against allocated source originals and previously chosen candidates",
            "skipped": skipped,
        },
    )
    return rows


def metadata():
    records = inference.get_json("https://api.deepinfra.com/models/list")
    matches = [
        r
        for r in records
        if r["model_name"] == model_inputs.ENCODER and not r.get("deprecated")
    ]
    require(
        len(matches) == 1 and matches[0]["max_tokens"] == 512,
        "E5 provider unavailable/context changed",
    )
    model = matches[0]
    require(model["pricing"]["type"] == "input_tokens", "Unexpected E5 pricing")
    return {
        "api": "deepinfra",
        "endpoint": inference.DEEPINFRA,
        "model": model_inputs.ENCODER,
        "dimension": 1024,
        "input_price_usd": float(model["pricing"]["cents_per_input_token"]) / 100,
        "output_price_usd": 0,
        "quantization": model.get("quantization"),
        "served_weight_revision": None,
    }


def validate_sources(rows):
    """Recheck raw provenance and family/text separation at release and execution."""
    processed = verified_snapshot(ROOT / "data/processed")
    verified_snapshot(ROOT / "data/accepted")
    verified_snapshot(inference.MAIN)
    events = [r for r in rows if r["kind"] == "event"]
    require(
        len({r["input_id"] for r in rows}) == len(rows), "Duplicate validation input ID"
    )
    require(
        len({r["source_family_id"] for r in events}) == len(events),
        "Validation families are not independent",
    )
    allocations = [
        r
        for name in ("candidates.csv", "pilot.csv", "reserves.csv")
        for r in read_csv(ROOT / "data/processed" / name)
    ]
    main = read_csv(inference.MAIN / "stimuli.csv")
    exclusions = read_csv(ROOT / "data/reviews/pilot_exclusions.csv")
    blocked = {r["event_group_id"] for r in allocations} | {
        r["source_family_id"] for r in main + exclusions
    }
    blocked_ids = {r["event_id_cnty"] for r in allocations} | {
        r["source_event_id"] for r in main + exclusions
    }
    require(
        not blocked & {r["source_family_id"] for r in events},
        "Validation overlaps allocated families",
    )
    require(
        not blocked_ids & {r["source_event_id"] for r in events},
        "Validation overlaps allocated event IDs",
    )
    family = {}
    for pair in read_csv(ROOT / "data/processed/related_event_pairs.csv"):
        for name in ("event_id_1", "event_id_2"):
            family[pair[name]] = pair["event_group_id"]
    needed = {r["source_event_id"] for r in events}
    raw_by_id, protected = {}, []
    for item in processed["inputs"]:
        path = ROOT / "conflict_data" / item["file"]
        require(file_digest(path) == item["sha256"], "Raw source changed")
        for ordinal, raw in enumerate(read_csv(path), 1):
            ident = raw["event_id_cnty"]
            if ident in blocked_ids:
                protected.append(raw)
            if ident in needed:
                require(ident not in raw_by_id, "Duplicate raw event ID")
                raw_by_id[ident] = {
                    **raw,
                    "source_file": path.name,
                    "source_record": str(ordinal),
                }
    invalid = {
        r["event_id_cnty"] for r in read_csv(ROOT / "data/processed/exclusions.csv")
    }
    require(not needed & invalid, "Validation includes an excluded source record")
    for row in events:
        raw = raw_by_id[row["source_event_id"]]
        country = {"United States of America": "United States"}.get(
            raw["country"], raw["country"]
        )
        require(
            row["input_id"] == row["source_event_id"]
            and row["source_file"] == raw["source_file"]
            and row["source_record"] == raw["source_record"]
            and row["country"] == country
            and row["text"] == raw["notes"]
            and row["label"] == prepare_data.LABELS[raw["event_type"]]
            and row["source_family_id"]
            == family.get(row["input_id"], "group_" + digest(row["input_id"])[:16]),
            "Validation source identity/text/label changed",
        )
    source_notes = [
        {"event_id_cnty": r["input_id"], "notes": r["text"]} for r in events
    ]
    checked, hits = similarity_hits(source_notes, protected)
    require(not hits, "Validation has near-duplicate protected or split source notes")
    references = {r["input_id"]: r for r in reference_rows()}
    actual = [r for r in rows if r["kind"] != "event"]
    require({r["input_id"] for r in actual} == set(references), "Reference set changed")
    for row in actual:
        require(
            all(
                row[key] == value for key, value in references[row["input_id"]].items()
            ),
            "Reference definition/example changed",
        )
    require(
        all(r["text_sha256"] == digest(r["text"]) for r in rows),
        "Input text fingerprint changed",
    )
    return {
        "independent_event_families": len(events),
        "protected_source_comparisons_including_within_validation": checked,
        "near_duplicate_hits": 0,
    }


def validate_reviews(rows, reviews):
    contexts = reviews["reviews"]
    require(
        len(contexts) == len({r["reviewer"] for r in contexts}) == 2,
        "Need two separate review contexts",
    )
    decisions = {r["input_id"]: r for r in reviews["adjudications"]}
    require(len(decisions) == len(reviews["adjudications"]), "Duplicate adjudication")
    assessments = [
        {r["input_id"]: r for r in context["assessments"]} for context in contexts
    ]
    for mapping, context in zip(assessments, contexts):
        require(
            len(mapping) == len(context["assessments"])
            and set(mapping) == set(decisions),
            "Incomplete/duplicate reviewer judgments",
        )
    for row in rows:
        decision = decisions[row["input_id"]]
        require(
            decision["text_sha256"] == row["text_sha256"],
            "Adjudication fingerprint changed",
        )
        for mapping in assessments:
            review = mapping[row["input_id"]]
            require(
                review["text_sha256"] == row["text_sha256"]
                and bool(review["evidence"])
                and review["evidence"] in row["text"],
                "Review lacks matching literal evidence",
            )
        if decision["decision"] == "accept":
            require(
                all(
                    mapping[row["input_id"]]["decision"] == "accept"
                    and mapping[row["input_id"]]["predicted_label"] == row["label"]
                    for mapping in assessments
                ),
                "Release requires concordant review acceptance and source label",
            )


def prepare(directory, review_dir):
    require(not directory.exists(), "Use a new validation input directory")
    rows = read_csv(review_dir / "candidates.csv")
    reviews = read_json(review_dir / "reviews.json")
    separation = validate_sources(rows)
    validate_reviews(rows, reviews)
    decisions = {r["input_id"]: r for r in reviews["adjudications"]}
    require(set(decisions) == {r["input_id"] for r in rows}, "Incomplete adjudication")
    for row in rows:
        require(row["text_sha256"] == digest(row["text"]), "Review text changed")
        judgment = decisions[row["input_id"]]
        require(
            judgment["text_sha256"] == row["text_sha256"],
            "Adjudication fingerprint changed",
        )
        if row["kind"] != "event":
            require(
                judgment["decision"] == "accept",
                "A reference requires correction and re-review",
            )
    for label in CLASSES:
        accepted = [
            r
            for r in rows
            if r["kind"] == "event"
            and r["label"] == label
            and decisions[r["input_id"]]["decision"] == "accept"
        ]
        require(
            len(accepted) >= 10,
            f"Fewer than ten reviewed independent events for {label}",
        )
        # Candidate order is frozen before review. Use first ten accepted; split by a separate seed.
        selected = sorted(
            accepted[:10], key=lambda r: digest("e5-split-42/" + r["input_id"])
        )
        for index, row in enumerate(selected):
            row["split"] = "development" if index < 5 else "test"
    rows = [r for r in rows if r["split"] in {"development", "test", "reference"}]
    counter, tokenizer = model_inputs.load_token_counter()
    require(
        tokenizer == read_json(inference.MAIN / "protocol.json")["tokenizer"],
        "Tokenizer changed",
    )
    baseline_instruction = read_json(inference.MAIN / "protocol.json")[
        "encoder_instruction"
    ]
    tokens = {}
    for row in rows:
        instructions = (
            (baseline_instruction, SHORT_INSTRUCTION)
            if row["kind"] == "event"
            else (None,)
        )
        for instruction in instructions:
            text = format_input(row["text"], instruction)
            tokens[digest(text)] = counter(text)
            require(
                0 < tokens[digest(text)] <= 512, "Validation input exceeds E5 context"
            )
    config = {
        "scope": "e5_classifier_validation_only",
        "main_inference_performed": False,
        "models": {model_inputs.ENCODER: metadata()},
        "class_order": CLASSES,
        "methods": METHODS,
        "instructions": {"baseline": baseline_instruction, "short": SHORT_INSTRUCTION},
        "tokenizer": tokenizer,
        "tokens": tokens,
        "development_n": 30,
        "test_n": 30,
        "source_separation": separation,
        "selection": "Highest development macro-F1, then accuracy, then simpler method in method-list order; only this winner and baseline evaluated on test",
        "readiness_guardrail": {
            "test_correct_at_least": 24,
            "test_correct_per_class_at_least": 3,
            "improvement_over_baseline_at_least": 3,
            "interpretation": "Prespecified practical screening thresholds, not a power calculation or proof of geographic invariance",
        },
        "budget_usd": BUDGET,
        "batch_size": BATCH_SIZE,
        "references": [
            "https://huggingface.co/intfloat/multilingual-e5-large-instruct",
            "https://github.com/microsoft/unilm/blob/master/e5/utils.py",
        ],
        "inputs": {
            "main_manifest_sha256": file_digest(inference.MAIN / "manifest.json"),
            "main_protocol_sha256": file_digest(inference.MAIN / "protocol.json"),
            "validation_code_sha256": file_digest(Path(__file__)),
            "inference_code_sha256": file_digest(Path(inference.__file__)),
            "utilities_sha256": file_digest(ROOT / "src/utils.py"),
            "candidates_sha256": file_digest(review_dir / "candidates.csv"),
        },
    }
    directory.mkdir(parents=True)
    write_csv(directory / "inputs.csv", FIELDS, rows)
    write_json(directory / "reviews.json", reviews)
    write_json(
        directory / "selection_audit.json",
        read_json(review_dir / "selection_audit.json"),
    )
    config["outputs"] = {
        p.name: {"sha256": file_digest(p)} for p in directory.iterdir()
    }
    write_json(directory / "manifest.json", config)
    return config


def format_input(text, instruction):
    return f"Instruct: {instruction}\nQuery: {text}" if instruction else text


def metrics(predictions):
    n = len(predictions)
    correct = sum(r["label"] == r["prediction"] for r in predictions)
    recalls, f1s = {}, []
    for label in CLASSES:
        tp = sum(r["label"] == label and r["prediction"] == label for r in predictions)
        actual = sum(r["label"] == label for r in predictions)
        predicted = sum(r["prediction"] == label for r in predictions)
        recalls[label] = {"correct": tp, "n": actual}
        f1s.append(2 * tp / (actual + predicted) if actual + predicted else 0)
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else 0,
        "macro_f1": sum(f1s) / len(CLASSES),
        "by_class": recalls,
    }


def select_method(scores):
    require(set(scores) == set(METHODS), "Incomplete development methods")
    return max(
        METHODS,
        key=lambda m: (scores[m]["macro_f1"], scores[m]["accuracy"], -METHODS.index(m)),
    )


def class_vectors(rows, vectors, method):
    result = []
    for label in CLASSES:
        references = [
            vectors[digest(r["text"])]
            for r in rows
            if r["kind"] == method and r["label"] == label
        ]
        require(
            len(references) == (3 if method == "example_centroids" else 1),
            "Invalid class references",
        )
        centroid = [math.fsum(values) / len(references) for values in zip(*references)]
        result.append(inference.normalise(centroid))
    return result


def predict(rows, vectors, config, method, split):
    prototypes = class_vectors(rows, vectors, method)
    instruction = config["instructions"][
        "baseline" if method == "baseline" else "short"
    ]
    predictions = []
    for row in rows:
        if row["split"] != split:
            continue
        vector = vectors[digest(format_input(row["text"], instruction))]
        scores = [
            math.fsum(a * b for a, b in zip(vector, prototype))
            for prototype in prototypes
        ]
        prediction = CLASSES[max(range(6), key=scores.__getitem__)]
        predictions.append(
            {
                "input_id": row["input_id"],
                "split": split,
                "method": method,
                "label": row["label"],
                "prediction": prediction,
                "scores": dict(zip(CLASSES, scores)),
            }
        )
    return predictions


def batch_requests(rows, config, phase, selected=None):
    batch_size = config["batch_size"]
    require(
        type(batch_size) is int and 0 < batch_size <= BATCH_SIZE,
        "Invalid validation batch size",
    )
    texts = {}
    if phase == "development":
        texts.update(
            (digest(r["text"]), r["text"]) for r in rows if r["split"] == "reference"
        )
    methods = (
        METHODS
        if phase == "development"
        else list(dict.fromkeys(["baseline", selected]))
    )
    for method in methods:
        instruction = config["instructions"][
            "baseline" if method == "baseline" else "short"
        ]
        for row in rows:
            if row["split"] == phase:
                text = format_input(row["text"], instruction)
                texts[digest(text)] = text
    ordered = sorted(texts.items())
    return [ordered[i : i + batch_size] for i in range(0, len(ordered), batch_size)]


def fetch_batch(path, ident, batch, config, key):
    model = config["models"][model_inputs.ENCODER]
    body = {
        "model": model["model"],
        "input": [text for _, text in batch],
        "encoding_format": "float",
    }
    fingerprint = inference.object_hash(body)
    events, complete = inference.journal_state(path)
    if ident in complete:
        previous = complete[ident]
        require(
            previous["payload_sha256"] == fingerprint and previous["status"] == "valid",
            "Saved validation request changed or failed",
        )
        response = json.loads(previous["raw_response"])
    else:
        require(
            not any(e.get("status") == "fatal" for e in events),
            "A failed request requires reconciliation",
        )
        require(len(complete) < 12, "Validation request limit reached")
        reserve = inference.cost_bound(model, body)
        budget = config["budget_usd"]
        require(
            type(budget) in (int, float)
            and math.isfinite(budget)
            and 0 < budget <= BUDGET,
            "Invalid validation budget",
        )
        require(
            sum(e.get("accounted_cost_usd", 0) for e in events) + reserve <= budget,
            "Validation budget reached",
        )
        common = {
            "request_id": ident,
            "attempt": 1,
            "model": model["model"],
            "payload_sha256": fingerprint,
        }
        append_jsonl(path, {**common, "event": "started", "time": inference.now()})
        started = time.monotonic()
        result = {
            **common,
            "event": "finished",
            "time": inference.now(),
            "terminal": True,
            "status": "fatal",
            "accounted_cost_usd": reserve,
        }
        try:
            raw, _ = inference.send(model, body, key)
            result["raw_response"] = raw.replace(key, "[REDACTED]")
            response = json.loads(raw)
            inference.embeddings(
                response,
                model,
                len(batch),
                sum(config["tokens"][sha] for sha, _ in batch),
            )
            cost, basis = inference.response_cost(response, model)
            result.update(status="valid", accounted_cost_usd=cost, cost_basis=basis)
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            IndexError,
            AttributeError,
        ) as error:
            result["error"] = (
                f"{type(error).__name__}: {str(error).replace(key, '[REDACTED]')}"
            )
            result.update(
                time=inference.now(), elapsed_seconds=time.monotonic() - started
            )
            append_jsonl(path, result)
            raise ValueError("Validation request failed; no automatic retry") from None
        result.update(time=inference.now(), elapsed_seconds=time.monotonic() - started)
        append_jsonl(path, result)
    embeddings = inference.embeddings(
        response, model, len(batch), sum(config["tokens"][sha] for sha, _ in batch)
    )
    return dict(zip((sha for sha, _ in batch), embeddings))


def guardrail_passed(candidate, baseline, thresholds):
    keys = (
        "test_correct_at_least",
        "test_correct_per_class_at_least",
        "improvement_over_baseline_at_least",
    )
    require(
        all(type(thresholds[key]) is int and thresholds[key] >= 0 for key in keys),
        "Invalid screening thresholds",
    )
    return (
        candidate["correct"] >= thresholds["test_correct_at_least"]
        and min(row["correct"] for row in candidate["by_class"].values())
        >= thresholds["test_correct_per_class_at_least"]
        and candidate["correct"] - baseline["correct"]
        >= thresholds["improvement_over_baseline_at_least"]
    )


def render_report(input_dir, output_dir, run):
    config = read_json(input_dir / "manifest.json")
    selection = read_json(output_dir / "selection.json")
    scores = read_json(output_dir / "test_metrics.json")
    predictions = read_json(output_dir / "test_predictions.json")
    selected = run["selected_method"]
    lines = [
        "# E5 Classification Validation",
        "",
        f"{config['development_n']} development and {config['test_n']} untouched test events. "
        "No main, reserve or former pilot source families were submitted. Only E5 was called.",
        "",
        "| Method | Development correct | Development macro-F1 |",
        "| --- | --- | --- |",
    ]
    for method in METHODS:
        score = selection["scores"][method]
        lines.append(
            f"| {method} | {score['correct']}/{score['n']} | {score['macro_f1']:.3f} |"
        )
    lines += [
        "",
        f"Selected before test requests: **{selected}**, using development macro-F1, then accuracy, then method-list order. Test predictions were produced only for that method and the baseline.",
        "",
        "| Method | Test correct | Test macro-F1 |",
        "| --- | --- | --- |",
    ]
    for method, score in scores.items():
        lines.append(
            f"| {method} | {score['correct']}/{score['n']} | {score['macro_f1']:.3f} |"
        )
    lines += [
        "",
        "| Class | Baseline correct | Selected correct |",
        "| --- | --- | --- |",
    ]
    for label in CLASSES:
        baseline_class = scores["baseline"]["by_class"][label]
        selected_class = scores[selected]["by_class"][label]
        lines.append(
            f"| {label} | {baseline_class['correct']}/{baseline_class['n']} | "
            f"{selected_class['correct']}/{selected_class['n']} |"
        )
    baseline = {r["input_id"]: r for r in predictions if r["method"] == "baseline"}
    corrected = [r for r in predictions if r["method"] == selected]
    fixed = sum(
        r["prediction"] == r["label"]
        and baseline[r["input_id"]]["prediction"] != r["label"]
        for r in corrected
    )
    introduced = sum(
        r["prediction"] != r["label"]
        and baseline[r["input_id"]]["prediction"] == r["label"]
        for r in corrected
    )
    _, complete = inference.journal_state(output_dir / "attempts.jsonl")
    cost = math.fsum(r["accounted_cost_usd"] for r in complete.values())
    thresholds = config["readiness_guardrail"]
    lines += [
        "",
        f"Paired test changes: {fixed} baseline errors corrected; "
        f"{introduced} previously correct labels changed incorrectly. "
        f"The prespecified practical guardrail **{'passed' if run['guardrail_passed'] else 'did not pass'}**: "
        f"at least {thresholds['test_correct_at_least']} correct overall, "
        f"{thresholds['test_correct_per_class_at_least']} correct in each class, and "
        f"{thresholds['improvement_over_baseline_at_least']} more correct than baseline.",
        "",
        f"{len(complete)} embedding POSTs; estimated DeepInfra token charges ${cost:.6f}. Raw vectors, response indices, exact input-token counts, payload hashes and timings are retained in the journal. Selection is checksummed before test.",
        "",
        "The baseline uses the existing instruction and six definitions. Short definitions use one task sentence and concise class descriptions. Example centroids average three normalized, country-neutral example vectors per class and normalize each mean. Both corrections use the same short instruction; predictions select the largest cosine similarity with the frozen class-order tie rule. No model weights were trained.",
        "",
        "Two separate adversarial AI contexts reviewed source-label support and reference semantics; one source-note review was blind to expected labels. Unclear notes were withheld before splitting. Additional V and S candidates were drawn after review shortfalls, using named-source-actor and nonviolent-subtype filters respectively. These curated 25–100-word notes are not an operational or country-balanced benchmark. Small class-specific test samples give limited precision, and the bundled changes do not isolate which component caused any gain.",
        "",
        "No method or reference was changed after development or test predictions. This check concerns basic classification utility, not geographic invariance. The main study, its stimuli and frozen protocol remain unchanged; adopting a correction requires a separately versioned protocol and fresh length checks before main inference.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase(input_dir, output_dir, phase):
    require(phase in {"development", "test"}, "Invalid validation phase")
    config = verified_snapshot(input_dir)
    require(
        config["scope"] == "e5_classifier_validation_only",
        "Only E5 validation is executable",
    )
    for name, path in (
        ("validation_code_sha256", Path(__file__)),
        ("inference_code_sha256", Path(inference.__file__)),
        ("utilities_sha256", ROOT / "src/utils.py"),
        ("main_manifest_sha256", inference.MAIN / "manifest.json"),
        ("main_protocol_sha256", inference.MAIN / "protocol.json"),
    ):
        require(
            config["inputs"][name] == file_digest(path),
            "Validation input/code binding changed",
        )
    rows = read_csv(input_dir / "inputs.csv")
    validate_sources(rows)
    validate_reviews(rows, read_json(input_dir / "reviews.json"))
    require(
        Counter((r["split"], r["label"]) for r in rows if r["kind"] == "event")
        == Counter(
            {
                (split, label): 5
                for split in ("development", "test")
                for label in CLASSES
            }
        ),
        "Invalid balanced validation split",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / ".active"
    lock.mkdir()
    try:
        manifest_path = output_dir / "manifest.json"
        binding = file_digest(input_dir / "manifest.json")
        run = (
            read_json(manifest_path)
            if manifest_path.exists()
            else {"input_manifest_sha256": binding, "main_inference_performed": False}
        )
        require(
            run["input_manifest_sha256"] == binding,
            "Validation run configuration changed",
        )
        if manifest_path.exists():
            # Completed files are immutable. Only a running phase may extend its journal.
            for name, info in run.get("outputs", {}).items():
                path = output_dir / name
                if name == "attempts.jsonl" and run.get("phase_running"):
                    data = path.read_bytes()
                    require(
                        digest(data[: run["committed_journal_bytes"]].decode("utf-8"))
                        == info["sha256"],
                        "Completed journal prefix changed",
                    )
                else:
                    require(
                        file_digest(path) == info["sha256"],
                        "Completed validation artifact changed",
                    )
        else:
            require(
                not any(p != lock for p in output_dir.iterdir()),
                "Use an empty output directory",
            )
        if run.get(phase + "_complete"):
            return run
        require(
            not run.get("phase_running") or run["phase_running"] == phase,
            "Resume the interrupted phase before switching phases",
        )
        require(
            phase == "development" or run.get("development_complete"),
            "Select a method on development before test",
        )
        require(
            metadata() == config["models"][model_inputs.ENCODER],
            "E5 provider metadata changed",
        )
        key = load_secret("DEEPINFRA_API_KEY", ROOT / ".env")
        if phase == "test":
            require(
                file_digest(output_dir / "selection.json") == run["selection_sha256"],
                "Development selection changed",
            )
            selection = read_json(output_dir / "selection.json")
            require(
                run["selected_method"]
                == selection["selected_method"]
                == select_method(selection["scores"]),
                "Stored development winner changed",
            )
        run["phase_running"] = phase
        write_json(manifest_path, run)
        selected = run.get("selected_method")
        vectors = {}
        for index, batch in enumerate(batch_requests(rows, config, "development")):
            if phase == "test":
                # Recover development vectors locally; never resend development during test.
                _, complete = inference.journal_state(output_dir / "attempts.jsonl")
                finish = complete[f"development:{index}"]
                values = inference.embeddings(
                    json.loads(finish["raw_response"]),
                    config["models"][model_inputs.ENCODER],
                    len(batch),
                    sum(config["tokens"][sha] for sha, _ in batch),
                )
                vectors.update(zip((sha for sha, _ in batch), values))
            else:
                vectors.update(
                    fetch_batch(
                        output_dir / "attempts.jsonl",
                        f"development:{index}",
                        batch,
                        config,
                        key,
                    )
                )
        if phase == "test":
            for index, batch in enumerate(
                batch_requests(rows, config, "test", selected)
            ):
                vectors.update(
                    fetch_batch(
                        output_dir / "attempts.jsonl",
                        f"test:{index}",
                        batch,
                        config,
                        key,
                    )
                )
        methods = (
            METHODS
            if phase == "development"
            else list(dict.fromkeys(["baseline", selected]))
        )
        predictions = [
            prediction
            for method in methods
            for prediction in predict(rows, vectors, config, method, phase)
        ]
        scores = {
            method: metrics([r for r in predictions if r["method"] == method])
            for method in methods
        }
        write_json(output_dir / (phase + "_predictions.json"), predictions)
        if phase == "development":
            selected = select_method(scores)
            write_json(
                output_dir / "selection.json",
                {
                    "selected_method": selected,
                    "scores": scores,
                    "test_accessed": False,
                    "rule": config["selection"],
                },
            )
            run.update(
                selected_method=selected,
                selection_sha256=file_digest(output_dir / "selection.json"),
            )
        else:
            write_json(output_dir / "test_metrics.json", scores)
            candidate = scores[selected]
            run["guardrail_passed"] = guardrail_passed(
                candidate, scores["baseline"], config["readiness_guardrail"]
            )
        run[phase + "_complete"] = True
        run[phase + "_completed_at"] = inference.now()
        run.pop("phase_running", None)
        if phase == "test":
            render_report(input_dir, output_dir, run)
        run["committed_journal_bytes"] = (output_dir / "attempts.jsonl").stat().st_size
        run["outputs"] = {
            p.name: {"sha256": file_digest(p)}
            for p in output_dir.iterdir()
            if p.is_file() and p != manifest_path
        }
        write_json(manifest_path, run)
        return run
    finally:
        lock.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    draft = sub.add_parser("draft-review")
    draft.add_argument("--output-dir", type=Path, required=True)
    draft.add_argument("--supplement-from", type=Path)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--review-dir", type=Path, required=True)
    prepare_parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/e5_validation"
    )
    for phase in ("development", "test"):
        run = sub.add_parser(phase)
        run.add_argument("--input-dir", type=Path, default=ROOT / "data/e5_validation")
        run.add_argument("--output-dir", type=Path, default=ROOT / "runs/e5_validation")
    args = parser.parse_args()
    if args.command == "draft-review":
        draft_review(args.output_dir, args.supplement_from)
    elif args.command == "prepare":
        prepare(args.output_dir, args.review_dir)
    else:
        result = run_phase(args.input_dir, args.output_dir, args.command)
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in ("selected_method", "guardrail_passed")
                    if k in result
                }
            )
        )


if __name__ == "__main__":
    main()
