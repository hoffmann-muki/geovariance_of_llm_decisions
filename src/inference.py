"""Resumable model requests; the CLI executes only a six-event held-out check."""

import argparse
import json
import math
import random
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import model_inputs
from .utils import (
    append_jsonl,
    digest,
    file_digest,
    load_secret,
    read_csv,
    require,
    verified_snapshot,
    write_csv,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "data/variants/main"
OPENROUTER = "https://openrouter.ai/api/v1"
DEEPINFRA = "https://api.deepinfra.com/v1/openai/embeddings"
PROVIDER = "coreweave/bf16"
INPUT_FIELDS = [
    "input_id",
    "source_event_id",
    "source_family_id",
    "source_file",
    "source_record",
    "expected_label",
    "text",
    "text_sha256",
    "encoder_tokens",
]
ORDER_FIELDS = ["sequence", "model", "input_id", "replicate", "request_id"]


def now():
    return datetime.now(timezone.utc).isoformat()


def object_hash(value):
    return digest(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    )


def get_json(url, key=None):
    headers = {"Authorization": "Bearer " + key} if key else {}
    with urllib.request.build_opener(NoRedirect).open(
        urllib.request.Request(url, headers=headers), timeout=30
    ) as response:
        return json.load(response)


def model_metadata():
    catalog = {r["id"]: r for r in get_json(OPENROUTER + "/models")["data"]}
    model = catalog[model_inputs.DECODER]
    endpoints = get_json(OPENROUTER + f"/models/{model_inputs.DECODER}/endpoints")[
        "data"
    ]["endpoints"]
    matches = [r for r in endpoints if r["tag"] == PROVIDER and r["status"] == 0]
    require(len(matches) == 1, "Pinned Llama provider unavailable")
    decoder = matches[0]
    require(
        {"structured_outputs", "response_format", "temperature", "seed", "max_tokens"}
        <= set(decoder["supported_parameters"]),
        "Pinned Llama endpoint lacks required settings",
    )
    encoders = get_json("https://api.deepinfra.com/models/list")
    matches = [
        r
        for r in encoders
        if r["model_name"] == model_inputs.ENCODER and not r.get("deprecated")
    ]
    require(
        len(matches) == 1 and matches[0]["max_tokens"] == 512,
        "E5 endpoint unavailable or context changed",
    )
    encoder = matches[0]
    require(encoder["pricing"]["type"] == "input_tokens", "Unsupported E5 pricing")
    return {
        model_inputs.ENCODER: {
            "api": "deepinfra",
            "endpoint": DEEPINFRA,
            "model": model_inputs.ENCODER,
            "dimension": 1024,
            "input_price_usd": float(encoder["pricing"]["cents_per_input_token"]) / 100,
            "output_price_usd": 0,
            "quantization": encoder.get("quantization"),
            "served_weight_revision": None,
        },
        model_inputs.DECODER: {
            "api": "openrouter",
            "endpoint": OPENROUTER + "/chat/completions",
            "model": model_inputs.DECODER,
            "canonical_slug": model["canonical_slug"],
            "provider": PROVIDER,
            "provider_name": decoder["provider_name"],
            "quantization": decoder["quantization"],
            "input_price_usd": float(decoder["pricing"]["prompt"]),
            "output_price_usd": float(decoder["pricing"]["completion"]),
            "served_weight_revision": None,
        },
    }


def heldout_records():
    """Select one shortest original per class from previously excluded pilot families."""
    verified_snapshot(ROOT / "data/accepted")
    verified_snapshot(MAIN)
    excluded = {
        r["source_event_id"]: r
        for r in read_csv(ROOT / "data/reviews/pilot_exclusions.csv")
    }
    candidates = []
    for row in read_csv(ROOT / "data/accepted/pilot.csv"):
        candidates.append(
            {
                "input_id": row["event_id_cnty"],
                "source_event_id": row["event_id_cnty"],
                "source_family_id": row["event_group_id"],
                "source_file": row["source_file"],
                "source_record": row["source_record"],
                "expected_label": row["label"],
                "text": row["notes"],
            }
        )
    extras = set(excluded) - {r["input_id"] for r in candidates}
    mapping = {"Explosions/Remote violence": "E", "Strategic developments": "S"}
    for path in sorted((ROOT / "conflict_data").glob("*.csv")):
        for ordinal, row in enumerate(read_csv(path), 1):
            event = row["event_id_cnty"]
            if event in extras:
                source = excluded[event]
                require(
                    source["source_file"] == path.name
                    and source["source_record"] == str(ordinal)
                    and source["notes_sha256"] == digest(row["notes"]),
                    "Held-out source changed",
                )
                candidates.append(
                    {
                        "input_id": event,
                        "source_event_id": event,
                        "source_family_id": source["source_family_id"],
                        "source_file": path.name,
                        "source_record": str(ordinal),
                        "expected_label": mapping[row["event_type"]],
                        "text": row["notes"],
                    }
                )
    chosen = [
        min(
            (r for r in candidates if r["expected_label"] == label),
            key=lambda r: (len(r["text"].split()), r["input_id"]),
        )
        for label in model_inputs.CLASSES
    ]
    main = read_csv(MAIN / "stimuli.csv")
    require(
        not {r["source_family_id"] for r in main}
        & {r["source_family_id"] for r in chosen},
        "Compatibility inputs overlap main source families",
    )
    require(
        not {r["text_sha256"] for r in main} & {digest(r["text"]) for r in chosen},
        "Compatibility text appears in main stimuli",
    )
    require(
        len({r["input_id"] for r in chosen}) == 6, "Need six distinct held-out events"
    )
    for row in chosen:
        source = excluded[row["input_id"]]
        require(
            source["source_family_id"] == row["source_family_id"]
            and source["notes_sha256"] == digest(row["text"]),
            "Held-out exclusion binding changed",
        )
    return chosen


def prepare_check(output_dir, metadata=None, token_counter=None):
    require(not output_dir.exists(), "Compatibility input directory already exists")
    protocol = json.loads((MAIN / "protocol.json").read_text())
    require(
        protocol["encoder_model"] == model_inputs.ENCODER
        and protocol["decoder_model"] == model_inputs.DECODER
        and protocol["encoder_instruction"] == model_inputs.ENCODER_INSTRUCTION,
        "Prepared model protocol changed",
    )
    rows = heldout_records()
    counter, tokenizer = (
        model_inputs.load_token_counter() if token_counter is None else token_counter
    )
    require(
        tokenizer == protocol["tokenizer"], "Tokenizer differs from frozen main inputs"
    )
    for row in rows:
        row["text_sha256"] = digest(row["text"])
        row["encoder_tokens"] = counter(
            protocol["encoder_input_template"].format(
                instruction=protocol["encoder_instruction"], text=row["text"]
            )
        )
        require(0 < row["encoder_tokens"] <= 512, "Held-out input exceeds E5 context")
    require(
        {label: counter(text) for label, text in protocol["class_definitions"].items()}
        == protocol["prototype_tokens"],
        "Prototype token counts changed",
    )
    ordered = sorted(rows, key=lambda r: (r["encoder_tokens"], r["input_id"]))
    repeats = [
        ordered[index]["input_id"] for index in (0, len(ordered) // 2, len(ordered) - 1)
    ]
    schedule = []
    for model in (model_inputs.ENCODER, model_inputs.DECODER):
        for ident, replicate in [(r["input_id"], 0) for r in rows] + [
            (ident, 1) for ident in repeats
        ]:
            schedule.append(
                {
                    "model": model,
                    "input_id": ident,
                    "replicate": replicate,
                    "request_id": f"{model}:{ident}:replicate={replicate}",
                }
            )
    random.Random(42).shuffle(schedule)
    for index, row in enumerate(schedule, 1):
        row["sequence"] = index
    metadata = model_metadata() if metadata is None else metadata
    config = {
        "format_version": 1,
        "scope": "heldout_compatibility_only",
        "main_experiment_started": False,
        "protocol": protocol,
        "models": metadata,
        "repeat_input_ids": repeats,
        "selection_rule": "Shortest original per class by word count, with event-ID ties, from excluded pilot source families; repeats are shortest/median/longest E5 input lengths.",
        "planned_classifications": 18,
        "prototype_embeddings": 6,
        "planned_posts_before_retries": 19,
        "inputs": {
            "main_manifest_sha256": file_digest(MAIN / "manifest.json"),
            "exclusions_sha256": file_digest(
                ROOT / "data/reviews/pilot_exclusions.csv"
            ),
            "runner_sha256": file_digest(Path(__file__)),
            "model_code_sha256": file_digest(Path(model_inputs.__file__)),
            "utilities_sha256": file_digest(ROOT / "src/utils.py"),
        },
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".compatibility-", dir=output_dir.parent
    ) as directory:
        stage = Path(directory) / "export"
        stage.mkdir()
        write_csv(stage / "inputs.csv", INPUT_FIELDS, rows)
        write_csv(stage / "request_order.csv", ORDER_FIELDS, schedule)
        config["outputs"] = {
            p.name: {"sha256": file_digest(p)} for p in stage.iterdir()
        }
        write_json(stage / "manifest.json", config)
        stage.rename(output_dir)
    return config


def normalise(vector, dimension=1024):
    require(
        isinstance(vector, list) and len(vector) == dimension,
        "Invalid embedding dimension",
    )
    require(
        all(type(x) in (int, float) and math.isfinite(x) for x in vector),
        "Nonfinite/invalid embedding component",
    )
    norm = math.hypot(*vector)
    require(math.isfinite(norm) and norm > 0, "Zero/nonfinite embedding norm")
    return [x / norm for x in vector]


def embeddings(response, model, count, expected_tokens):
    require(
        isinstance(response, dict) and response.get("model") == model["model"],
        "Embedding model drift",
    )
    data = response.get("data")
    require(isinstance(data, list) and len(data) == count, "Wrong embedding count")
    require(
        all(isinstance(row, dict) and type(row.get("index")) is int for row in data),
        "Invalid embedding index",
    )
    require(
        {r["index"] for r in data} == set(range(count)),
        "Duplicate/missing embedding indices",
    )
    reported_tokens = (response.get("usage") or {}).get("prompt_tokens")
    require(
        type(reported_tokens) is int and reported_tokens == expected_tokens,
        f"E5 token count mismatch: local {expected_tokens}, provider {reported_tokens}",
    )
    return [
        normalise(row["embedding"], model["dimension"])
        for row in sorted(data, key=lambda row: row["index"])
    ]


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def decoder_result(response, model, class_order, cache_status):
    require(cache_status != "HIT", "Response cache hit")
    require(
        response.get("model") in {model["model"], model["canonical_slug"]},
        "Decoder model drift",
    )
    require(
        response.get("provider") == model["provider_name"], "Decoder provider drift"
    )
    require(
        isinstance(response.get("id"), str) and response["id"], "Missing generation ID"
    )
    choices = response["choices"]
    require(len(choices) == 1, "Wrong choice count")
    choice = choices[0]
    require(not choice["message"].get("refusal"), "Decoder refusal")
    require(choice["finish_reason"] == "stop", "Incomplete decoder output")
    parsed = json.loads(choice["message"]["content"], object_pairs_hook=unique_object)
    require(
        type(parsed) is dict
        and set(parsed) == {"label"}
        and parsed["label"] in class_order,
        "Invalid label schema",
    )
    return {"label": parsed["label"], "generation_id": response["id"]}


def payload(model, row, protocol):
    if model["api"] == "deepinfra":
        text = protocol["encoder_input_template"].format(
            instruction=protocol["encoder_instruction"], text=row["text"]
        )
        return {
            "model": model["model"],
            "input": [text],
            **protocol["encoder_parameters"],
        }
    return {
        "model": model["model"],
        "messages": [
            {"role": "system", "content": protocol["decoder_system_prompt"]},
            {"role": "user", "content": row["text"]},
        ],
        "stream": False,
        "provider": {
            "only": [model["provider"]],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
        **protocol["decoder_parameters"],
    }


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        raise urllib.error.HTTPError(
            request.full_url, code, "Redirect refused", headers, fp
        )


def send(model, body, key):
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    if model["api"] == "openrouter":
        headers["X-OpenRouter-Cache"] = "false"
    request = urllib.request.Request(
        model["endpoint"], data=json.dumps(body).encode("utf-8"), headers=headers
    )
    with urllib.request.build_opener(NoRedirect).open(request, timeout=60) as response:
        return response.read().decode("utf-8"), response.headers.get(
            "X-OpenRouter-Cache-Status"
        )


def journal_state(path):
    events = (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )
    starts, finishes, complete = {}, {}, {}
    for event in events:
        ident = (event["request_id"], event["attempt"])
        if event["event"] == "started":
            require(ident not in starts, "Duplicate attempt start")
            starts[ident] = event
        elif event["event"] == "finished":
            require(
                ident in starts and ident not in finishes,
                "Unmatched/duplicate attempt finish",
            )
            require(
                event["payload_sha256"] == starts[ident]["payload_sha256"],
                "Attempt payload changed",
            )
            finishes[ident] = event
            if event["terminal"]:
                require(
                    event["request_id"] not in complete, "Duplicate terminal request"
                )
                complete[event["request_id"]] = event
        else:
            raise ValueError("Unknown journal event")
    require(
        set(starts) == set(finishes),
        "Unfinished network attempt; reconcile before resuming to avoid duplicate billing",
    )
    return events, complete


def cost_bound(model, body):
    """Conservative accounting reserve, including special/template tokens."""
    inputs = body.get("input") or [m["content"] for m in body["messages"]]
    bound = sum(len(text.encode("utf-8")) + 64 for text in inputs)
    return (
        bound * model["input_price_usd"]
        + body.get("max_tokens", 0) * model["output_price_usd"]
    )


def response_cost(response, model):
    require(isinstance(response, dict), "Invalid response object")
    usage = response.get("usage") or {}
    require(isinstance(usage, dict), "Invalid usage object")
    prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens", 0)
    require(
        type(prompt) is int
        and prompt > 0
        and type(completion) is int
        and completion >= 0,
        "Missing/invalid token usage",
    )
    reported = usage.get("cost") if model["api"] == "openrouter" else None
    if reported is not None:
        require(
            type(reported) in (float, int)
            and math.isfinite(reported)
            and reported >= 0,
            "Invalid reported cost",
        )
        return float(reported), "provider_reported"
    return (
        prompt * model["input_price_usd"] + completion * model["output_price_usd"],
        "token_price_estimate",
    )


def interpret(response, model, protocol, row, prototypes, cache_status):
    if model["api"] == "openrouter":
        return decoder_result(response, model, protocol["class_order"], cache_status)
    if row is None:
        vectors = embeddings(
            response,
            model,
            len(protocol["class_order"]),
            sum(protocol["prototype_tokens"].values()),
        )
        return {"prototype_count": len(vectors), "dimension": model["dimension"]}
    vector = embeddings(response, model, 1, int(row["encoder_tokens"]))[0]
    scores = [
        math.fsum(a * b for a, b in zip(vector, prototype)) for prototype in prototypes
    ]
    winner = max(range(len(scores)), key=scores.__getitem__)
    ordered = sorted(scores, reverse=True)
    return {
        "label": protocol["class_order"][winner],
        "scores": dict(zip(protocol["class_order"], scores)),
        "margin": ordered[0] - ordered[1],
    }


def request_once(
    journal,
    request_id,
    model,
    body,
    key,
    protocol,
    row,
    prototypes,
    budget,
    sender=send,
    pause=time.sleep,
):
    """Retry explicit HTTP throttling/server failures once; never retry unknown delivery."""
    for _ in range(2):
        events, complete = journal_state(journal)
        if request_id in complete:
            require(
                complete[request_id]["payload_sha256"] == object_hash(body)
                and complete[request_id]["model"] == model["model"],
                "Completed request payload/model changed",
            )
            return complete[request_id]
        finishes = [e for e in events if e["event"] == "finished"]
        require(
            not any(e["status"] == "fatal" for e in finishes),
            "A fatal attempt requires reconciliation before continuing",
        )
        starts = [e for e in events if e["event"] == "started"]
        attempt = 1 + sum(e["request_id"] == request_id for e in starts)
        require(attempt <= 2 and len(starts) < 24, "Request-attempt limit reached")
        reserve = cost_bound(model, body)
        spent = math.fsum(e["accounted_cost_usd"] for e in finishes)
        require(spent + reserve <= budget, "Client accounting budget reached")
        fingerprint = object_hash(body)
        previous = [e for e in starts if e["request_id"] == request_id]
        require(
            all(e["payload_sha256"] == fingerprint for e in previous),
            "Retry payload changed",
        )
        common = {
            "request_id": request_id,
            "attempt": attempt,
            "model": model["model"],
            "payload_sha256": fingerprint,
        }
        append_jsonl(
            journal,
            {**common, "event": "started", "time": now(), "cost_reserve_usd": reserve},
        )
        started = time.monotonic()
        event = {
            **common,
            "event": "finished",
            "terminal": True,
            "status": "fatal",
            "accounted_cost_usd": reserve,
            "cost_basis": "conservative_reserve",
            "result": None,
        }
        try:
            raw, cache = sender(model, body, key)
            # Providers should not echo credentials, but keep them out of logs even if they do.
            raw = raw.replace(key, "[REDACTED]")
            event.update(raw_response=raw, cache_status=cache, http_status=200)
            response = json.loads(raw)
            cost, basis = response_cost(response, model)
            event.update(accounted_cost_usd=cost, cost_basis=basis)
            event["result"] = interpret(
                response, model, protocol, row, prototypes, cache
            )
            event["status"] = "valid"
        except urllib.error.HTTPError as error:
            event.update(http_status=error.code, error="HTTP request rejected")
            if error.code == 429 or 500 <= error.code < 600:
                event["terminal"] = attempt == 2
                event["status"] = "http_error" if event["terminal"] else "retryable"
        except (urllib.error.URLError, TimeoutError, OSError):
            event["error"] = (
                "Transport outcome unknown; reconcile billing before retrying"
            )
        except (ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
            event["error"] = str(error).replace(key, "[REDACTED]")
        event.update(time=now(), elapsed_seconds=time.monotonic() - started)
        append_jsonl(journal, event)
        if event["terminal"]:
            return event
        pause(2)
    raise ValueError("Retry limit reached")


def validate_check(directory):
    config = verified_snapshot(directory)
    require(
        config["scope"] == "heldout_compatibility_only"
        and config["planned_classifications"] == 18,
        "Only the held-out check is executable",
    )
    bindings = {
        "runner_sha256": Path(__file__),
        "model_code_sha256": Path(model_inputs.__file__),
        "utilities_sha256": ROOT / "src/utils.py",
        "main_manifest_sha256": MAIN / "manifest.json",
        "exclusions_sha256": ROOT / "data/reviews/pilot_exclusions.csv",
    }
    for name, path in bindings.items():
        require(
            config["inputs"][name] == file_digest(path),
            f"Compatibility input binding changed: {name}",
        )
    rows, schedule = read_csv(directory / "inputs.csv"), read_csv(
        directory / "request_order.csv"
    )
    originals = {r["input_id"]: r for r in heldout_records()}
    require(
        len(rows) == 6
        and len(originals) == 6
        and {r["input_id"] for r in rows} == set(originals),
        "Held-out selection changed",
    )
    for row in rows:
        require(
            all(row[k] == v for k, v in originals[row["input_id"]].items())
            and row["text_sha256"] == digest(row["text"]),
            "Held-out input changed",
        )
    require(
        len(schedule) == 18 and len({r["request_id"] for r in schedule}) == 18,
        "Invalid check schedule",
    )
    expected = Counter(
        (model, row["input_id"], "0") for model in config["models"] for row in rows
    )
    expected.update(
        (model, ident, "1")
        for model in config["models"]
        for ident in config["repeat_input_ids"]
    )
    require(
        Counter((r["model"], r["input_id"], r["replicate"]) for r in schedule)
        == expected,
        "Check schedule changed",
    )
    require(
        config["protocol"] == json.loads((MAIN / "protocol.json").read_text()),
        "Frozen protocol changed",
    )
    return config, {r["input_id"]: r for r in rows}, schedule


def summarize(output_dir, config, rows, schedule, events, complete):
    predictions = []
    for task in schedule:
        finish = complete.get(task["request_id"])
        if finish is None:
            continue
        result = finish.get("result") or {}
        predictions.append(
            {
                **task,
                "expected_label": rows[task["input_id"]]["expected_label"],
                "label": result.get("label", ""),
                "status": finish["status"],
                "generation_id": result.get("generation_id", ""),
                "margin": result.get("margin", ""),
            }
        )
    write_csv(
        output_dir / "predictions.csv",
        [*ORDER_FIELDS, "expected_label", "label", "status", "generation_id", "margin"],
        predictions,
    )
    finishes = [e for e in events if e["event"] == "finished"]
    generation_ids = [
        e["result"]["generation_id"]
        for e in finishes
        if e.get("result") and e["result"].get("generation_id")
    ]
    unique = len(generation_ids) == len(set(generation_ids))
    passed = (
        len(complete) == 19
        and all(e["status"] == "valid" for e in complete.values())
        and unique
    )
    lines = [
        "# Provider Compatibility Check",
        "",
        f"Technical gate: **{'passed' if passed else 'not passed'}**. Six held-out events; no main-study texts submitted.",
        "",
        "| Model | Valid classifications | Original labels correct | Repeat labels unchanged |",
        "| --- | --- | --- | --- |",
    ]
    for model in config["models"]:
        selected = [r for r in predictions if r["model"] == model]
        originals = {r["input_id"]: r for r in selected if r["replicate"] == "0"}
        correct = sum(r["label"] == r["expected_label"] for r in originals.values())
        repeats = [r for r in selected if r["replicate"] == "1"]
        stable = sum(
            bool(r["label"])
            and r["label"] == originals.get(r["input_id"], {}).get("label")
            for r in repeats
        )
        valid = sum(r["status"] == "valid" for r in selected)
        lines.append(f"| {model} | {valid}/9 | {correct}/6 | {stable}/3 |")
    reported = sum(
        e["accounted_cost_usd"]
        for e in finishes
        if e["cost_basis"] == "provider_reported"
    )
    estimated = sum(
        e["accounted_cost_usd"]
        for e in finishes
        if e["cost_basis"] != "provider_reported"
    )
    lines += [
        "",
        f"{len(finishes)} POST attempts, including the six-prototype E5 batch. Provider-reported charges: ${reported:.6f}; other usage/reserves estimated: ${estimated:.6f}.",
        "",
        "E5 vectors are checked for dimension, finite nonzero norms, index order and exact input-token accounting. Llama uses strict JSON labels, CoreWeave bf16 routing without fallbacks, and disabled OpenRouter response caching. Fresh generation IDs are checked for uniqueness.",
        "",
        "This small check establishes API compatibility, not classification validity or geographic invariance. Six source labels provide descriptive diagnostics only; prompts and main stimuli were not tuned from these predictions. Repeat requests do not add independent events.",
        "",
        "The manifest freezes settings, selection and code hashes. The journal retains raw responses, usage, timing and failures. Providers do not expose an immutable served-weight revision; a later main run must recheck availability and routing.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    return passed


def run_check(input_dir, output_dir, budget=0.10):
    require(
        math.isfinite(budget) and 0 < budget <= 0.25,
        "Check budget must be positive and at most $0.25",
    )
    config, rows, schedule = validate_check(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / ".active"
    lock.mkdir()  # An interrupted process requires explicit reconciliation.
    try:
        run_path = output_dir / "manifest.json"
        binding = file_digest(input_dir / "manifest.json")
        if run_path.exists():
            run = json.loads(run_path.read_text())
            require(
                run["input_manifest_sha256"] == binding and run["budget_usd"] == budget,
                "Run configuration changed",
            )
            if run.get("complete"):
                verified_snapshot(output_dir)
                return run["technical_gate_passed"]
        else:
            require(
                not any(p != lock for p in output_dir.iterdir()),
                "Use an empty output directory",
            )
            run = {
                "input_manifest_sha256": binding,
                "input_directory": (
                    str(input_dir.relative_to(ROOT))
                    if input_dir.is_relative_to(ROOT)
                    else str(input_dir)
                ),
                "started_at": now(),
                "budget_usd": budget,
                "complete": False,
                "main_experiment_started": False,
            }
            write_json(run_path, run)
        journal = output_dir / "attempts.jsonl"
        events, complete = journal_state(journal)
        require(
            not any(e.get("status") == "fatal" for e in complete.values()),
            "Fatal attempt requires reconciliation",
        )
        current = model_metadata()
        require(
            current == config["models"],
            "Provider metadata/routing/pricing changed; prepare a new check",
        )
        keys = {
            "deepinfra": load_secret("DEEPINFRA_API_KEY", ROOT / ".env"),
            "openrouter": load_secret("OPENROUTER_API_KEY", ROOT / ".env"),
        }
        key_info = get_json(OPENROUTER + "/key", keys["openrouter"])["data"]
        require(
            not key_info.get("is_management_key")
            and not key_info.get("is_provisioning_key"),
            "OpenRouter requires an inference API key",
        )
        protocol = config["protocol"]
        encoder = config["models"][model_inputs.ENCODER]
        prototype_body = {
            "model": encoder["model"],
            "input": [
                protocol["class_definitions"][label]
                for label in protocol["class_order"]
            ],
            **protocol["encoder_parameters"],
        }
        result = request_once(
            journal,
            "encoder:prototypes",
            encoder,
            prototype_body,
            keys["deepinfra"],
            protocol,
            None,
            None,
            budget,
        )
        require(
            result["status"] == "valid",
            "Prototype request failed; inspect the redacted journal",
        )
        prototypes = embeddings(
            json.loads(result["raw_response"]),
            encoder,
            6,
            sum(protocol["prototype_tokens"].values()),
        )
        for task in schedule:
            model, row = config["models"][task["model"]], rows[task["input_id"]]
            result = request_once(
                journal,
                task["request_id"],
                model,
                payload(model, row, protocol),
                keys[model["api"]],
                protocol,
                row,
                prototypes,
                budget,
            )
            if result["status"] == "fatal":
                break
        events, complete = journal_state(journal)
        passed = summarize(output_dir, config, rows, schedule, events, complete)
        run.update(
            complete=len(complete) == 19, technical_gate_passed=passed, updated_at=now()
        )
        run["outputs"] = {
            p.name: {"sha256": file_digest(p)}
            for p in output_dir.iterdir()
            if p.is_file() and p.name != "manifest.json"
        }
        write_json(run_path, run)
        return passed
    finally:
        lock.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser(
        "prepare-check", help="Freeze six held-out originals and provider metadata"
    )
    prepare.add_argument("--output-dir", type=Path, default=ROOT / "data/compatibility")
    execute = sub.add_parser(
        "check", help="Execute only the prepared 18-classification compatibility check"
    )
    execute.add_argument("--input-dir", type=Path, default=ROOT / "data/compatibility")
    execute.add_argument("--output-dir", type=Path, default=ROOT / "runs/compatibility")
    execute.add_argument("--budget", type=float, default=0.10)
    args = parser.parse_args()
    if args.command == "prepare-check":
        prepare_check(args.output_dir)
        print("Prepared six held-out inputs and 18 classification requests.")
    else:
        passed = run_check(args.input_dir, args.output_dir, args.budget)
        print(
            "Compatibility gate "
            + ("passed." if passed else "not passed; inspect the report.")
        )


if __name__ == "__main__":
    main()
