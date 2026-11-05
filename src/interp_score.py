"""Score ambiguity-ladder prompts by answer log-probability.

Each answer ("Protest", "Violence against civilians", ...) can span several
tokens, so a prompt's score for an answer is the teacher-forced sum of the
log-probabilities of that answer's tokens, starting at the first position
after the model's assistant header. `score` journals one model and stage.
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .utils import (
    append_jsonl,
    file_digest,
    load_secret,
    read_csv,
    require,
    verified_snapshot,
    write_json,
)

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "interpretability" / "data" / "stimuli"
RUNS = ROOT / "interpretability" / "runs"
ENV_FILE = ROOT / "interpretability" / ".env"
PRODUCER = Path(__file__).resolve()

# Each entry fixes how the chat template is rendered and where the answer starts.
MODELS = {
    "qwen3.5-4b": {
        "hf_id": "Qwen/Qwen3.5-4B",
        "loader": "AutoModelForImageTextToText",
        "gated": False,
        "template_kwargs": {"enable_thinking": False},
        "assistant_suffix": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "required_fragments": [],
    },
    "llama3.2-3b": {
        "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
        "loader": "AutoModelForCausalLM",
        "gated": True,
        "template_kwargs": {"date_string": "26 Jul 2024"},
        "assistant_suffix": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "required_fragments": ["Today Date: 26 Jul 2024\n"],
    },
}
STAGES = ("A", "B")
TOP_TOKENS = 5
REPEATS = 20
MAIN_PASS = "main"
REPEAT_PASS = "repeat"


# ---------------------------------------------------------------- prompts


def as_ids(value):
    """Normalize apply_chat_template output (list or BatchEncoding) to ids."""
    if hasattr(value, "keys"):
        value = value["input_ids"]
    return [int(token) for token in value]


def render_prompt(tokenizer, spec, system_prompt, user_text):
    """Return the generation-prompt text and its token ids, verified."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    options = {"add_generation_prompt": True, **spec["template_kwargs"]}
    text = tokenizer.apply_chat_template(messages, tokenize=False, **options)
    require(
        text.endswith(spec["assistant_suffix"]),
        "Rendered prompt does not end at the assistant answer position",
    )
    require(text.count(system_prompt) == 1, "System prompt not rendered verbatim")
    require(text.count(user_text) == 1, "User text not rendered verbatim")
    for fragment in spec["required_fragments"]:
        require(fragment in text, f"Rendered prompt lacks {fragment!r}")
    ids = tokenizer.encode(text, add_special_tokens=False)
    require(
        ids
        == as_ids(tokenizer.apply_chat_template(messages, tokenize=True, **options)),
        "Encoding the rendered text differs from the template's token ids",
    )
    return text, ids


def answer_token_ids(tokenizer, prompt_text, prompt_ids, answer):
    """Tokens the model must emit, after the prompt, to produce `answer`."""
    full = tokenizer.encode(prompt_text + answer, add_special_tokens=False)
    require(
        full[: len(prompt_ids)] == prompt_ids,
        f"Appending {answer!r} changes the prompt's tokenization",
    )
    tail = full[len(prompt_ids) :]
    require(tail, f"Answer {answer!r} produced no tokens")
    require(
        tokenizer.decode(tail) == answer,
        f"Answer tokens do not decode to {answer!r}",
    )
    return tail


def resolve_answers(tokenizer, prompt_text, prompt_ids, classes):
    answers = {
        name: answer_token_ids(tokenizer, prompt_text, prompt_ids, name)
        for name in classes
    }
    first, second = (answers[name] for name in classes)
    require(
        first[: len(second)] != second and second[: len(first)] != first,
        "One answer's tokens are a prefix of the other's",
    )
    return answers


def scoring_positions(prompt_length, answer_length):
    """Logit rows that predict each answer token (row i predicts token i + 1)."""
    return [prompt_length - 1 + offset for offset in range(answer_length)]


def answer_logprobs(log_probs, prompt_length, answer_ids):
    """Per-token answer log-probabilities from a [position, vocab] table."""
    rows = scoring_positions(prompt_length, len(answer_ids))
    return [float(log_probs[row][token]) for row, token in zip(rows, answer_ids)]


def model_input(prompt_ids, answer_ids):
    """The last answer token is predicted, never consumed."""
    return list(prompt_ids) + list(answer_ids[:-1])


# ---------------------------------------------------------------- model


def load_model(spec, revision, device):
    import torch
    import transformers
    from huggingface_hub import HfApi

    if spec["gated"] and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = load_secret("HF_TOKEN", ENV_FILE)
    resolved = HfApi().model_info(spec["hf_id"], revision=revision).sha
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        spec["hf_id"], revision=resolved
    )
    loader = getattr(transformers, spec["loader"])
    model, info = loader.from_pretrained(
        spec["hf_id"],
        revision=resolved,
        dtype=torch.bfloat16,
        output_loading_info=True,
    )
    require(not info["missing_keys"], f"Weights missing: {info['missing_keys'][:5]}")
    require(not info.get("mismatched_keys"), "Checkpoint weights have wrong shapes")
    model.to(device).eval()
    return tokenizer, model, resolved


def score_prompt(model, tokenizer, spec, system_prompt, user_text, classes, device):
    """Answer log-probabilities for one prompt (one forward pass per answer)."""
    import torch

    prompt_text, prompt_ids = render_prompt(tokenizer, spec, system_prompt, user_text)
    answers = resolve_answers(tokenizer, prompt_text, prompt_ids, classes)
    result, top = {}, None
    for name, answer_ids in answers.items():
        ids = model_input(prompt_ids, answer_ids)
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([ids], device=device)).logits[0]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        token_logprobs = answer_logprobs(log_probs, len(prompt_ids), answer_ids)
        result[name] = {
            "token_ids": answer_ids,
            "tokens": [tokenizer.decode([token]) for token in answer_ids],
            "token_logprobs": token_logprobs,
            "logprob": sum(token_logprobs),
        }
        if top is None:
            first_row = scoring_positions(len(prompt_ids), 1)[0]
            values, indices = torch.topk(log_probs[first_row], TOP_TOKENS)
            top = [
                [tokenizer.decode([int(index)]), float(value.exp())]
                for value, index in zip(values, indices)
            ]
    return {
        "answers": result,
        "answer_mass": sum(math.exp(entry["logprob"]) for entry in result.values()),
        "first_position_top": top,
        "prompt_tokens": len(prompt_ids),
    }


# ---------------------------------------------------------------- run


def load_journal(path):
    records = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            key = (record["pass"], record["prompt_id"])
            require(key not in records, f"Duplicate journal record: {key}")
            records[key] = record
    return records


def stage_schedule(snapshot_dir, stage):
    stimuli = {
        row["stimulus_id"]: row for row in read_csv(snapshot_dir / "stimuli.csv")
    }
    tasks = {
        row["task_id"]: row for row in read_csv(snapshot_dir / "system_prompts.csv")
    }
    rows = [
        row for row in read_csv(snapshot_dir / "schedule.csv") if row["stage"] == stage
    ]
    require(rows, f"No prompts for stage {stage}")
    return [
        {
            "prompt_id": row["prompt_id"],
            "stimulus_id": row["stimulus_id"],
            "task_id": row["task_id"],
            "system_prompt": tasks[row["task_id"]]["system_prompt"],
            "text": stimuli[row["stimulus_id"]]["text"],
            "classes": [
                tasks[row["task_id"]]["lower_class"],
                tasks[row["task_id"]]["higher_class"],
            ],
        }
        for row in rows
    ]


def score(
    model_key, stage, output_dir, snapshot_dir=SNAPSHOT, revision="main", device="cuda"
):
    import torch
    import transformers

    require(model_key in MODELS, f"Unknown model: {model_key}")
    require(stage in STAGES, f"Unknown stage: {stage}")
    spec = MODELS[model_key]
    verified_snapshot(snapshot_dir)
    require(not (output_dir / "manifest.json").exists(), "Scores are already final")
    prompts = stage_schedule(snapshot_dir, stage)
    tokenizer, model, resolved = load_model(spec, revision, device)
    config = {
        "model_key": model_key,
        "hf_id": spec["hf_id"],
        "resolved_revision": resolved,
        "stage": stage,
        "snapshot_manifest_sha256": file_digest(snapshot_dir / "manifest.json"),
        "spec": spec,
        "dtype": "bfloat16",
        "batch_size": 1,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": torch.cuda.get_device_name() if device == "cuda" else device,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        require(previous == config, "Run settings differ from the existing journal")
    else:
        write_json(config_path, config)

    journal = output_dir / "scores.jsonl"
    done = load_journal(journal)
    plan = [(MAIN_PASS, prompt) for prompt in prompts]
    plan += [(REPEAT_PASS, prompt) for prompt in prompts[:REPEATS]]
    pending = [(name, p) for name, p in plan if (name, p["prompt_id"]) not in done]
    print(f"{len(plan) - len(pending)} done, {len(pending)} pending", file=sys.stderr)
    started = time.monotonic()
    for count, (pass_name, prompt) in enumerate(pending, start=1):
        record = score_prompt(
            model,
            tokenizer,
            spec,
            prompt["system_prompt"],
            prompt["text"],
            prompt["classes"],
            device,
        )
        append_jsonl(
            journal,
            {
                "pass": pass_name,
                "prompt_id": prompt["prompt_id"],
                "stimulus_id": prompt["stimulus_id"],
                "task_id": prompt["task_id"],
                **record,
            },
        )
        if count % 500 == 0:
            rate = count / (time.monotonic() - started)
            print(f"{count}/{len(pending)} at {rate:.1f}/s", file=sys.stderr)

    records = load_journal(journal)
    require(len(records) == len(plan), "Scoring is incomplete or has extra records")
    write_json(
        output_dir / "manifest.json",
        {
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "producer_sha256": file_digest(PRODUCER),
            "counts": {"main": len(prompts), "repeat": min(REPEATS, len(prompts))},
            "outputs": {
                name: {"sha256": file_digest(output_dir / name)}
                for name in ["run.json", "scores.jsonl"]
            },
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--snapshot-dir", type=Path, default=SNAPSHOT)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    output_dir = args.output_dir or RUNS / args.model / f"stage_{args.stage}"
    score(
        args.model,
        args.stage,
        output_dir,
        args.snapshot_dir,
        args.revision,
        args.device,
    )


if __name__ == "__main__":
    main()
