"""Score a classification prompt by teacher-forced answer log-probability.

Each answer ("Protest", "Violence against civilians", ...) can span several
tokens, so a prompt's score for an answer is the sum of the log-probabilities
of that answer's tokens, starting at the first position after the model's
assistant header. One prompt per forward pass per answer; no batching or
padding (batched bf16 scoring moved log-probabilities by up to 0.6 nats).
"""

import math
import os
from pathlib import Path

from .utils import load_secret, require

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

# Each entry fixes how the chat template is rendered and where the answer starts.
MODELS = {
    "qwen3-4b": {
        "hf_id": "Qwen/Qwen3-4B",
        "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "loader": "AutoModelForCausalLM",
        "dtype": "float32",
        "gated": False,
        "template_kwargs": {"enable_thinking": False},
        "assistant_suffix": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "required_fragments": [],
    },
    "llama3.2-3b": {
        "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
        "revision": "0cb88a4f764b7a12671c53f0838cd831a0843b95",
        "loader": "AutoModelForCausalLM",
        "dtype": "bfloat16",
        "gated": True,
        "template_kwargs": {"date_string": "26 Jul 2024"},
        "assistant_suffix": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "required_fragments": ["Today Date: 26 Jul 2024\n"],
    },
}
TOP_TOKENS = 5


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


def load_model(spec, device):
    """Tokenizer, model and the resolved commit for spec["revision"]."""
    import torch
    import transformers
    from huggingface_hub import HfApi

    if spec["gated"] and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = load_secret("HF_TOKEN", ENV_FILE)
    resolved = HfApi().model_info(spec["hf_id"], revision=spec["revision"]).sha
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        spec["hf_id"], revision=resolved
    )
    loader = getattr(transformers, spec["loader"])
    model, info = loader.from_pretrained(
        spec["hf_id"],
        revision=resolved,
        dtype=getattr(torch, spec["dtype"]),
        output_loading_info=True,
    )
    require(
        not info["missing_keys"], f"Weights missing: {sorted(info['missing_keys'])[:5]}"
    )
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


def score_first_token(
    model, tokenizer, spec, system_prompt, user_text, letters, device
):
    """Log-probabilities of single-token answers at the first answer position.

    One forward pass over the prompt. Each answer in `letters` must be exactly
    one token when appended to the rendered prompt."""
    import torch

    prompt_text, prompt_ids = render_prompt(tokenizer, spec, system_prompt, user_text)
    ids = {}
    for letter in letters:
        tail = answer_token_ids(tokenizer, prompt_text, prompt_ids, letter)
        require(len(tail) == 1, f"Answer {letter!r} is {len(tail)} tokens, not one")
        ids[letter] = tail[0]
    with torch.inference_mode():
        logits = model(input_ids=torch.tensor([prompt_ids], device=device)).logits[
            0, -1
        ]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    values, indices = torch.topk(log_probs, TOP_TOKENS)
    lp = {letter: float(log_probs[token]) for letter, token in ids.items()}
    return {
        "lp": lp,
        "mass": sum(math.exp(v) for v in lp.values()),
        "top": [
            [tokenizer.decode([int(i)]), float(v)] for v, i in zip(values, indices)
        ],
        "prompt_tokens": len(prompt_ids),
    }
