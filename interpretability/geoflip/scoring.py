"""Score a two-option prompt by the log-probabilities of its answer letters.

The prompt is rendered with the model's chat template (verified token by
token), and log P("A") and log P("B") are read at the first answer position.
One prompt per forward pass; no batching or padding (batched bf16 scoring
moved log-probabilities by up to 0.6 nats).
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
