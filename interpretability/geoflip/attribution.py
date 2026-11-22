"""Token-matched integrated gradients and residual steering.

Integrated gradients run between two prompts of equal length that differ only
in the name span: E(a) = E_base + a (E_target - E_base) over midpoint steps
a_k = (k + 0.5) / N. The attribution to residual position t at layer l is

    A[l, t] = sum_k grad_{h_l,t} d(a_k) . (h_l,t(a_k + 1/2N) - h_l,t(a_k - 1/2N)),

so that, for every layer, the attributions over positions sum to
d(1) - d(0) up to discretisation error (every path from the input to the
output passes through layer l). Layer -1 is the input embedding, where only the
name positions differ. The model runs in its own dtype (bf16 for Llama); the
products are accumulated in float32.
"""

from .hooks import _hidden, letter_d
from .utils import require

GRAD_MODE = "bf16 forward/backward, float32 accumulation of grad x delta-h"


def _forward(model, embeds, keep_grad):
    """Logits at the last position and [h_-1, h_0, ..] (h_-1 = embeds)."""
    states = [embeds]
    handles = []

    def hook(module, args, output):
        hidden = _hidden(output)
        if keep_grad:
            hidden.retain_grad()
        states.append(hidden)

    for layer in model.model.layers:
        handles.append(layer.register_forward_hook(hook))
    try:
        logits = model(inputs_embeds=embeds).logits[0, -1]
    finally:
        for handle in handles:
            handle.remove()
    return logits, states


def layer_ig(model, ids_base, ids_target, letter_ids, task, steps=50):
    """Attributions [L + 1, seq] (row 0 = embedding) for d, plus endpoint d's."""
    import torch

    require(len(ids_base) == len(ids_target), "IG pair differs in length")
    device = next(model.parameters()).device
    embed = model.model.embed_tokens
    with torch.no_grad():
        e0 = embed(torch.tensor([ids_base], device=device)).float()
        e1 = embed(torch.tensor([ids_target], device=device)).float()
    dtype = embed.weight.dtype
    delta = e1 - e0
    differ = (delta.abs().sum(-1)[0] > 0).nonzero().flatten()
    start = int(differ.min()) if len(differ) else len(ids_base) - 1

    def at(alpha):
        return (e0 + alpha * delta).to(dtype)

    def grid_states(alpha):
        with torch.no_grad():
            logits, states = _forward(model, at(alpha), keep_grad=False)
        d = float(letter_d(torch.log_softmax(logits.float(), -1), letter_ids, task))
        return d, [s[0, start:].float() for s in states]

    n_layers = len(model.model.layers)
    attr = torch.zeros(n_layers + 1, len(ids_base), device=device)
    d0, prev = grid_states(0.0)
    d_start = d0
    for k in range(steps):
        d_next, nxt = grid_states((k + 1) / steps)
        embeds = at((k + 0.5) / steps).requires_grad_(True)
        logits, states = _forward(model, embeds, keep_grad=True)
        d = letter_d(torch.log_softmax(logits.float(), -1), letter_ids, task)
        model.zero_grad(set_to_none=True)
        d.backward()
        for layer, state in enumerate(states):
            grad = state.grad[0, start:].float()
            attr[layer, start:] += (grad * (nxt[layer] - prev[layer])).sum(-1)
        del states, embeds, logits, d
        prev = nxt
        d_end = d_next
    totals = attr.sum(-1)
    return {
        "attr": attr.cpu(),
        "d_base": d_start,
        "d_target": d_end,
        "completeness_error": (totals - (d_end - d_start)).abs().max().item(),
    }


def steer(runner, ids, layer, positions, vector):
    """Log-probabilities with `vector` added to h_layer at `positions`."""
    log_probs, _ = runner.run(ids, add={layer: (positions, vector)})
    return log_probs
