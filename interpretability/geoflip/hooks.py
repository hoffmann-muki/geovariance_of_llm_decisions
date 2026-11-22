"""Residual-stream capture, RMSNorm logit lens and span patching for Llama models.

h_l (l = 0..L-1) is the hidden-state tensor returned by model.model.layers[l]
(the residual stream after attention and MLP), shape [1, seq, hidden]; h_-1 is
the output of model.model.embed_tokens. Every run is one prompt per forward
pass, in the model's own dtype, and reads the answer exactly as
geoflip.scoring.score_first_token does (log-softmax of the last position's
logits in float32).
"""

from contextlib import contextmanager

from .scoring import answer_token_ids
from .utils import require


def _hidden(output):
    """Decoder layers return a tensor or a tuple whose first item is the tensor."""
    return output[0] if isinstance(output, tuple) else output


def _replace(output, hidden):
    return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden


class Runner:
    """One-prompt forward passes with optional capture, patching and addition."""

    def __init__(self, model, tokenizer, spec, device):
        self.model, self.tokenizer, self.spec, self.device = (
            model,
            tokenizer,
            spec,
            device,
        )
        self.layers = model.model.layers
        self.n_layers = len(self.layers)

    def letter_ids(self, prompt_text, prompt_ids, letters=("A", "B")):
        out = {}
        for letter in letters:
            tail = answer_token_ids(self.tokenizer, prompt_text, prompt_ids, letter)
            require(len(tail) == 1, f"Answer {letter!r} is not a single token")
            out[letter] = tail[0]
        return out

    @contextmanager
    def _hooks(self, capture=None, patch=None, add=None):
        """capture: list to fill with [h_-1, h_0, ..]; patch: {layer: (positions,
        values[len(positions), hidden])}; add: {layer: (positions, vector)}.
        Layer -1 is the embedding output."""
        import torch

        handles = []
        patch, add = patch or {}, add or {}

        def make(layer):
            def hook(module, args, output):
                hidden = _hidden(output)
                if layer in patch or layer in add:
                    hidden = hidden.clone()
                    if layer in patch:
                        positions, values = patch[layer]
                        hidden[0, positions] = values.to(hidden.dtype)
                    if layer in add:
                        positions, vector = add[layer]
                        hidden[0, positions] += vector.to(hidden.dtype)
                    output = _replace(output, hidden)
                if capture is not None:
                    capture.append(hidden.detach())
                return output

            return hook

        try:
            handles.append(
                self.model.model.embed_tokens.register_forward_hook(make(-1))
            )
            for index, layer in enumerate(self.layers):
                handles.append(layer.register_forward_hook(make(index)))
            with torch.inference_mode():
                yield
        finally:
            for handle in handles:
                handle.remove()

    def run(self, ids, capture=False, patch=None, add=None, embeds=None):
        """Log-probabilities at the last position (float32) and, if asked, the
        hidden states [h_-1, h_0, .., h_{L-1}] each of shape [seq, hidden]."""
        import torch

        states = [] if capture else None
        with self._hooks(states, patch, add):
            if embeds is not None:
                logits = self.model(inputs_embeds=embeds).logits[0, -1]
            else:
                logits = self.model(
                    input_ids=torch.tensor([ids], device=self.device)
                ).logits[0, -1]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        if capture:
            require(len(states) == self.n_layers + 1, "Missing captured layers")
            states = [s[0] for s in states]
        return log_probs, states

    def lens(self, states, position):
        """RMSNorm-calibrated logit lens: log-softmax of lm_head(norm(h_l[t])) for
        every layer l = 0..L-1, float32, shape [L, vocab]."""
        import torch

        with torch.inference_mode():
            h = torch.stack([s[position] for s in states[1:]])
            logits = self.model.lm_head(self.model.model.norm(h))
        return torch.log_softmax(logits.float(), dim=-1)

    def d_fp32(self, states, letter_ids, task):
        """d at the last position from the final residual, with the final norm
        and the two letter rows of lm_head in float32. bf16 logits move in
        steps of 0.125 nats, too coarse for single-position patching effects."""
        import torch

        norm = self.model.model.norm
        with torch.no_grad():
            h = states[-1][-1].float()
            h = h * torch.rsqrt(h.pow(2).mean() + norm.variance_epsilon)
            h = h * norm.weight.float()
            w = self.model.lm_head.weight
            hi = w[letter_ids[task["letter_higher"]]].float()
            lo = w[letter_ids[task["letter_lower"]]].float()
            return float(h @ (hi - lo))


def letter_d(log_probs, letter_ids, task):
    """d = log P(higher letter) - log P(lower letter); works on [..., vocab]."""
    return (
        log_probs[..., letter_ids[task["letter_higher"]]]
        - log_probs[..., letter_ids[task["letter_lower"]]]
    )


def span_patch(states, layer, positions):
    """Patch spec replacing `positions` at `layer` with values from `states`."""
    return {layer: (positions, states[layer + 1][positions])}
