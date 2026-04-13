from __future__ import annotations

import torch

from core.metrics import DecodeMetrics
from core.utils import Timer


def baseline_greedy_decode(
    model,
    prompt: str,
    max_new_tokens: int = 50,
    eos_token_id: int | None = None,
):
    input_ids = model.encode(prompt)
    generated = input_ids.clone()

    metrics = DecodeMetrics(mode="baseline")

    with Timer() as t:
        for _ in range(max_new_tokens):
            next_token = model.greedy_next_token(generated)
            next_tensor = torch.tensor([[next_token]], device=generated.device, dtype=generated.dtype)
            generated = torch.cat([generated, next_tensor], dim=1)
            metrics.generated_tokens += 1

            if eos_token_id is not None and next_token == eos_token_id:
                break

    metrics.total_time_sec = t.elapsed
    text = model.decode(generated)
    return text, generated, metrics

