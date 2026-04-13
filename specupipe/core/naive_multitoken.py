from __future__ import annotations

import torch

from core.metrics import DecodeMetrics, StepRecord
from core.utils import Timer


def naive_multitoken_decode(
    model,
    prompt: str,
    max_new_tokens: int = 50,
    block_size: int = 4,
    eos_token_id: int | None = None,
):
    """
    Naive multi-token decoding:
    - repeatedly predicts multiple future tokens
    - directly commits them without target-model verification
    - faster in spirit, but not guaranteed correct relative to a stronger target model
    """
    prefix = model.encode(prompt)
    generated_tokens = 0
    round_id = 0

    metrics = DecodeMetrics(mode="naive_multitoken")

    with Timer() as total_timer:
        while generated_tokens < max_new_tokens:
            round_id += 1
            running = prefix.clone()
            block_ids: list[int] = []

            with Timer() as draft_timer:
                for _ in range(block_size):
                    if generated_tokens + len(block_ids) >= max_new_tokens:
                        break

                    next_id = model.greedy_next_token(running)
                    block_ids.append(next_id)

                    next_tensor = torch.tensor([[next_id]], device=running.device, dtype=running.dtype)
                    running = torch.cat([running, next_tensor], dim=1)

                    if eos_token_id is not None and next_id == eos_token_id:
                        break

            with Timer() as commit_timer:
                if block_ids:
                    block_tensor = torch.tensor([block_ids], device=prefix.device, dtype=prefix.dtype)
                    prefix = torch.cat([prefix, block_tensor], dim=1)
                    generated_tokens += len(block_ids)

            metrics.add_step(
                StepRecord(
                    round_id=round_id,
                    drafted_tokens=len(block_ids),
                    accepted_tokens=len(block_ids),
                    rejected=False,
                    corrective_token_added=False,
                    committed_length_after_round=prefix.shape[1],
                    draft_time_sec=draft_timer.elapsed,
                    verify_time_sec=0.0,
                    commit_time_sec=commit_timer.elapsed,
                    rollback_penalty_sec=0.0,
                    speculation_depth=block_size,
                    round_acceptance=1.0 if block_ids else 0.0,
                    mode_used="naive_multitoken",
                    speculative_tokens_peak=len(block_ids),
                    estimated_kv_overhead_units=0.0,
                    pipeline_busy_sec=draft_timer.elapsed + commit_timer.elapsed,
                    pipeline_bubble_sec=0.0,
                )
            )

            metrics.generated_tokens = generated_tokens

            if eos_token_id is not None and prefix[0, -1].item() == eos_token_id:
                break

    metrics.total_time_sec = total_timer.elapsed
    metrics.generated_tokens = generated_tokens
    text = model.decode(prefix)
    return text, prefix, metrics
