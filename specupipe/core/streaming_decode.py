from __future__ import annotations

import torch

from core.cache import KVCacheManager
from core.metrics import DecodeMetrics, StepRecord
from core.utils import Timer


def stream_baseline_greedy_decode(model, prompt: str, max_new_tokens: int = 50, eos_token_id: int | None = None):
    input_ids = model.encode(prompt)
    generated = input_ids.clone()

    metrics = DecodeMetrics(mode="baseline")

    with Timer() as total_timer:
        for _ in range(max_new_tokens):
            next_token = model.greedy_next_token(generated)
            next_tensor = torch.tensor([[next_token]], device=generated.device, dtype=generated.dtype)
            generated = torch.cat([generated, next_tensor], dim=1)
            metrics.generated_tokens += 1

            current_text = model.decode(generated)
            yield {
                "type": "token",
                "text": current_text,
            }

            if eos_token_id is not None and next_token == eos_token_id:
                break

    metrics.total_time_sec = total_timer.elapsed
    final_text = model.decode(generated)

    yield {
        "type": "final",
        "text": final_text,
        "ids": generated,
        "metrics": metrics,
    }


def stream_speculative_greedy_decode(
    draft_model,
    target_model,
    prompt: str,
    max_new_tokens: int = 50,
    speculation_depth: int = 4,
    eos_token_id: int | None = None,
    adaptive: bool = False,
    min_depth: int = 2,
    max_depth: int = 8,
    grow_threshold: float = 0.80,
    shrink_threshold: float = 0.40,
    hybrid_gated: bool = False,
    gate_threshold: float = 0.45,
):
    prefix = target_model.encode(prompt)

    cache = KVCacheManager(committed_length=prefix.shape[1], speculative_length=0)
    metrics = DecodeMetrics(mode="hybrid" if hybrid_gated else ("adaptive" if adaptive else "fixed"))

    round_id = 0
    new_tokens_generated = 0
    current_depth = speculation_depth
    recent_acceptances: list[float] = []

    with Timer() as total_timer:
        while new_tokens_generated < max_new_tokens:
            round_id += 1
            cache.checkpoint()

            recent_avg = sum(recent_acceptances[-3:]) / min(len(recent_acceptances), 3) if recent_acceptances else 1.0
            use_baseline_fallback = hybrid_gated and round_id > 1 and recent_avg < gate_threshold

            if use_baseline_fallback:
                with Timer() as verify_timer:
                    next_id = target_model.greedy_next_token(prefix)

                with Timer() as commit_timer:
                    next_tensor = torch.tensor([[next_id]], device=prefix.device, dtype=prefix.dtype)
                    prefix = torch.cat([prefix, next_tensor], dim=1)
                    new_tokens_generated += 1

                metrics.add_step(
                    StepRecord(
                        round_id=round_id,
                        drafted_tokens=0,
                        accepted_tokens=0,
                        rejected=False,
                        corrective_token_added=False,
                        committed_length_after_round=cache.committed_length,
                        draft_time_sec=0.0,
                        verify_time_sec=verify_timer.elapsed,
                        commit_time_sec=commit_timer.elapsed,
                        rollback_penalty_sec=0.0,
                        speculation_depth=0,
                        round_acceptance=0.0,
                        mode_used="baseline_fallback",
                        speculative_tokens_peak=0,
                        estimated_kv_overhead_units=0.0,
                        pipeline_busy_sec=verify_timer.elapsed + commit_timer.elapsed,
                        pipeline_bubble_sec=0.0,
                    )
                )
                metrics.generated_tokens = new_tokens_generated

                current_text = target_model.decode(prefix)
                yield {
                    "type": "token",
                    "text": current_text,
                    "round_id": round_id,
                    "accepted": accepted,
                    "drafted": drafted_count,
                    "rejected": rejected,
                    "acceptance_rate_so_far": metrics.acceptance_rate,
                    "generated_tokens": new_tokens_generated,
                }

                if eos_token_id is not None and prefix[0, -1].item() == eos_token_id:
                    break
                continue

            draft_ids: list[int] = []
            running = prefix.clone()

            with Timer() as draft_timer:
                for _ in range(current_depth):
                    if new_tokens_generated + len(draft_ids) >= max_new_tokens:
                        break

                    next_id = draft_model.greedy_next_token(running)
                    draft_ids.append(next_id)

                    next_tensor = torch.tensor([[next_id]], device=running.device, dtype=running.dtype)
                    running = torch.cat([running, next_tensor], dim=1)
                    cache.append_speculative_token()

                    if eos_token_id is not None and next_id == eos_token_id:
                        break

            if len(draft_ids) == 0:
                break

            with Timer() as verify_timer:
                target_preds = target_model.verify_block(prefix, draft_ids)

            accepted = 0
            corrective_token = None

            for d_tok, t_tok in zip(draft_ids, target_preds):
                if d_tok == t_tok:
                    accepted += 1
                else:
                    corrective_token = t_tok
                    break

            drafted_count = len(draft_ids)
            remaining_budget = max_new_tokens - new_tokens_generated
            accepted = min(accepted, remaining_budget)

            with Timer() as commit_timer:
                if accepted > 0:
                    accepted_tensor = torch.tensor(
                        [draft_ids[:accepted]],
                        device=prefix.device,
                        dtype=prefix.dtype,
                    )
                    prefix = torch.cat([prefix, accepted_tensor], dim=1)
                    cache.commit_accepted_prefix(accepted)
                    new_tokens_generated += accepted
                else:
                    cache.commit_accepted_prefix(0)

            rejected = accepted < drafted_count
            corrective_added = False

            with Timer() as rollback_timer:
                if new_tokens_generated >= max_new_tokens:
                    cache.discard_remaining_speculative()
                elif rejected:
                    cache.discard_remaining_speculative()

                    if corrective_token is None:
                        corrective_token = target_model.greedy_next_token(prefix)

                    corr_tensor = torch.tensor(
                        [[corrective_token]],
                        device=prefix.device,
                        dtype=prefix.dtype,
                    )
                    prefix = torch.cat([prefix, corr_tensor], dim=1)
                    cache.commit_corrective_token()
                    new_tokens_generated += 1
                    corrective_added = True
                else:
                    cache.discard_remaining_speculative()

            round_acceptance = accepted / drafted_count if drafted_count > 0 else 0.0
            recent_acceptances.append(round_acceptance)

            speculative_tokens_peak = drafted_count
            estimated_kv_overhead_units = 2.0 * 12 * 768 * speculative_tokens_peak
            pipeline_busy_sec = draft_timer.elapsed + verify_timer.elapsed + commit_timer.elapsed
            pipeline_bubble_sec = rollback_timer.elapsed if rejected else 0.0

            metrics.add_step(
                StepRecord(
                    round_id=round_id,
                    drafted_tokens=drafted_count,
                    accepted_tokens=accepted,
                    rejected=rejected,
                    corrective_token_added=corrective_added,
                    committed_length_after_round=cache.committed_length,
                    draft_time_sec=draft_timer.elapsed,
                    verify_time_sec=verify_timer.elapsed,
                    commit_time_sec=commit_timer.elapsed,
                    rollback_penalty_sec=rollback_timer.elapsed,
                    speculation_depth=current_depth,
                    round_acceptance=round_acceptance,
                    mode_used="speculative",
                    speculative_tokens_peak=speculative_tokens_peak,
                    estimated_kv_overhead_units=estimated_kv_overhead_units,
                    pipeline_busy_sec=pipeline_busy_sec,
                    pipeline_bubble_sec=pipeline_bubble_sec,
                )
            )

            metrics.generated_tokens = new_tokens_generated

            current_text = target_model.decode(prefix)
            yield {
                "type": "token",
                "text": current_text,
                "round_id": round_id,
                "accepted": accepted,
                "drafted": drafted_count,
                "rejected": rejected,
            }

            if adaptive:
                if round_acceptance >= grow_threshold and current_depth < max_depth:
                    current_depth += 1
                elif round_acceptance <= shrink_threshold and current_depth > min_depth:
                    current_depth -= 1

            if eos_token_id is not None and prefix[0, -1].item() == eos_token_id:
                break

    metrics.total_time_sec = total_timer.elapsed
    metrics.generated_tokens = new_tokens_generated
    final_text = target_model.decode(prefix)

    yield {
        "type": "final",
        "text": final_text,
        "ids": prefix,
        "metrics": metrics,
        "cache": cache,
    }
