from __future__ import annotations


def baseline_latency_per_token(total_time_sec: float, generated_tokens: int) -> float:
    if generated_tokens == 0:
        return 0.0
    return total_time_sec / generated_tokens


def speculative_latency_per_token(total_time_sec: float, generated_tokens: int) -> float:
    if generated_tokens == 0:
        return 0.0
    return total_time_sec / generated_tokens


def speedup(baseline_time: float, speculative_time: float) -> float:
    if speculative_time == 0:
        return 0.0
    return baseline_time / speculative_time


def explain_run_difference(baseline_metrics: dict, speculative_metrics: dict) -> str:
    base_tps = baseline_metrics["tokens_per_second"]
    spec_tps = speculative_metrics["tokens_per_second"]
    acc = speculative_metrics["acceptance_rate"]
    rollback = speculative_metrics["rollback_count"]

    if spec_tps > base_tps:
        speed_phrase = "Speculative decoding was faster than baseline in this run."
    elif spec_tps < base_tps:
        speed_phrase = "Speculative decoding was slower than baseline in this run."
    else:
        speed_phrase = "Speculative decoding matched baseline in this run."

    if acc > 0.75:
        acc_phrase = "The acceptance rate was high, so most drafted tokens were useful."
    elif acc > 0.4:
        acc_phrase = "The acceptance rate was moderate, so speculative decoding still helped but with some wasted work."
    else:
        acc_phrase = "The acceptance rate was low, so rollback overhead likely reduced the benefit."

    if rollback == 0:
        rb_phrase = "There were no rollback events."
    else:
        rb_phrase = f"There were {rollback} rollback events, which introduced control overhead."

    return f"{speed_phrase} {acc_phrase} {rb_phrase}"
