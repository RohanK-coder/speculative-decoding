import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from core.models import CausalLMWrapper
from core.baseline import baseline_greedy_decode
from core.speculative import speculative_greedy_decode
from core.analysis import speedup
from core.utils import pick_device


PROMPTS = [
    {"category": "explanatory", "text": "Speculative decoding is useful because"},
    {"category": "explanatory", "text": "The role of verification in speculative decoding is"},
    {"category": "technical", "text": "The role of a KV cache in transformer inference is"},
    {"category": "technical", "text": "Verification becomes a bottleneck when"},
    {"category": "algorithmic", "text": "A rollback is required in speculative decoding when"},
    {"category": "algorithmic", "text": "Adaptive speculation depth should increase when"},
    {"category": "open_ended", "text": "Explain why efficient inference matters in production systems"},
    {"category": "open_ended", "text": "Discuss the tradeoff between speed and correctness in LLM inference"},
]

DEPTHS = [2, 3, 4, 6]
OUTPUT_LENGTHS = [32, 64]

SPEED_RATIO_CONFIGS = [
    ("fast_draft", 0.7, 1.0),
    ("balanced", 1.0, 1.0),
    ("slow_draft", 1.3, 1.0),
]


def main():
    out_dir = Path("outputs/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device("cpu")

    draft = CausalLMWrapper("distilgpt2", device=device)
    target = CausalLMWrapper("gpt2", device=device)
    eos_id = target.tokenizer.eos_token_id

    mode_configs = [
        ("fixed", {"adaptive": False, "hybrid_gated": False, "use_category_policy": False}),
        ("adaptive_only", {"adaptive": True, "hybrid_gated": False, "use_category_policy": False}),
        ("hybrid_only", {"adaptive": False, "hybrid_gated": True, "use_category_policy": False}),
        ("category_only", {"adaptive": False, "hybrid_gated": False, "use_category_policy": True}),
        ("hybrid_category", {"adaptive": True, "hybrid_gated": True, "use_category_policy": True}),
    ]

    rows = []

    total_runs = len(PROMPTS) * len(OUTPUT_LENGTHS) * len(DEPTHS) * len(mode_configs) * len(SPEED_RATIO_CONFIGS)
    with tqdm(total=total_runs) as pbar:
        for item in PROMPTS:
            prompt = item["text"]
            category = item["category"]

            for max_new_tokens in OUTPUT_LENGTHS:
                baseline_text, _, baseline_metrics = baseline_greedy_decode(
                    model=target,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=eos_id,
                )

                for ratio_name, draft_scale, verify_scale in SPEED_RATIO_CONFIGS:
                    for k in DEPTHS:
                        for mode_name, kwargs in mode_configs:
                            prompt_category = category if kwargs["use_category_policy"] else None

                            spec_text, _, spec_metrics, _ = speculative_greedy_decode(
                                draft_model=draft,
                                target_model=target,
                                prompt=prompt,
                                max_new_tokens=max_new_tokens,
                                speculation_depth=k,
                                eos_token_id=eos_id,
                                min_depth=2,
                                max_depth=8,
                                adaptive=kwargs["adaptive"],
                                hybrid_gated=kwargs["hybrid_gated"],
                                prompt_category=prompt_category,
                                draft_time_scale=draft_scale,
                                verify_time_scale=verify_scale,
                            )

                            rows.append(
                                {
                                    "prompt": prompt,
                                    "category": category,
                                    "max_new_tokens": max_new_tokens,
                                    "mode": mode_name,
                                    "k": k,
                                    "speed_ratio_case": ratio_name,
                                    "draft_time_scale": draft_scale,
                                    "verify_time_scale": verify_scale,
                                    "baseline_time_sec": baseline_metrics.total_time_sec,
                                    "baseline_tps": baseline_metrics.tokens_per_second,
                                    "baseline_energy_proxy_units": baseline_metrics.energy_proxy_units,
                                    "baseline_energy_per_token_proxy": baseline_metrics.energy_per_token_proxy,
                                    "spec_time_sec": spec_metrics.total_time_sec,
                                    "spec_tps": spec_metrics.tokens_per_second,
                                    "acceptance_rate": spec_metrics.acceptance_rate,
                                    "avg_round_acceptance": spec_metrics.avg_round_acceptance,
                                    "rollback_count": spec_metrics.rollback_count,
                                    "baseline_fallback_steps": spec_metrics.baseline_fallback_steps,
                                    "wasted_draft_tokens": spec_metrics.wasted_draft_tokens,
                                    "draft_time_sec": spec_metrics.total_draft_time_sec,
                                    "verify_time_sec": spec_metrics.total_verify_time_sec,
                                    "commit_time_sec": spec_metrics.total_commit_time_sec,
                                    "rollback_penalty_sec": spec_metrics.total_rollback_penalty_sec,
                                    "pipeline_utilization": spec_metrics.pipeline_utilization,
                                    "accepted_tokens_per_round": spec_metrics.accepted_tokens_per_round,
                                    "draft_time_per_accepted_token": spec_metrics.draft_time_per_accepted_token,
                                    "verify_time_per_accepted_token": spec_metrics.verify_time_per_accepted_token,
                                    "wasted_draft_per_accepted_token": spec_metrics.wasted_draft_per_accepted_token,
                                    "stall_rounds": spec_metrics.stall_rounds,
                                    "backpressure_events": spec_metrics.backpressure_events,
                                    "verify_bottleneck_ratio": spec_metrics.verify_bottleneck_ratio,
                                    "energy_proxy_units": spec_metrics.energy_proxy_units,
                                    "energy_per_token_proxy": spec_metrics.energy_per_token_proxy,
                                    "energy_per_accepted_token_proxy": spec_metrics.energy_per_accepted_token_proxy,
                                    "peak_speculative_tokens": spec_metrics.peak_speculative_tokens,
                                    "peak_estimated_kv_overhead_units": spec_metrics.peak_estimated_kv_overhead_units,
                                    "speedup": speedup(
                                        baseline_metrics.total_time_sec,
                                        spec_metrics.total_time_sec,
                                    ),
                                    "output_match": baseline_text == spec_text,
                                }
                            )
                            pbar.update(1)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "grid_results.csv", index=False)

    summary = df.groupby(["mode", "k", "max_new_tokens", "speed_ratio_case"], as_index=False).mean(numeric_only=True)
    summary.to_csv(out_dir / "grid_summary.csv", index=False)

    category_summary = df.groupby(["category", "mode", "k", "max_new_tokens", "speed_ratio_case"], as_index=False).mean(numeric_only=True)
    category_summary.to_csv(out_dir / "category_summary.csv", index=False)

    with open(out_dir / "grid_results.json", "w") as f:
        json.dump(rows, f, indent=2)

    print(summary.head(30))
    print()
    print(category_summary.head(40))


if __name__ == "__main__":
    main()
