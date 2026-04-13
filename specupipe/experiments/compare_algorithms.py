import argparse
import json

from core.models import CausalLMWrapper
from core.baseline import baseline_greedy_decode
from core.naive_multitoken import naive_multitoken_decode
from core.speculative import speculative_greedy_decode
from core.analysis import speedup
from core.utils import pick_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--draft_model", type=str, default="distilgpt2")
    parser.add_argument("--target_model", type=str, default="gpt2")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mode", choices=["fixed", "adaptive", "hybrid"], default="hybrid")
    args = parser.parse_args()

    device = pick_device(args.device)

    draft = CausalLMWrapper(args.draft_model, device=device)
    target = CausalLMWrapper(args.target_model, device=device)
    eos_id = target.tokenizer.eos_token_id

    baseline_text, _, baseline_metrics = baseline_greedy_decode(
        model=target,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=eos_id,
    )

    naive_text, _, naive_metrics = naive_multitoken_decode(
        model=draft,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        block_size=args.k,
        eos_token_id=eos_id,
    )

    spec_text, _, spec_metrics, _ = speculative_greedy_decode(
        draft_model=draft,
        target_model=target,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        speculation_depth=args.k,
        eos_token_id=eos_id,
        adaptive=(args.mode in ["adaptive", "hybrid"]),
        hybrid_gated=(args.mode == "hybrid"),
        min_depth=2,
        max_depth=8,
    )

    result = {
        "prompt": args.prompt,
        "baseline_text": baseline_text,
        "naive_multitoken_text": naive_text,
        "speculative_text": spec_text,
        "baseline_metrics": baseline_metrics.to_dict(),
        "naive_multitoken_metrics": naive_metrics.to_dict(),
        "speculative_metrics": spec_metrics.to_dict(),
        "naive_matches_baseline": naive_text == baseline_text,
        "speculative_matches_baseline": spec_text == baseline_text,
        "naive_speedup_vs_baseline": speedup(baseline_metrics.total_time_sec, naive_metrics.total_time_sec),
        "speculative_speedup_vs_baseline": speedup(baseline_metrics.total_time_sec, spec_metrics.total_time_sec),
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
