import argparse
import json

from core.models import CausalLMWrapper
from core.baseline import baseline_greedy_decode
from core.speculative import speculative_greedy_decode
from core.analysis import speedup
from core.utils import pick_device
from core.prompting import format_question_for_family


MODEL_FAMILIES = {
    "gpt2": {
        "draft": "distilgpt2",
        "target": "gpt2",
    },
    "tinyllama": {
        "draft": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "target": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["gpt2", "tinyllama"], required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mode", choices=["fixed", "adaptive", "hybrid"], default="hybrid")
    args = parser.parse_args()

    cfg = MODEL_FAMILIES[args.family]
    device = pick_device(args.device)
    formatted_prompt = format_question_for_family(args.question, args.family)

    print(f"Loading family: {args.family}")
    print(f"Draft model : {cfg['draft']}")
    print(f"Target model: {cfg['target']}")
    print(f"Device      : {device}")
    print(f"Question    : {args.question}")

    draft = CausalLMWrapper(cfg["draft"], device=device)
    target = CausalLMWrapper(cfg["target"], device=device)

    eos_id = target.tokenizer.eos_token_id

    baseline_text, _, baseline_metrics = baseline_greedy_decode(
        model=target,
        prompt=formatted_prompt,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=eos_id,
    )

    spec_text, _, spec_metrics, _ = speculative_greedy_decode(
        draft_model=draft,
        target_model=target,
        prompt=formatted_prompt,
        max_new_tokens=args.max_new_tokens,
        speculation_depth=args.k,
        eos_token_id=eos_id,
        adaptive=(args.mode in ["adaptive", "hybrid"]),
        hybrid_gated=(args.mode == "hybrid"),
        min_depth=2,
        max_depth=8,
    )

    result = {
        "family": args.family,
        "mode": args.mode,
        "question": args.question,
        "formatted_prompt": formatted_prompt,
        "baseline_text": baseline_text,
        "speculative_text": spec_text,
        "output_match": baseline_text == spec_text,
        "baseline_metrics": baseline_metrics.to_dict(),
        "speculative_metrics": spec_metrics.to_dict(),
        "speedup": speedup(baseline_metrics.total_time_sec, spec_metrics.total_time_sec),
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
