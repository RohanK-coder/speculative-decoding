import pandas as pd

from core.models import CausalLMWrapper
from core.baseline import baseline_greedy_decode
from core.speculative import speculative_greedy_decode
from core.utils import pick_device


PROMPTS = [
    {"category": "explanatory", "text": "Speculative decoding is useful because"},
    {"category": "technical", "text": "The role of a KV cache in transformer inference is"},
    {"category": "algorithmic", "text": "A rollback is required in speculative decoding when"},
    {"category": "analysis", "text": "The tradeoff between verification cost and speculation depth is"},
    {"category": "open_ended", "text": "Write a short paragraph about future AI hardware"},
]

OUTPUT_LENGTHS = [32, 64]
DEPTHS = [2, 4, 8]


def main():
    device = pick_device("cpu")

    draft = CausalLMWrapper("distilgpt2", device=device)
    target = CausalLMWrapper("gpt2", device=device)
    eos_id = target.tokenizer.eos_token_id

    rows = []

    for item in PROMPTS:
        prompt = item["text"]
        category = item["category"]

        for max_new_tokens in OUTPUT_LENGTHS:
            baseline_text, _, _ = baseline_greedy_decode(
                model=target,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_id,
            )

            for k in DEPTHS:
                for mode_name, kwargs in [
                    ("fixed", {"adaptive": False, "hybrid_gated": False}),
                    ("adaptive", {"adaptive": True, "hybrid_gated": False}),
                    ("hybrid", {"adaptive": True, "hybrid_gated": True}),
                ]:
                    spec_text, _, _, _ = speculative_greedy_decode(
                        draft_model=draft,
                        target_model=target,
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        speculation_depth=k,
                        eos_token_id=eos_id,
                        min_depth=2,
                        max_depth=8,
                        prompt_category=category,
                        **kwargs,
                    )

                    match = baseline_text == spec_text
                    rows.append(
                        {
                            "category": category,
                            "prompt": prompt,
                            "max_new_tokens": max_new_tokens,
                            "mode": mode_name,
                            "k": k,
                            "output_match": match,
                        }
                    )

    df = pd.DataFrame(rows)
    total = len(df)
    passed = int(df["output_match"].sum())
    failed = total - passed

    print("\nValidation summary")
    print("------------------")
    print(f"Total cases : {total}")
    print(f"Passed      : {passed}")
    print(f"Failed      : {failed}")

    if failed > 0:
        print("\nFailed cases:")
        print(df[df["output_match"] == False].to_string(index=False))
    else:
        print("\nAll correctness checks passed.")

    df.to_csv("outputs/results/validation_results.csv", index=False)
    print("\nSaved: outputs/results/validation_results.csv")


if __name__ == "__main__":
    main()
