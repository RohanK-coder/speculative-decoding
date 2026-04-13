import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.models import CausalLMWrapper
from core.baseline import baseline_greedy_decode
from core.speculative import speculative_greedy_decode
from core.utils import pick_device


def test_speculative_matches_target_greedy():
    device = pick_device("cuda")

    draft = CausalLMWrapper("distilgpt2", device=device)
    target = CausalLMWrapper("gpt2", device=device)

    prompt = "The purpose of speculative decoding is"
    eos_id = target.tokenizer.eos_token_id

    baseline_text, baseline_ids, _ = baseline_greedy_decode(
        model=target,
        prompt=prompt,
        max_new_tokens=20,
        eos_token_id=eos_id,
    )

    speculative_text, speculative_ids, _, _ = speculative_greedy_decode(
        draft_model=draft,
        target_model=target,
        prompt=prompt,
        max_new_tokens=20,
        speculation_depth=4,
        eos_token_id=eos_id,
    )

    assert baseline_ids.tolist() == speculative_ids.tolist()
    assert baseline_text == speculative_text

