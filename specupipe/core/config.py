from dataclasses import dataclass


@dataclass
class DecodeConfig:
    max_new_tokens: int = 50
    temperature: float = 1.0
    do_sample: bool = False
    speculation_depth: int = 4
    eos_token_id: int | None = None


@dataclass
class ExperimentConfig:
    draft_model_name: str = "distilgpt2"
    target_model_name: str = "gpt2"
    device: str = "cuda"

