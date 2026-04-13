from __future__ import annotations

from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelOutput:
    logits: torch.Tensor


class CausalLMWrapper:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = {"low_cpu_mem_usage": True}

        if device == "mps":
            load_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()
        self.model.to(device)

    def encode(self, text: str) -> torch.Tensor:
        enc = self.tokenizer(text, return_tensors="pt")
        return enc["input_ids"].to(self.device)

    def decode(self, input_ids: torch.Tensor) -> str:
        if input_ids.dim() == 2:
            input_ids = input_ids[0]
        return self.tokenizer.decode(input_ids, skip_special_tokens=True)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> ModelOutput:
        out = self.model(input_ids=input_ids)
        return ModelOutput(logits=out.logits)

    @torch.no_grad()
    def greedy_next_token(self, input_ids: torch.Tensor) -> int:
        out = self.model(input_ids=input_ids)
        next_token_logits = out.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).item()
        return next_token

    @torch.no_grad()
    def verify_block(self, prefix_ids: torch.Tensor, draft_ids: list[int]) -> list[int]:
        if len(draft_ids) == 0:
            return []

        draft_tensor = torch.tensor([draft_ids], device=self.device, dtype=prefix_ids.dtype)
        full_input = torch.cat([prefix_ids, draft_tensor], dim=1)

        out = self.model(input_ids=full_input)
        logits = out.logits
        n = prefix_ids.shape[1]

        predictions = []
        for i in range(len(draft_ids)):
            pos = n - 1 + i
            token_id = torch.argmax(logits[:, pos, :], dim=-1).item()
            predictions.append(token_id)

        return predictions
