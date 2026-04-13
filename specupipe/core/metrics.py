from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepRecord:
    round_id: int
    drafted_tokens: int
    accepted_tokens: int
    rejected: bool
    corrective_token_added: bool
    committed_length_after_round: int
    draft_time_sec: float = 0.0
    verify_time_sec: float = 0.0
    commit_time_sec: float = 0.0
    rollback_penalty_sec: float = 0.0
    speculation_depth: int = 0
    round_acceptance: float = 0.0
    mode_used: str = "speculative"
    speculative_tokens_peak: int = 0
    estimated_kv_overhead_units: float = 0.0
    pipeline_busy_sec: float = 0.0
    pipeline_bubble_sec: float = 0.0
    backpressure_event: bool = False
    stall_event: bool = False


@dataclass
class DecodeMetrics:
    mode: str
    total_time_sec: float = 0.0
    generated_tokens: int = 0
    rounds: int = 0
    draft_tokens_total: int = 0
    accepted_tokens_total: int = 0
    rollback_count: int = 0
    baseline_fallback_steps: int = 0
    peak_speculative_tokens: int = 0
    peak_estimated_kv_overhead_units: float = 0.0
    step_records: list[StepRecord] = field(default_factory=list)

    def add_step(self, record: StepRecord):
        self.step_records.append(record)
        self.rounds += 1
        self.draft_tokens_total += record.drafted_tokens
        self.accepted_tokens_total += record.accepted_tokens
        if record.rejected:
            self.rollback_count += 1
        if record.mode_used == "baseline_fallback":
            self.baseline_fallback_steps += 1
        self.peak_speculative_tokens = max(self.peak_speculative_tokens, record.speculative_tokens_peak)
        self.peak_estimated_kv_overhead_units = max(
            self.peak_estimated_kv_overhead_units,
            record.estimated_kv_overhead_units,
        )

    @property
    def acceptance_rate(self) -> float:
        if self.draft_tokens_total == 0:
            return 0.0
        return self.accepted_tokens_total / self.draft_tokens_total

    @property
    def tokens_per_second(self) -> float:
        if self.total_time_sec == 0:
            return 0.0
        return self.generated_tokens / self.total_time_sec

    @property
    def wasted_draft_tokens(self) -> int:
        return self.draft_tokens_total - self.accepted_tokens_total

    @property
    def avg_round_acceptance(self) -> float:
        vals = [r.round_acceptance for r in self.step_records if r.mode_used != "baseline_fallback"]
        if not vals:
            return 0.0
        return sum(vals) / len(vals)

    @property
    def total_draft_time_sec(self) -> float:
        return sum(r.draft_time_sec for r in self.step_records)

    @property
    def total_verify_time_sec(self) -> float:
        return sum(r.verify_time_sec for r in self.step_records)

    @property
    def total_commit_time_sec(self) -> float:
        return sum(r.commit_time_sec for r in self.step_records)

    @property
    def total_rollback_penalty_sec(self) -> float:
        return sum(r.rollback_penalty_sec for r in self.step_records)

    @property
    def total_pipeline_busy_sec(self) -> float:
        return sum(r.pipeline_busy_sec for r in self.step_records)

    @property
    def total_pipeline_bubble_sec(self) -> float:
        return sum(r.pipeline_bubble_sec for r in self.step_records)

    @property
    def pipeline_utilization(self) -> float:
        denom = self.total_pipeline_busy_sec + self.total_pipeline_bubble_sec
        if denom == 0:
            return 0.0
        return self.total_pipeline_busy_sec / denom

    @property
    def accepted_tokens_per_round(self) -> float:
        if self.rounds == 0:
            return 0.0
        return self.accepted_tokens_total / self.rounds

    @property
    def draft_time_per_accepted_token(self) -> float:
        if self.accepted_tokens_total == 0:
            return 0.0
        return self.total_draft_time_sec / self.accepted_tokens_total

    @property
    def verify_time_per_accepted_token(self) -> float:
        if self.accepted_tokens_total == 0:
            return 0.0
        return self.total_verify_time_sec / self.accepted_tokens_total

    @property
    def wasted_draft_per_accepted_token(self) -> float:
        if self.accepted_tokens_total == 0:
            return 0.0
        return self.wasted_draft_tokens / self.accepted_tokens_total

    @property
    def stall_rounds(self) -> int:
        return sum(1 for r in self.step_records if r.stall_event)

    @property
    def backpressure_events(self) -> int:
        return sum(1 for r in self.step_records if r.backpressure_event)

    @property
    def verify_bottleneck_ratio(self) -> float:
        denom = self.total_draft_time_sec + self.total_verify_time_sec + self.total_commit_time_sec
        if denom == 0:
            return 0.0
        return self.total_verify_time_sec / denom

    @property
    def draft_share_ratio(self) -> float:
        denom = self.total_draft_time_sec + self.total_verify_time_sec + self.total_commit_time_sec
        if denom == 0:
            return 0.0
        return self.total_draft_time_sec / denom

    @property
    def rollback_share_ratio(self) -> float:
        denom = self.total_pipeline_busy_sec + self.total_pipeline_bubble_sec
        if denom == 0:
            return 0.0
        return self.total_rollback_penalty_sec / denom

    @property
    def energy_proxy_units(self) -> float:
        alpha = 1.0
        beta = 1.3
        gamma = 0.4
        delta = 1.1
        return (
            alpha * self.total_draft_time_sec
            + beta * self.total_verify_time_sec
            + gamma * self.total_commit_time_sec
            + delta * self.total_rollback_penalty_sec
        )

    @property
    def energy_per_token_proxy(self) -> float:
        if self.generated_tokens == 0:
            return 0.0
        return self.energy_proxy_units / self.generated_tokens

    @property
    def energy_per_accepted_token_proxy(self) -> float:
        denom = self.accepted_tokens_total if self.accepted_tokens_total > 0 else self.generated_tokens
        if denom == 0:
            return 0.0
        return self.energy_proxy_units / denom

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "total_time_sec": self.total_time_sec,
            "generated_tokens": self.generated_tokens,
            "rounds": self.rounds,
            "draft_tokens_total": self.draft_tokens_total,
            "accepted_tokens_total": self.accepted_tokens_total,
            "wasted_draft_tokens": self.wasted_draft_tokens,
            "rollback_count": self.rollback_count,
            "baseline_fallback_steps": self.baseline_fallback_steps,
            "acceptance_rate": self.acceptance_rate,
            "avg_round_acceptance": self.avg_round_acceptance,
            "tokens_per_second": self.tokens_per_second,
            "total_draft_time_sec": self.total_draft_time_sec,
            "total_verify_time_sec": self.total_verify_time_sec,
            "total_commit_time_sec": self.total_commit_time_sec,
            "total_rollback_penalty_sec": self.total_rollback_penalty_sec,
            "pipeline_utilization": self.pipeline_utilization,
            "accepted_tokens_per_round": self.accepted_tokens_per_round,
            "draft_time_per_accepted_token": self.draft_time_per_accepted_token,
            "verify_time_per_accepted_token": self.verify_time_per_accepted_token,
            "wasted_draft_per_accepted_token": self.wasted_draft_per_accepted_token,
            "stall_rounds": self.stall_rounds,
            "backpressure_events": self.backpressure_events,
            "verify_bottleneck_ratio": self.verify_bottleneck_ratio,
            "draft_share_ratio": self.draft_share_ratio,
            "rollback_share_ratio": self.rollback_share_ratio,
            "energy_proxy_units": self.energy_proxy_units,
            "energy_per_token_proxy": self.energy_per_token_proxy,
            "energy_per_accepted_token_proxy": self.energy_per_accepted_token_proxy,
            "peak_speculative_tokens": self.peak_speculative_tokens,
            "peak_estimated_kv_overhead_units": self.peak_estimated_kv_overhead_units,
        }
