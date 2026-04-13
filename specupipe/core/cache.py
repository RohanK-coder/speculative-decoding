from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CacheCheckpoint:
    committed_length: int
    speculative_length: int


@dataclass
class CacheEvent:
    event: str
    committed_length: int
    speculative_length: int


@dataclass
class KVCacheManager:
    committed_length: int = 0
    speculative_length: int = 0
    history: list[CacheEvent] = field(default_factory=list)

    def checkpoint(self) -> CacheCheckpoint:
        cp = CacheCheckpoint(
            committed_length=self.committed_length,
            speculative_length=self.speculative_length,
        )
        self.history.append(
            CacheEvent(
                event="checkpoint",
                committed_length=self.committed_length,
                speculative_length=self.speculative_length,
            )
        )
        return cp

    def append_speculative_token(self):
        self.speculative_length += 1
        self.history.append(
            CacheEvent(
                event="append_speculative",
                committed_length=self.committed_length,
                speculative_length=self.speculative_length,
            )
        )

    def commit_accepted_prefix(self, accepted_count: int):
        if accepted_count < 0 or accepted_count > self.speculative_length:
            raise ValueError("Invalid accepted_count")

        self.committed_length += accepted_count
        self.speculative_length -= accepted_count
        self.history.append(
            CacheEvent(
                event="commit_prefix",
                committed_length=self.committed_length,
                speculative_length=self.speculative_length,
            )
        )

    def discard_remaining_speculative(self):
        self.speculative_length = 0
        self.history.append(
            CacheEvent(
                event="discard_speculative",
                committed_length=self.committed_length,
                speculative_length=self.speculative_length,
            )
        )

    def commit_corrective_token(self):
        self.committed_length += 1
        self.history.append(
            CacheEvent(
                event="commit_corrective",
                committed_length=self.committed_length,
                speculative_length=self.speculative_length,
            )
        )

    def reset_to_checkpoint(self, checkpoint: CacheCheckpoint):
        self.committed_length = checkpoint.committed_length
        self.speculative_length = checkpoint.speculative_length
        self.history.append(
            CacheEvent(
                event="reset_to_checkpoint",
                committed_length=self.committed_length,
                speculative_length=self.speculative_length,
            )
        )

