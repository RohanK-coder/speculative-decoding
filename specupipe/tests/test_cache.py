import os
import sys

# Ensure `core/*` absolute imports resolve by adding `specupipe/` to sys.path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.cache import KVCacheManager


def test_cache_commit_and_discard():
    cache = KVCacheManager(committed_length=10)

    cp = cache.checkpoint()
    assert cp.committed_length == 10

    cache.append_speculative_token()
    cache.append_speculative_token()
    assert cache.speculative_length == 2

    cache.commit_accepted_prefix(1)
    assert cache.committed_length == 11
    assert cache.speculative_length == 1

    cache.discard_remaining_speculative()
    assert cache.speculative_length == 0

    cache.commit_corrective_token()
    assert cache.committed_length == 12

