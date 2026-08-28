"""Tests for the content-addressed response cache."""

from vlm_evaluation_harness.cache import ResponseCache, response_key


def test_response_key_is_stable():
    k1 = response_key("mock:m1", "hello", None, ["h1"], {"temperature": 0.0})
    k2 = response_key("mock:m1", "hello", None, ["h1"], {"temperature": 0.0})
    assert k1 == k2


def test_response_key_changes_with_prompt():
    k1 = response_key("mock:m1", "hello", None, [], {})
    k2 = response_key("mock:m1", "goodbye", None, [], {})
    assert k1 != k2


def test_response_key_changes_with_params():
    k1 = response_key("mock:m1", "hello", None, [], {"temperature": 0.0})
    k2 = response_key("mock:m1", "hello", None, [], {"temperature": 0.7})
    assert k1 != k2


def test_cache_roundtrip(tmp_path):
    cache = ResponseCache(tmp_path / "cache.sqlite")
    key = response_key("mock:m1", "hello", None, [], {})
    assert cache.get(key) is None
    cache.put(key, "mock:m1", {"text": "hi"})
    assert cache.get(key) == {"text": "hi"}
    cache.close()


def test_cache_stats_track_hits_and_misses(tmp_path):
    cache = ResponseCache(tmp_path / "cache.sqlite")
    key = response_key("mock:m1", "hello", None, [], {})
    cache.get(key)  # miss
    cache.put(key, "mock:m1", {"text": "hi"})
    cache.get(key)  # hit
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1
    cache.close()


def test_cache_disabled_never_hits(tmp_path):
    cache = ResponseCache(tmp_path / "cache.sqlite", enabled=False)
    key = response_key("mock:m1", "hello", None, [], {})
    cache.put(key, "mock:m1", {"text": "hi"})
    assert cache.get(key) is None


def test_cache_clear_by_model(tmp_path):
    cache = ResponseCache(tmp_path / "cache.sqlite")
    k1 = response_key("mock:m1", "a", None, [], {})
    k2 = response_key("mock:m2", "a", None, [], {})
    cache.put(k1, "mock:m1", {"text": "1"})
    cache.put(k2, "mock:m2", {"text": "2"})
    removed = cache.clear(model_id="mock:m1")
    assert removed == 1
    assert cache.get(k1) is None
    assert cache.get(k2) is not None
    cache.close()
