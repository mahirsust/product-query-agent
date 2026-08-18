import asyncio

from app.cache import response_cache

USER_A = 1
USER_B = 2


def _run(coro):
    return asyncio.run(coro)


def test_set_then_get_roundtrip(fake_redis):
    _run(response_cache.set("What is the price of the macbook?", "It's $1999.99.", USER_A))
    assert _run(response_cache.get("What is the price of the macbook?", USER_A)) == "It's $1999.99."


def test_get_miss_returns_none(fake_redis):
    assert _run(response_cache.get("never asked this before", USER_A)) is None


def test_normalization_is_case_and_whitespace_insensitive(fake_redis):
    _run(response_cache.set("  What Is The Price  ", "answer", USER_A))
    assert _run(response_cache.get("what is the price", USER_A)) == "answer"
    assert _run(response_cache.get("WHAT   IS   THE   PRICE", USER_A)) == "answer"


def test_different_questions_do_not_collide(fake_redis):
    _run(response_cache.set("question a", "answer a", USER_A))
    _run(response_cache.set("question b", "answer b", USER_A))
    assert _run(response_cache.get("question a", USER_A)) == "answer a"
    assert _run(response_cache.get("question b", USER_A)) == "answer b"


def test_custom_ttl_is_passed_through(fake_redis):
    _run(response_cache.set("q", "a", USER_A, ttl=1))
    assert _run(response_cache.get("q", USER_A)) == "a"


def test_cached_answer_is_not_shared_across_users(fake_redis):
    """Answers are personalized (long_term_memory injects the asking user's stored preferences
    into the system prompt), so a question-only cache key leaked one user's answer to everyone
    else asking the same question."""
    _run(response_cache.set("what laptops do you have", "Given your $1000 budget...", USER_A))
    assert _run(response_cache.get("what laptops do you have", USER_A)) is not None
    assert _run(response_cache.get("what laptops do you have", USER_B)) is None


def test_raw_question_text_is_not_stored_in_the_redis_key(fake_redis):
    """The cache is keyed off the unredacted request payload, so PII typed into a question would
    otherwise sit in a Redis key for the whole TTL — outside middleware/pii.py's redaction."""
    _run(response_cache.set("my email is alice@example.com", "answer", USER_A))
    keys = _run(fake_redis.keys("*"))
    assert keys, "expected the entry to be written"
    for key in keys:
        assert "alice@example.com" not in key
        assert "email" not in key
