import asyncio

from app.cache import usage_tracker
from app.cache.usage_tracker import UsageCheckResult
from app.config import settings


def _run(coro):
    return asyncio.run(coro)


def test_zero_dollar_budget_blocks_immediately(fake_redis, monkeypatch):
    """Set the budget explicitly rather than relying on the shipped default leaking through the
    ambient `.env`: this asserts the *behaviour* of a $0 cap. That the shipped default IS $0 is a
    separate, env-isolated assertion in test_config.py — previously this test conflated the two
    and silently started passing for the wrong reason once a local .env set a real budget."""
    monkeypatch.setattr(settings, "cost_budget_usd_per_user_per_day", 0.0)
    assert _run(usage_tracker.check_and_reserve(user_id=1)) == UsageCheckResult.BUDGET_EXCEEDED


def test_ok_when_budget_raised_and_under_all_other_limits(fake_redis, monkeypatch):
    monkeypatch.setattr(settings, "cost_budget_usd_per_user_per_day", 10.0)
    assert _run(usage_tracker.check_and_reserve(user_id=1)) == UsageCheckResult.OK


def test_rate_limited_on_calls_per_minute(fake_redis, monkeypatch):
    monkeypatch.setattr(settings, "cost_budget_usd_per_user_per_day", 10.0)
    monkeypatch.setattr(settings, "max_llm_calls_per_minute_per_user", 2)
    _run(usage_tracker.record_usage(user_id=1, cost=0.0, tokens=0, calls=2))
    assert _run(usage_tracker.check_and_reserve(user_id=1)) == UsageCheckResult.RATE_LIMITED


def test_rate_limited_on_tokens_per_minute(fake_redis, monkeypatch):
    monkeypatch.setattr(settings, "cost_budget_usd_per_user_per_day", 10.0)
    monkeypatch.setattr(settings, "max_tokens_per_minute_per_user", 100)
    _run(usage_tracker.record_usage(user_id=1, cost=0.0, tokens=150, calls=1))
    assert _run(usage_tracker.check_and_reserve(user_id=1)) == UsageCheckResult.RATE_LIMITED


def test_budget_exceeded_on_daily_cost_cap(fake_redis, monkeypatch):
    monkeypatch.setattr(settings, "cost_budget_usd_per_user_per_day", 1.0)
    _run(usage_tracker.record_usage(user_id=1, cost=1.5, tokens=0, calls=1))
    assert _run(usage_tracker.check_and_reserve(user_id=1)) == UsageCheckResult.BUDGET_EXCEEDED


def test_budget_exceeded_on_daily_call_cap(fake_redis, monkeypatch):
    monkeypatch.setattr(settings, "cost_budget_usd_per_user_per_day", 10.0)
    monkeypatch.setattr(settings, "max_llm_calls_per_user_per_day", 1)
    _run(usage_tracker.record_usage(user_id=1, cost=0.0, tokens=0, calls=1))
    assert _run(usage_tracker.check_and_reserve(user_id=1)) == UsageCheckResult.BUDGET_EXCEEDED


def test_budget_exceeded_on_daily_token_cap(fake_redis, monkeypatch):
    monkeypatch.setattr(settings, "cost_budget_usd_per_user_per_day", 10.0)
    monkeypatch.setattr(settings, "max_tokens_per_user_per_day", 100)
    _run(usage_tracker.record_usage(user_id=1, cost=0.0, tokens=200, calls=1))
    assert _run(usage_tracker.check_and_reserve(user_id=1)) == UsageCheckResult.BUDGET_EXCEEDED


def test_different_users_are_isolated(fake_redis, monkeypatch):
    monkeypatch.setattr(settings, "cost_budget_usd_per_user_per_day", 10.0)
    monkeypatch.setattr(settings, "max_llm_calls_per_user_per_day", 1)
    _run(usage_tracker.record_usage(user_id=1, cost=0.0, tokens=0, calls=1))
    assert _run(usage_tracker.check_and_reserve(user_id=1)) == UsageCheckResult.BUDGET_EXCEEDED
    assert _run(usage_tracker.check_and_reserve(user_id=2)) == UsageCheckResult.OK


def test_record_usage_accumulates_across_calls(fake_redis, monkeypatch):
    monkeypatch.setattr(settings, "cost_budget_usd_per_user_per_day", 10.0)
    _run(usage_tracker.record_usage(user_id=5, cost=1.5, tokens=100, calls=1))
    _run(usage_tracker.record_usage(user_id=5, cost=2.5, tokens=50, calls=1))
    monkeypatch.setattr(settings, "cost_budget_usd_per_user_per_day", 3.9)
    assert _run(usage_tracker.check_and_reserve(user_id=5)) == UsageCheckResult.BUDGET_EXCEEDED
