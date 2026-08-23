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


def _generous_per_user(monkeypatch) -> None:
    """Raise every per-user cap so only the account-wide one can trip."""
    monkeypatch.setattr(settings, "cost_budget_usd_per_user_per_day", 10.0)
    monkeypatch.setattr(settings, "max_llm_calls_per_user_per_day", 10_000)
    monkeypatch.setattr(settings, "max_llm_calls_per_minute_per_user", 10_000)
    monkeypatch.setattr(settings, "max_tokens_per_user_per_day", 10_000_000)
    monkeypatch.setattr(settings, "max_tokens_per_minute_per_user", 10_000_000)


def test_account_cap_trips_on_tokens_spent_by_other_users(fake_redis, monkeypatch):
    """The gap this cap exists to close: several users, each within their own daily limit,
    collectively exhausting the provider's account-wide quota."""
    _generous_per_user(monkeypatch)
    monkeypatch.setattr(settings, "max_tokens_all_users_per_day", 1_000)
    for user_id in (1, 2, 3, 4):
        _run(usage_tracker.record_usage(user_id=user_id, cost=0.0, tokens=250, calls=1))

    # A fifth user who has spent nothing at all is still refused.
    assert _run(usage_tracker.check_and_reserve(user_id=5)) == UsageCheckResult.ACCOUNT_EXHAUSTED


def test_account_cap_allows_traffic_below_the_ceiling(fake_redis, monkeypatch):
    _generous_per_user(monkeypatch)
    monkeypatch.setattr(settings, "max_tokens_all_users_per_day", 1_000)
    _run(usage_tracker.record_usage(user_id=1, cost=0.0, tokens=999, calls=1))
    assert _run(usage_tracker.check_and_reserve(user_id=2)) == UsageCheckResult.OK


def test_account_cap_of_zero_disables_the_check(fake_redis, monkeypatch):
    """0 must mean "off", not "reject everything" — otherwise a misconfiguration bricks the app."""
    _generous_per_user(monkeypatch)
    monkeypatch.setattr(settings, "max_tokens_all_users_per_day", 0)
    _run(usage_tracker.record_usage(user_id=1, cost=0.0, tokens=999_999, calls=1))
    assert _run(usage_tracker.check_and_reserve(user_id=2)) == UsageCheckResult.OK


def test_per_user_limit_reported_before_account_limit(fake_redis, monkeypatch):
    """A user over their own share is told that, rather than blaming shared capacity."""
    _generous_per_user(monkeypatch)
    monkeypatch.setattr(settings, "max_tokens_per_user_per_day", 100)
    monkeypatch.setattr(settings, "max_tokens_all_users_per_day", 100)
    _run(usage_tracker.record_usage(user_id=1, cost=0.0, tokens=500, calls=1))
    assert _run(usage_tracker.check_and_reserve(user_id=1)) == UsageCheckResult.BUDGET_EXCEEDED


def test_account_total_is_recorded_even_when_cap_disabled(fake_redis, monkeypatch):
    """Usage must accumulate while the cap is off, so enabling it later has real data."""
    _generous_per_user(monkeypatch)
    monkeypatch.setattr(settings, "max_tokens_all_users_per_day", 0)
    _run(usage_tracker.record_usage(user_id=1, cost=0.0, tokens=400, calls=1))
    _run(usage_tracker.record_usage(user_id=2, cost=0.0, tokens=400, calls=1))

    monkeypatch.setattr(settings, "max_tokens_all_users_per_day", 800)
    assert _run(usage_tracker.check_and_reserve(user_id=3)) == UsageCheckResult.ACCOUNT_EXHAUSTED
