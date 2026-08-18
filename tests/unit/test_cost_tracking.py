import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from app.agent.context import AgentContext
from app.agent.middleware.cost_tracking import CostTrackingMiddleware, _estimate_cost
from app.config import settings


class _Runtime:
    def __init__(self, context):
        self.context = context


def _run(coro):
    return asyncio.run(coro)


def test_estimate_cost_uses_input_and_output_rates():
    """One million of each token type costs exactly the configured per-million pair."""
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
    expected = settings.llm_input_price_per_million_usd + settings.llm_output_price_per_million_usd
    assert _estimate_cost(usage) == pytest.approx(expected)


def test_estimate_cost_scales_below_one_million():
    """Guards the per-million divisor: rates are quoted per 1M tokens, not per token."""
    assert _estimate_cost({"input_tokens": 1_000_000, "output_tokens": 0}) == pytest.approx(
        settings.llm_input_price_per_million_usd
    )
    assert _estimate_cost({"input_tokens": 1, "output_tokens": 0}) == pytest.approx(
        settings.llm_input_price_per_million_usd / 1_000_000
    )


def test_estimate_cost_missing_keys_defaults_to_zero():
    assert _estimate_cost({}) == 0.0


def test_noop_without_context():
    mw = CostTrackingMiddleware()
    msg = AIMessage(
        content="hi", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    )
    state = {"messages": [msg]}
    with patch(
        "app.agent.middleware.cost_tracking.usage_tracker.record_usage", new_callable=AsyncMock
    ) as mock_record:
        _run(mw.aafter_model(state, runtime=_Runtime(context=None)))
        mock_record.assert_not_called()


def test_noop_without_user_id_in_context():
    mw = CostTrackingMiddleware()
    msg = AIMessage(
        content="hi", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    )
    state = {"messages": [msg]}
    with patch(
        "app.agent.middleware.cost_tracking.usage_tracker.record_usage", new_callable=AsyncMock
    ) as mock_record:
        _run(mw.aafter_model(state, runtime=_Runtime(context=AgentContext(user_id=None))))
        mock_record.assert_not_called()


def test_noop_without_usage_metadata():
    mw = CostTrackingMiddleware()
    state = {"messages": [AIMessage(content="hi")]}
    with patch(
        "app.agent.middleware.cost_tracking.usage_tracker.record_usage", new_callable=AsyncMock
    ) as mock_record:
        _run(mw.aafter_model(state, runtime=_Runtime(context=AgentContext(user_id=1))))
        mock_record.assert_not_called()


def test_records_usage_with_correct_cost_and_tokens():
    mw = CostTrackingMiddleware()
    msg = AIMessage(
        content="hi", usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
    )
    state = {"messages": [msg]}
    with patch(
        "app.agent.middleware.cost_tracking.usage_tracker.record_usage", new_callable=AsyncMock
    ) as mock_record:
        _run(mw.aafter_model(state, runtime=_Runtime(context=AgentContext(user_id=7))))
        mock_record.assert_awaited_once()
        _, kwargs = mock_record.call_args
        assert kwargs["user_id"] == 7
        assert kwargs["tokens"] == 150
        assert kwargs["cost"] == pytest.approx(
            (
                100 * settings.llm_input_price_per_million_usd
                + 50 * settings.llm_output_price_per_million_usd
            )
            / 1_000_000
        )
