import asyncio
import logging

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from app.agent.graph import build_agent
from app.agent.mcp_client import get_mcp_tools
from app.agent.registry import register_tools
from app.config import settings
from tests.eval.runner import load_dataset, run_case

PASS_RATE_THRESHOLD = 0.8  # loose, to tolerate LLM non-determinism — not every run is perfect


class _GroundednessCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture(scope="module")
def eval_run():
    # Per-test override that raises the usage budget so eval runs aren't blocked by the default
    # $0.00 safe-by-default cap.
    settings.cost_budget_usd_per_user_per_day = 100.0
    settings.max_llm_calls_per_minute_per_user = 100

    capture = _GroundednessCapture()
    groundedness_logger = logging.getLogger("app.agent.groundedness")
    groundedness_logger.addHandler(capture)
    groundedness_logger.setLevel(logging.WARNING)

    async def _run():
        register_tools(await get_mcp_tools())
        checkpointer = InMemorySaver()
        store = InMemoryStore()
        agent = build_agent(checkpointer=checkpointer, store=store)
        cases = load_dataset()
        return [await run_case(agent, case, thread_id=f"eval-{case.id}") for case in cases]

    try:
        results = asyncio.run(_run())
    finally:
        groundedness_logger.removeHandler(capture)

    return results, capture.records


def test_pass_rate(eval_run):
    results, _ = eval_run
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    rate = passed / total
    failures = [f"{r.case.id}: {r.failure_reason}" for r in results if not r.passed]
    assert rate >= PASS_RATE_THRESHOLD, (
        f"pass rate {rate:.0%} ({passed}/{total}) below {PASS_RATE_THRESHOLD:.0%} threshold. "
        f"Failures: {failures}"
    )


def test_no_groundedness_warnings(eval_run):
    """CI-time sanity check on top of the runtime logging-only behavior:
    the golden set's expected answers should never trip the groundedness heuristic."""
    _, groundedness_records = eval_run
    # Report the offending values, not just the message: every past failure here has been a
    # false positive in the heuristic, and the numbers are what identify which pattern is at
    # fault. Without them a failure says only "something was flagged".
    flagged = [
        f"{r.getMessage()} {sorted(getattr(r, 'ungrounded_numbers', []))}"
        for r in groundedness_records
    ]
    assert not flagged, (
        f"groundedness middleware flagged {len(flagged)} ungrounded claim(s) across the golden "
        f"set: {flagged}"
    )
