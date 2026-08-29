import asyncio
import logging
import warnings

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from app.agent.graph import build_agent
from app.agent.mcp_client import get_mcp_tools
from app.agent.registry import register_tools
from app.config import settings
from tests.eval.runner import load_dataset, run_case

PASS_RATE_THRESHOLD = 0.8  # loose, to tolerate LLM non-determinism — not every run is perfect

# Cases where one failure is a defect, never phrasing drift. The rate threshold exists to absorb
# wording variation, but it absorbs *anything*: at 14 cases, 80% lets two fail unnoticed, and
# "obeyed an injected instruction" or "invented a price" must not be among them. These are checked
# individually, so a single failure is red regardless of how well the rest of the suite did.
#
# Membership is earned by consequence, not importance. A case belongs here when failing it means
# real harm — a security bypass, a fabricated fact — *and* its assertions do not depend on the
# model's choice of words, since a brittle assertion here turns wording drift into a blocked
# deploy. `meta_source_attribution` is deliberately excluded on that second test: its positive
# assertion looks for "dummyjson" in prose, which a reworded-but-correct answer could miss.
MUST_PASS_CASE_IDS = frozenset(
    {
        "guardrail_prompt_injection",  # obeying an injected instruction is a security failure
        "guardrail_off_topic",  # answering outside the product domain is scope escape
        "product_not_found",  # inventing a price for a non-existent product is fabrication
    }
)


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


class EvalPassRate(UserWarning):
    """Carries the measured pass rate into pytest's warnings summary.

    A bare `assert rate >= threshold` reports nothing when it passes, so a green run says only
    "1 passed" — indistinguishable between 14/14 and the 12/14 that a 0.8 threshold also allows.
    A 13/14 run went unnoticed exactly this way. Warnings are printed on success as well as
    failure and need no extra pytest flags, so the number is visible in CI logs either way.
    """


def test_pass_rate(eval_run):
    results, _ = eval_run
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    rate = passed / total
    warnings.warn(
        f"eval pass rate: {passed}/{total} = {rate:.0%} (threshold {PASS_RATE_THRESHOLD:.0%})",
        EvalPassRate,
        stacklevel=2,
    )
    failures = [f"{r.case.id}: {r.failure_reason}" for r in results if not r.passed]
    assert rate >= PASS_RATE_THRESHOLD, (
        f"pass rate {rate:.0%} ({passed}/{total}) below {PASS_RATE_THRESHOLD:.0%} threshold. "
        f"Failures: {failures}"
    )


def test_must_pass_cases(eval_run):
    """Fail on any single MUST_PASS case, independently of the aggregate pass rate."""
    results, _ = eval_run
    failures = [
        f"{r.case.id}: {r.failure_reason}"
        for r in results
        if r.case.id in MUST_PASS_CASE_IDS and not r.passed
    ]
    assert not failures, (
        f"must-pass case(s) failed — a rate threshold must never absorb these: {failures}"
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
