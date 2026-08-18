"""The provider's raw exception text used to become the user-visible answer. A real Groq
rate-limit error carries the account's organization id, model name, service tier, quota ceiling
and current usage — all of which were returned verbatim in the /chat response body."""

from app.agent.graph import _generic_model_failure
from app.agent.prompts import MODEL_FAILURE_MESSAGE

_REAL_GROQ_ERROR = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
    "`llama-3.3-70b-versatile` in organization `org_01kys565zefb089ve2w6dq4y83` service tier "
    "`on_demand` on tokens per day (TPD): Limit 100000, Used 99527, Requested 783.', "
    "'code': 'rate_limit_exceeded'}}"
)


def test_provider_details_are_not_returned_to_the_user():
    message = _generic_model_failure(RuntimeError(_REAL_GROQ_ERROR))

    assert message == MODEL_FAILURE_MESSAGE
    for leaked in ("org_01kys565zefb089ve2w6dq4y83", "llama-3.3", "on_demand", "100000", "429"):
        assert leaked not in message


def test_underlying_error_is_still_logged_for_operators(caplog):
    with caplog.at_level("ERROR", logger="app.agent.graph"):
        _generic_model_failure(RuntimeError(_REAL_GROQ_ERROR))

    assert any("model call failed" in r.getMessage() for r in caplog.records)
