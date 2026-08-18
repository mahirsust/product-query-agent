"""Tests for PII redaction.

The detectors are langchain built-ins; what is tested here is the composition — that chaining
several redactors inside one hook still redacts every type, and reports changes in the shape the
graph expects.
"""

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.middleware.pii import PIIRedactionMiddleware


def _redact(*messages) -> dict | None:
    return PIIRedactionMiddleware().before_model({"messages": list(messages)}, None)


def test_email_is_redacted():
    update = _redact(HumanMessage(content="reach me at alice@example.com", id="m1"))
    assert "alice@example.com" not in update["messages"][0].content


def test_credit_card_is_redacted():
    update = _redact(HumanMessage(content="card 4111 1111 1111 1111", id="m1"))
    assert "4111" not in update["messages"][0].content


def test_phone_is_redacted():
    update = _redact(HumanMessage(content="call 555-123-4567", id="m1"))
    assert "555-123-4567" not in update["messages"][0].content


def test_all_types_redacted_in_one_message():
    """The composition's real risk: a later redactor seeing the unredacted original and undoing
    an earlier one's work."""
    update = _redact(
        HumanMessage(
            content="alice@example.com, 4111 1111 1111 1111, 555-123-4567",
            id="m1",
        )
    )
    content = update["messages"][0].content
    assert "alice@example.com" not in content
    assert "4111" not in content
    assert "555-123-4567" not in content


def test_clean_input_reports_no_change():
    """Returning None rather than the unchanged list keeps the graph from writing a pointless
    state update."""
    assert _redact(HumanMessage(content="what is the price of the macbook?", id="m1")) is None


def test_only_changed_messages_are_returned():
    update = _redact(
        HumanMessage(content="hello there", id="m1"),
        AIMessage(content="hi", id="m2"),
        HumanMessage(content="mail alice@example.com", id="m3"),
    )
    assert [m.id for m in update["messages"]] == ["m3"]


def test_redaction_preserves_message_identity():
    """Ids must survive so the reducer replaces the message instead of appending a duplicate."""
    update = _redact(HumanMessage(content="mail alice@example.com", id="m1"))
    assert update["messages"][0].id == "m1"
    assert isinstance(update["messages"][0], HumanMessage)
