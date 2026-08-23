import asyncio
import logging

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.store.memory import InMemoryStore

from app.agent.context import AgentContext
from app.agent.middleware.groundedness import GroundednessMiddleware


class _Runtime:
    def __init__(self, context, store):
        self.context = context
        self.store = store


def _run(coro):
    return asyncio.run(coro)


def _capture_warnings():
    records: list[logging.LogRecord] = []

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Handler()
    logger = logging.getLogger("app.agent.groundedness")
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    return logger, handler, records


def _check(question: str, tool_content: str, answer: str, preferences: list[str] | None = None):
    """Runs GroundednessMiddleware.aafter_model against a constructed single-tool-call turn and
    returns the list of warning LogRecords it emitted."""
    mw = GroundednessMiddleware()
    logger, handler, records = _capture_warnings()

    store = InMemoryStore()
    if preferences:
        _run(store.aput(("users", "0", "memory"), "profile", {"preferences": preferences}))

    state = {
        "messages": [
            HumanMessage(content=question),
            ToolMessage(content=tool_content, tool_call_id="1", name="tool"),
            AIMessage(content=answer),
        ],
        "turn_start_index": 0,
    }
    runtime = _Runtime(AgentContext(user_id=0), store)
    _run(mw.aafter_model(state, runtime=runtime))
    logger.removeHandler(handler)
    return records


def test_grounded_price_no_warning():
    records = _check(
        "what is the price of the gucci bloom perfume?",
        str({"price": 79.99, "rating": 2.74}),
        "The price is $79.99.",
    )
    assert not records


def test_fabricated_price_flags_warning():
    records = _check(
        "what is the price of the gucci bloom perfume?",
        str({"price": 79.99, "rating": 2.74}),
        "The price is $54321.",
    )
    assert len(records) == 1
    assert records[0].__dict__["ungrounded_numbers"] == ["54321"]


def test_padded_decimal_not_flagged():
    """A rating stored as 3.8 and written as "3.80" is the same value, not a fabricated one."""
    records = _check(
        "how is the macbook rated?",
        str({"price": 1999.99, "rating": 3.8}),
        "It is rated 3.80 by reviewers.",
    )
    assert not records


def test_padded_decimal_in_tool_output_not_flagged():
    """The reverse direction: tool output padded, answer terse."""
    records = _check(
        "how is the macbook rated?",
        str({"price": 1999.99, "rating": "3.80"}),
        "It is rated 3.8 by reviewers.",
    )
    assert not records


def test_thousands_separator_not_flagged():
    records = _check(
        "what is the price of the macbook?",
        str({"price": 1999.99}),
        "The MacBook Pro is listed for $1,999.99.",
    )
    assert not records


def test_trailing_zero_stripping_does_not_merge_integers():
    """Only zeros after a decimal point are dropped: 100 must not collapse to 1 or 10."""
    records = _check(
        "how many are in stock?",
        str({"stock": 10}),
        "There are 100 units in stock.",
    )
    assert len(records) == 1
    assert records[0].__dict__["ungrounded_numbers"] == ["100"]


def test_ordered_list_markers_not_flagged():
    records = _check(
        "what fragrances do you have under 100 dollars?",
        str(
            [
                {"name": "Calvin Klein CK One", "price": 49.99},
                {"name": "Dior J'adore", "price": 89.99},
            ]
        ),
        "1. Calvin Klein CK One - $49.99\n2. Dior J'adore - $89.99\n",
    )
    assert not records


def test_rating_scale_slash_phrasing_not_flagged():
    records = _check(
        "what is the price of the gucci bloom perfume?",
        str({"price": 79.99, "rating": 2.74}),
        "It's $79.99, rated 2.74/5.",
    )
    assert not records


def test_rating_scale_out_of_phrasing_not_flagged():
    records = _check(
        "what is the price of the gucci bloom perfume?",
        str({"price": 79.99, "rating": 2.74}),
        "It's $79.99. It has a rating of 2.74 out of 5 stars.",
    )
    assert not records


def test_echoed_question_threshold_not_flagged():
    records = _check(
        "what laptops do you have under 1500 dollars?",
        str([{"name": "Some Laptop", "price": 1399.99}]),
        "We have laptops under $1500, including Some Laptop at $1399.99.",
    )
    assert not records


def test_remembered_preference_grounds_claim():
    records = _check(
        "search for laptops",
        str([{"name": "Some Laptop", "price": 899.99}]),
        "Here are some laptops. Since you have a budget preference under $1000, I can filter further.",
        preferences=["budget laptops under $1000"],
    )
    assert not records


def test_remembered_preference_does_not_hide_a_real_hallucination():
    records = _check(
        "search for laptops",
        str([{"name": "Some Laptop", "price": 899.99}]),
        "This laptop costs $54321 and fits your $1000 budget.",
        preferences=["budget laptops under $1000"],
    )
    assert len(records) == 1
    assert "54321" in records[0].__dict__["ungrounded_numbers"]
    assert "1000" not in records[0].__dict__["ungrounded_numbers"]


def test_model_retry_failure_message_skipped():
    """ModelRetryMiddleware's default on_failure formatter produces a message like this when
    retries are exhausted (e.g. a real rate limit) — it's an infrastructure error, full of stray
    numbers (status codes, token counts), not a product claim to ground."""
    records = _check(
        "what is the price of the macbook?",
        str({"price": 1999.99, "rating": 3.65}),
        "Model call failed after 3 attempts with RateLimitError: Error code: 429 - "
        "{'error': {'message': 'Rate limit reached ... Limit 100000, Used 99516 ...'}}",
    )
    assert not records


def test_discounted_price_not_flagged():
    """The exact CI failure: 79.99 less 14.39% is 68.48, arithmetic over two grounded numbers."""
    records = _check(
        "what is the price of the gucci bloom perfume?",
        str({"price": 79.99, "discountPercentage": 14.39, "rating": 4.71}),
        "The Gucci Bloom is $79.99, or $68.48 with the 14.39% discount applied.",
    )
    assert not records


def test_discounted_price_rounded_to_whole_currency_not_flagged():
    records = _check(
        "how much is it after the discount?",
        str({"price": 100.0, "discountPercentage": 32.0}),
        "That works out to $68 after the discount.",
    )
    assert not records


def test_derived_values_do_not_ground_an_unrelated_number():
    """The relaxation must stay narrow: a fabricated price is still caught."""
    records = _check(
        "what is the price of the gucci bloom perfume?",
        str({"price": 79.99, "discountPercentage": 14.39}),
        "The Gucci Bloom is $54321.",
    )
    assert len(records) == 1
    assert records[0].__dict__["ungrounded_numbers"] == ["54321"]


def test_non_percentage_operands_do_not_create_derived_values():
    """Only percentage-shaped numbers are applied, so two large numbers cannot manufacture a
    third that then launders a hallucination."""
    records = _check(
        "how many are in stock?",
        str({"price": 1999.99, "stock": 150}),
        "There are 1200 units in stock.",
    )
    assert len(records) == 1
    assert records[0].__dict__["ungrounded_numbers"] == ["1200"]
