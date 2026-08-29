import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import ToolMessage

DATASET_PATH = Path(__file__).parent / "golden_dataset.jsonl"

# Typographic variants a model may legitimately produce for the same claim. Matching on the raw
# text made correct answers fail: "$1,999.99" does not contain "1999.99", and a curly apostrophe
# in "I don't have" does not contain the ASCII "don't have". Normalising here compares meaning
# rather than formatting — it is deliberately not a loosening of what the golden set asserts.
# Hyphens are folded to ASCII because NFKC does not do it: U+2011 NON-BREAKING HYPHEN decomposes
# to U+2010 HYPHEN, never to U+002D. En and em dashes are deliberately left alone — they separate
# clauses rather than join words, so folding them could join two values into one.
_PUNCTUATION_FOLD = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "‐": "-", "‑": "-"})
_THOUSANDS_SEPARATOR = re.compile(r"(?<=\d),(?=\d{3}\b)")


def normalize(text: str) -> str:
    """Fold formatting differences that do not change what an answer asserts.

    NFKC collapses the exotic spaces models emit (notably the narrow no-break space in "4.69 %");
    the rest folds smart quotes, hyphen variants, thousands separators and case.
    """
    text = unicodedata.normalize("NFKC", text).translate(_PUNCTUATION_FOLD)
    return _THOUSANDS_SEPARATOR.sub("", text).lower()


@dataclass
class EvalCase:
    """One golden-set case.

    `expect_tools` distinguishes three states: a list of names requires them, `[]` requires that
    *no* tool ran, and omitting it (None) means the case does not care. The third exists because
    some behaviours are about what the model *says*, and forcing a tool expectation on those would
    make the gate flaky over a choice that does not affect correctness.
    """

    id: str
    question: str
    expect_any_substring: list[str]
    expect_tools: list[str] | None = None
    # Substrings that must NOT appear. Some requirements are only expressible as negatives — the
    # answer must not name an internal tool, must not invent a link — and "contains one of" cannot
    # express them.
    expect_no_substring: list[str] | None = None
    # At least one of these must have run. Use when the requirement is "it consulted the catalogue"
    # rather than "it used this specific tool": `get_product` and `search_products` both resolve
    # through the same upstream search, so pinning one asserts an implementation detail that can
    # flip on a prompt edit without changing what the user sees.
    expect_any_tool: list[str] | None = None


@dataclass
class EvalResult:
    case: EvalCase
    passed: bool
    answer: str
    tool_calls: list[str]
    failure_reason: str | None = None


def load_dataset() -> list[EvalCase]:
    cases = []
    with DATASET_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(EvalCase(**json.loads(line)))
    return cases


async def run_case(agent, case: EvalCase, thread_id: str) -> EvalResult:
    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": case.question}]},
        config={"configurable": {"thread_id": thread_id}},
        context={"user_id": 0},
    )
    tool_calls = [m.name for m in response["messages"] if isinstance(m, ToolMessage) and m.name]
    answer = str(response["messages"][-1].content)

    if case.expect_tools is not None:
        if case.expect_tools == [] and tool_calls:
            return EvalResult(
                case, False, answer, tool_calls, f"expected no tools, got {tool_calls}"
            )

        missing_tools = [t for t in case.expect_tools if t not in tool_calls]
        if missing_tools:
            return EvalResult(
                case, False, answer, tool_calls, f"missing expected tools: {missing_tools}"
            )

    if case.expect_any_tool and not any(t in tool_calls for t in case.expect_any_tool):
        return EvalResult(
            case,
            False,
            answer,
            tool_calls,
            f"none of {case.expect_any_tool} ran; got {tool_calls}",
        )

    if case.expect_any_substring:
        answer_normalized = normalize(answer)
        if not any(normalize(s) in answer_normalized for s in case.expect_any_substring):
            return EvalResult(
                case,
                False,
                answer,
                tool_calls,
                f"none of {case.expect_any_substring} found in answer",
            )

    if case.expect_no_substring:
        answer_normalized = normalize(answer)
        forbidden = [s for s in case.expect_no_substring if normalize(s) in answer_normalized]
        if forbidden:
            return EvalResult(
                case, False, answer, tool_calls, f"answer contains forbidden text: {forbidden}"
            )

    return EvalResult(case, True, answer, tool_calls, None)
