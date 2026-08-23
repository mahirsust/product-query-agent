"""Streamlit chat UI.

Streamlit re-runs this module top to bottom on every interaction, so all state lives in
`st.session_state` and the presence of a token is what decides which screen renders.
"""

import uuid

import requests
import streamlit as st
from api_client import ApiError, login, post_chat, signup

st.set_page_config(page_title="Product Query Agent", page_icon="🛍️")

# Every product fact the agent states comes from the public DummyJSON catalogue. Surfacing this
# per answer — rather than as a page footer — keeps the attribution tied to the turn it describes.
DATA_SOURCE_NAME = "DummyJSON"
DATA_SOURCE_URL = "https://dummyjson.com/docs/products"

# Tool names as returned by the API, mapped to what they actually fetched.
_TOOL_LABELS = {
    "get_product": "product details",
    "search_products": "catalogue search",
    "remember_preference": "saved preference",
}


def _render_source(tool_calls: list[str]) -> None:
    """Caption an answer with where its data came from.

    Only rendered when a catalogue tool actually ran: cached answers and ones drawn from
    conversation context return no tool calls, and crediting the source there would imply a
    lookup that did not happen.
    """
    catalogue_tools = [t for t in tool_calls if t in ("get_product", "search_products")]
    if not catalogue_tools:
        return
    used = ", ".join(dict.fromkeys(_TOOL_LABELS.get(t, t) for t in catalogue_tools))
    st.caption(f"Source: [{DATA_SOURCE_NAME} product data]({DATA_SOURCE_URL}) — {used}")


def _reset_session() -> None:
    """Clear the signed-in state, returning the app to the auth screen."""
    for key in ("token", "username", "thread_id", "messages"):
        st.session_state.pop(key, None)


def _ensure_session_defaults() -> None:
    """Give this browser session its own conversation thread and transcript."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())


def _render_auth_screen() -> None:
    """Render the login/signup gate. Both paths return a token, so either signs the user in."""
    st.title("Product Query Agent")
    st.caption(
        f"A shopping assistant answering from the "
        f"[{DATA_SOURCE_NAME}]({DATA_SOURCE_URL}) product catalogue. Log in to start."
    )
    tab_login, tab_signup = st.tabs(["Log in", "Sign up"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log in")
        if submitted:
            try:
                result = login(username, password)
                st.session_state.token = result["access_token"]
                st.session_state.username = username
                st.rerun()
            except ApiError as exc:
                if exc.status_code == 401:
                    st.error("Invalid username or password.")
                else:
                    st.error(f"Login failed: {exc.detail}")
            except requests.exceptions.RequestException:
                st.error("Could not reach the backend. Is it running?")

    with tab_signup:
        with st.form("signup_form"):
            username = st.text_input("Username", key="signup_username")
            password = st.text_input("Password", type="password", key="signup_password")
            submitted = st.form_submit_button("Sign up")
        if submitted:
            try:
                result = signup(username, password)
                st.session_state.token = result["access_token"]
                st.session_state.username = username
                st.rerun()
            except ApiError as exc:
                if exc.status_code == 409:
                    st.error("That username is already taken.")
                else:
                    st.error(f"Signup failed: {exc.detail}")
            except requests.exceptions.RequestException:
                st.error("Could not reach the backend. Is it running?")


def _render_chat_screen() -> None:
    """Render the transcript and handle one new question per rerun."""
    _ensure_session_defaults()

    with st.sidebar:
        st.write(f"Logged in as **{st.session_state.username}**")
        if st.button("Log out", use_container_width=True):
            _reset_session()
            st.rerun()
        if st.button("New conversation", use_container_width=True):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.subheader("Data source")
        st.markdown(
            f"Product details, prices and reviews come from the public "
            f"[{DATA_SOURCE_NAME} catalogue]({DATA_SOURCE_URL}).\n\n"
            "It is demo data, so prices and stock are not real."
        )

    st.title("Product Query Agent")
    st.caption(
        f"Ask about prices, reviews and stock — answered from the "
        f"[{DATA_SOURCE_NAME}]({DATA_SOURCE_URL}) product catalogue."
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                _render_source(message.get("tool_calls", []))

    if not st.session_state.messages:
        st.info(
            "Try: *what is the price of the macbook?* · *what are its reviews?* · "
            "*what products do you have?*"
        )

    question = st.chat_input("Ask about product prices, reviews, or browse the catalog...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = post_chat(st.session_state.token, question, st.session_state.thread_id)
                answer = result["answer"]
                tool_calls = result.get("tool_calls", [])
                st.write(answer)
                _render_source(tool_calls)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "tool_calls": tool_calls}
                )
            except ApiError as exc:
                # The backend distinguishes "slow down" (429) from "out of budget" (402); showing
                # the same message for both would misdirect the user about whether to retry.
                if exc.status_code == 401:
                    st.error("Your session has expired. Please log in again.")
                    _reset_session()
                elif exc.status_code == 429:
                    st.warning(
                        "You're sending requests too quickly — slow down and try again in a moment."
                    )
                elif exc.status_code == 402:
                    st.warning("Daily usage budget exhausted for this account. Try again tomorrow.")
                elif exc.status_code == 503:
                    # Distinct from 402 on purpose: nothing the user did caused this.
                    st.warning(
                        "The service has reached its shared daily capacity across all users. "
                        "This isn't a limit on your account — please try again tomorrow."
                    )
                else:
                    st.error(f"Something went wrong: {exc.detail}")
            except requests.exceptions.RequestException:
                st.error("Could not reach the backend. Is it running?")


def main() -> None:
    """Route to the auth or chat screen based on whether this session holds a token."""
    if "token" not in st.session_state:
        _render_auth_screen()
    else:
        _render_chat_screen()


# Called at import: Streamlit executes the module itself as the app entry point.
main()
