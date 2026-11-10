"""The extraction path must work for a memory provider, which is an *exclusive* plugin.

A memory provider is activated by ``memory.provider``, not by the plugin manager, so the
host never reports it as ``enabled``. ``issue_side_question_token`` gates on exactly that
flag, so a provider that only knows how to use ``ctx.side_question`` can never extract a
fact — and never logs a reason, because the no-token branch was a silent defer.

These tests pin both halves: the fallback runs, and the no-op case says so in the log.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import _hwiki_path  # noqa: F401,E402  (registers the repo root as package `hwiki`)

from hwiki.facts_provider import FactsEngine


class _Result:
    def __init__(self, text):
        self.text = text


class _CtxWithLlm:
    """Exclusive-plugin context: no usable side-question token, but ``ctx.llm`` exists."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = []
        self.llm = self

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return _Result(json.dumps(self.payload))

    def side_question(self, token, prompt, timeout=25.0):  # pragma: no cover - must not run
        raise AssertionError("side_question must not be used when no token was issued")


class _CtxWithNothing:
    llm = None


FACTS = {"facts": [
    {"kind": "preference", "claim": "prefers terse replies", "evidence": "asked for brevity"},
]}


@pytest.fixture()
def engine(tmp_path):
    vault = tmp_path / "vault"
    (vault / "Hermes" / "Memory" / "Auto").mkdir(parents=True)
    # Explicit opt-in: extraction is off unless the user turns it on, so every test
    # that expects it to run has to say so.
    eng = FactsEngine(
        tmp_path / "home", vault, Path("Hermes/Memory/Auto"),
        config={"extract_enabled": True})
    eng.store()  # migrate
    return eng


def _statuses(engine):
    conn = sqlite3.connect(engine.store().database)
    try:
        return [(r[0], r[1], r[2]) for r in conn.execute(
            "select status, kind, claim from proposals order by rowid")]
    finally:
        conn.close()


def test_extraction_runs_without_a_side_question_token(engine):
    """The regression: an exclusive plugin gets no token, and used to extract nothing."""
    ctx = _CtxWithLlm(FACTS)
    handler = engine.hook_handler(ctx)

    handler(session_id="s1", user_message="keep it short please, I hate long answers",
            side_question_token=None,
            conversation_history=[{"role": "user", "content": "keep it short please"}])

    assert _statuses(engine), "no fact was proposed — the no-token path dropped the turn"
    assert ctx.calls, "ctx.llm was never called"
    messages, kwargs = ctx.calls[0]
    assert kwargs.get("purpose") == "hwiki.fact-extraction"
    # Instructions must be the system message and the conversation the user message;
    # a single user message reads as chat and the model answers it instead.
    assert [m["role"] for m in messages][:2] == ["system", "user"], \
        f"expected system+user messages, got {[m['role'] for m in messages]}"
    assert "keep it short" in messages[1]["content"]


def test_the_llm_fallback_is_only_used_when_there_is_no_token(engine):
    """A real token must still win: it reuses the session's warm cache."""
    token_calls = []

    class _CtxWithBoth:
        llm = None  # replaced below

        def __init__(self):
            self.llm = _Boom()

        def side_question(self, token, prompt, timeout=25.0):
            token_calls.append(token)
            return json.dumps(FACTS)

    class _Boom:
        def complete(self, *a, **k):  # pragma: no cover - must not run
            raise AssertionError("ctx.llm must not run when a token is present")

    handler = engine.hook_handler(_CtxWithBoth())
    handler(session_id="s2", user_message="keep it short please, I hate long answers",
            side_question_token="live-token", conversation_history=[])

    assert token_calls == ["live-token"]
    assert _statuses(engine)


def test_the_conversation_is_sent_as_data_not_as_chat(engine):
    """A prompt-only user message made a real model answer the transcript as chat.

    The extraction instructions have to be the system message so the conversation is
    data to analyse. Without this, a model replies "what would you like explained?"
    and nothing is ever proposed — silently, because parsing prose just returns None.
    """
    class _Chatty:
        def __init__(self):
            self.llm = self
            self.seen = []

        def complete(self, messages, **kwargs):
            self.seen.append(messages)
            return _Result("Sure! What would you like me to explain?")

        def side_question(self, *a, **k):  # pragma: no cover
            raise AssertionError("must not run")

    ctx = _Chatty()
    handler = engine.hook_handler(ctx)
    handler(session_id="s4", user_message="explain the thing to me in simple words please",
            side_question_token=None,
            conversation_history=[{"role": "user", "content": "explain the thing in simple words"}])

    assert ctx.seen, "ctx.llm was never called"
    messages = ctx.seen[0]
    assert messages[0]["role"] == "system", "instructions must be the system message"
    assert "Extract" in messages[0]["content"] or "fact" in messages[0]["content"].lower()
    assert any("explain the thing in simple words" in m["content"] for m in messages[1:]), \
        "the conversation was not passed as data"


def test_a_chatty_reply_proposes_nothing_and_says_why(engine, caplog):
    """Prose instead of JSON must log the reply, not vanish."""
    class _Prose:
        llm = None
        def __init__(self): self.llm = self
        def complete(self, messages, **kwargs): return _Result("Sure! Which part?")
        def side_question(self, *a, **k):  # pragma: no cover
            raise AssertionError("must not run")

    handler = engine.hook_handler(_Prose())
    with caplog.at_level("WARNING"):
        handler(session_id="s5", user_message="explain the thing to me in simple words please",
                side_question_token=None,
                conversation_history=[{"role": "user", "content": "explain in simple words"}])
    assert not _statuses(engine)
    assert any("failed to parse" in r.getMessage() for r in caplog.records), \
        "an unparseable reply produced no diagnostic"


def test_extraction_off_makes_no_request_at_all(engine):
    """``extract_enabled: false`` must mean zero LLM traffic, not "tries and gives up".

    Extraction spends the user's tokens on the user's provider. A switch that still
    builds a prompt and calls out would be a switch that only looks off.
    """
    eng = FactsEngine(engine.hermes_home, engine.vault_root, engine.memory_rel,
                      config={"extract_enabled": False})

    class _NeverCalled:
        def __init__(self):
            self.llm = self
            self.calls = 0

        def complete(self, *a, **k):
            self.calls += 1
            raise AssertionError("no LLM call may happen when extraction is off")

        def side_question(self, *a, **k):
            self.calls += 1
            raise AssertionError("no side question may happen when extraction is off")

    ctx = _NeverCalled()
    state = eng._session("off-1")
    state["force_review"] = True  # force everything that could possibly fire
    handler = eng.hook_handler(ctx)
    for _ in range(3):
        handler(session_id="off-1", user_message="my dev machine runs Fedora and has 8 GB of RAM",
                side_question_token="a-real-token", conversation_history=[])

    assert ctx.calls == 0, "extraction ran while disabled"
    assert not _statuses(eng)


def test_extraction_off_beats_a_real_token(engine):
    """A live token must not sneak past the switch."""
    eng = FactsEngine(engine.hermes_home, engine.vault_root, engine.memory_rel,
                      config={"extract_enabled": False})

    calls = []

    class _Ctx:
        llm = None

        def side_question(self, token, prompt, timeout=25.0):
            calls.append(token)
            return json.dumps(FACTS)

    state = eng._session("off-2")
    state["force_review"] = True
    eng.hook_handler(_Ctx())(session_id="off-2",
                             user_message="my dev machine runs Fedora and has 8 GB of RAM",
                             side_question_token="live-token", conversation_history=[])

    assert calls == [], "the side-question path ran while extraction was disabled"


def test_extraction_is_off_when_the_key_is_absent(engine):
    """An unconfigured store must not spend tokens: absent means off.

    A cost-bearing feature that is on by default takes a decision on the user's
    behalf that they never got to make.
    """
    eng = FactsEngine(engine.hermes_home, engine.vault_root, engine.memory_rel, config={})
    called = []

    class _Ctx:
        def __init__(self):
            self.llm = self

        def complete(self, *a, **k):
            called.append(1)
            return _Result(json.dumps(FACTS))

        def side_question(self, *a, **k):
            called.append(1)
            return json.dumps(FACTS)

    state = eng._session("d-1")
    state["force_review"] = True
    eng.hook_handler(_Ctx())(session_id="d-1",
                             user_message="my dev machine runs Fedora and has 8 GB of RAM",
                             side_question_token="tok", conversation_history=[])
    assert called == [], "extraction spent tokens without being opted into"


def test_a_turn_with_no_llm_and_no_token_is_logged_not_silent(engine, caplog):
    """Silence is not a pass: a skipped extraction must leave a reason in the log."""
    handler = engine.hook_handler(_CtxWithNothing())
    with caplog.at_level("INFO"):
        handler(session_id="s3", user_message="keep it short please, I hate long answers",
                side_question_token=None, conversation_history=[])

    assert any("no ctx.llm" in r.getMessage() or "no side_question_token" in r.getMessage()
               for r in caplog.records), "a skipped extraction produced no log line"
    assert not _statuses(engine)
