"""tool_completion must nudge the model to finalize before the budget exhausts.

Bug (2026-06-24 run): the AI PM Phase-1 and Phase-2 main passes "exhausted
without sealing" — the model spent every tool turn on grounding tools and never
called its submission tool, so tool_completion fell through to "[max iterations
reached]" and the caller's force-seal fallback had to compel the decision.
The loop never told the model it was about to run out of turns. This injects a
final-turn nudge into the last tool-result turn so the model gets one explicit
chance to finalize within budget.
"""
import ascent.llm.client as c


class _Usage:
    input_tokens = 10
    output_tokens = 5


class _Block:
    def __init__(self, type, **kw):
        self.type = type
        for k, v in kw.items():
            setattr(self, k, v)


class _Resp:
    def __init__(self, stop_reason, content):
        self.stop_reason = stop_reason
        self.content = content
        self.usage = _Usage()


class _FakeMessages:
    def __init__(self, parent):
        self.parent = parent

    def create(self, **kwargs):
        # record the messages each create saw, then always ask for a tool (never
        # end_turn) to simulate a model that over-grounds and exhausts the budget.
        self.parent.calls.append(kwargs["messages"])
        return _Resp("tool_use", [_Block("tool_use", id="t", name="grounding", input={})])


class _FakeClient:
    def __init__(self):
        self.calls = []
        self.messages = _FakeMessages(self)


def _texts_in(messages):
    out = []
    for m in messages:
        content = m["content"]
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text":
                    out.append(b.get("text", ""))
        elif isinstance(content, str):
            out.append(content)
    return out


def test_injects_final_turn_nudge_before_budget_exhausts(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(c, "_get_client", lambda: fake)
    monkeypatch.setattr(c, "_check_api_key", lambda: None)
    monkeypatch.setattr(c, "_record_usage", lambda *a, **k: None)

    c.tool_completion(
        system_prompt="sys",
        user_prompt="do it",
        tools=[{"name": "grounding"}],
        tool_executor=lambda n, i: "ok",
        model="m",
        max_tool_calls=3,
    )

    # The final create call must have seen the nudge somewhere in its messages.
    last_call_texts = " ".join(_texts_in(fake.calls[-1]))
    assert c._FINAL_TURN_NUDGE in last_call_texts, "final-turn nudge not injected"


def test_no_nudge_when_model_finishes_early(monkeypatch):
    """If the model returns end_turn on the first call, no nudge is injected."""
    fake = _FakeClient()
    fake.messages.create = lambda **kw: (
        fake.calls.append(kw["messages"]) or _Resp("end_turn", [_Block("text", text="done")])
    )
    monkeypatch.setattr(c, "_get_client", lambda: fake)
    monkeypatch.setattr(c, "_check_api_key", lambda: None)
    monkeypatch.setattr(c, "_record_usage", lambda *a, **k: None)

    out = c.tool_completion("sys", "do it", tools=[{"name": "g"}],
                            tool_executor=lambda n, i: "ok", model="m", max_tool_calls=3)
    assert out == "done"
    all_texts = " ".join(t for call in fake.calls for t in _texts_in(call))
    assert c._FINAL_TURN_NUDGE not in all_texts
