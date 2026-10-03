from vllm_thinking_lang import injection


def test_injects_when_no_system_message():
    messages = [{"role": "user", "content": "hi"}]
    out = injection.apply_messages(messages, {"thinking_language": "cs"})
    assert out[0]["role"] == "system"
    assert "česky" in out[0]["content"]
    assert out[1] == messages[0]


def test_appends_to_existing_system_message():
    messages = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "hi"},
    ]
    out = injection.apply_messages(messages, {"thinking_language": "cs"})
    assert len(out) == 2
    assert out[0]["content"].startswith("You are terse.")
    assert "česky" in out[0]["content"]


def test_does_not_mutate_input():
    messages = [{"role": "user", "content": "hi"}]
    snapshot = [dict(m) for m in messages]
    injection.apply_messages(messages, {"thinking_language": "cs"})
    assert messages == snapshot


def test_no_injection_when_off():
    messages = [{"role": "user", "content": "hi"}]
    assert injection.apply_messages(messages, {"thinking_language": "off"}) == messages


def test_no_injection_when_language_unset():
    messages = [{"role": "user", "content": "hi"}]
    assert injection.apply_messages(messages, {}) == messages
    assert injection.apply_messages(messages, None) == messages


def test_no_injection_when_thinking_disabled():
    messages = [{"role": "user", "content": "hi"}]
    out = injection.apply_messages(messages, {"thinking_language": "cs", "enable_thinking": False})
    assert out == messages


def test_injection_when_thinking_enabled_explicitly():
    messages = [{"role": "user", "content": "hi"}]
    out = injection.apply_messages(messages, {"thinking_language": "cs", "enable_thinking": True})
    assert out[0]["role"] == "system"


def test_env_default_applies(monkeypatch):
    monkeypatch.setenv("VLLM_THINKING_LANG", "cs")
    messages = [{"role": "user", "content": "hi"}]
    out = injection.apply_messages(messages, {})
    assert out[0]["role"] == "system"


def test_request_beats_env(monkeypatch):
    monkeypatch.setenv("VLLM_THINKING_LANG", "en")
    messages = [{"role": "user", "content": "hi"}]
    out = injection.apply_messages(messages, {"thinking_language": "cs"})
    assert "česky" in out[0]["content"]


def test_non_dict_kwargs_does_not_raise():
    messages = [{"role": "user", "content": "hi"}]
    out = injection.apply_messages(messages, "garbage")
    assert isinstance(out, list)


def test_messages_with_tools_are_untouched_beyond_injection():
    """A language hint must not disturb tool-call parsing."""
    messages = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function"}]},
        {"role": "tool", "tool_call_id": "1", "content": "sunny"},
    ]
    out = injection.apply_messages(messages, {"thinking_language": "cs"})
    assert out[1:] == messages[1:]
    assert "česky" in out[0]["content"]


def test_prefill_request_is_skipped():
    """Injecting a system turn would break a prefilled assistant continuation."""
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "The answer is"},
    ]
    out = injection.apply_messages(messages, {"thinking_language": "cs"}, prefill=True)
    assert out == messages


def test_multimodal_system_content_is_not_corrupted():
    """System content that is a list of parts must stay a list of parts."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are terse."}]},
        {"role": "user", "content": "hi"},
    ]
    out = injection.apply_messages(messages, {"thinking_language": "cs"})
    assert isinstance(out[0]["content"], list)
    assert out[0]["content"][0]["text"] == "You are terse."
    assert "česky" in out[0]["content"][1]["text"]

# --- apply/unapply wiring ---------------------------------------------------
# Global Constraint: register() must be re-entrant, and apply() is what
# register() calls. A missing-symbol vLLM must degrade, not crash.


def _fake_renderer_module(monkeypatch, with_symbol=True):
    import sys
    import types

    pkg = types.ModuleType("vllm.renderers")
    pkg.__path__ = []
    mod = types.ModuleType("vllm.renderers.online_renderer")

    class OnlineRenderer:
        async def preprocess_chat(self, request, messages, *args, **kwargs):
            return messages

    if with_symbol:
        mod.OnlineRenderer = OnlineRenderer
    pkg.online_renderer = mod

    vllm_pkg = types.ModuleType("vllm")
    vllm_pkg.__path__ = []
    vllm_pkg.renderers = pkg

    monkeypatch.setitem(sys.modules, "vllm", vllm_pkg)
    monkeypatch.setitem(sys.modules, "vllm.renderers", pkg)
    monkeypatch.setitem(sys.modules, "vllm.renderers.online_renderer", mod)
    return OnlineRenderer


def test_apply_patches_preprocess_chat(monkeypatch):
    original = _fake_renderer_module(monkeypatch)
    injection.apply()
    try:
        assert injection.is_applied() is True
        assert original.preprocess_chat is not injection._original
    finally:
        injection.unapply()


def test_apply_is_idempotent(monkeypatch):
    _fake_renderer_module(monkeypatch)
    injection.apply()
    first = injection._original
    injection.apply()
    assert injection._original is first
    injection.unapply()


def test_unapply_restores_original(monkeypatch):
    original = _fake_renderer_module(monkeypatch)
    baseline = original.preprocess_chat
    injection.apply()
    injection.unapply()
    assert original.preprocess_chat is baseline
    assert injection.is_applied() is False


def test_missing_renderer_class_is_not_fatal(monkeypatch):
    _fake_renderer_module(monkeypatch, with_symbol=False)
    injection.apply()
    assert injection.is_applied() is False
    injection.unapply()


def test_apply_respects_layer_flag(monkeypatch):
    _fake_renderer_module(monkeypatch)
    monkeypatch.setenv("VLLM_THINKING_LANG_INJECT", "0")
    injection.apply()
    assert injection.is_applied() is False
