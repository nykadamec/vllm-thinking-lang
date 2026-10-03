import sys
import types

import pytest

from vllm_thinking_lang import filter_patch


class FakeTokenizer:
    def __init__(self):
        self.allowed: set[str] = set()

    def apply_chat_template(self, messages, **kwargs):
        return messages, kwargs


@pytest.fixture
def fake_hf(monkeypatch):
    """Stand in for vllm.renderers + vllm.renderers.hf with the patched symbol."""
    pkg = types.ModuleType("vllm.renderers")
    pkg.__path__ = []
    mod = types.ModuleType("vllm.renderers.hf")

    def resolve_chat_template_kwargs(tokenizer, chat_template, chat_template_kwargs, raise_on_unexpected=True):
        return {k: v for k, v in chat_template_kwargs.items() if k in tokenizer.allowed}

    mod.resolve_chat_template_kwargs = resolve_chat_template_kwargs
    pkg.hf = mod

    vllm_pkg = types.ModuleType("vllm")
    vllm_pkg.__path__ = []
    vllm_pkg.renderers = pkg

    monkeypatch.setitem(sys.modules, "vllm", vllm_pkg)
    monkeypatch.setitem(sys.modules, "vllm.renderers", pkg)
    monkeypatch.setitem(sys.modules, "vllm.renderers.hf", mod)
    return mod, FakeTokenizer()


TEMPLATE_DECLARING = "{% if thinking_language %}x{% endif %}"


def test_patched_filter_lets_unknown_kwarg_through(fake_hf):
    _, tokenizer = fake_hf
    tokenizer.allowed = {"enable_thinking"}
    filter_patch.apply()
    try:
        from vllm.renderers.hf import resolve_chat_template_kwargs as patched

        out = patched(tokenizer, TEMPLATE_DECLARING, {"thinking_language": "cs", "enable_thinking": True})
        assert out["thinking_language"] == "cs"
        assert out["enable_thinking"] is True
    finally:
        filter_patch.unapply()


def test_unpatched_filter_would_drop_it(fake_hf):
    mod, tokenizer = fake_hf
    out = mod.resolve_chat_template_kwargs(tokenizer, TEMPLATE_DECLARING, {"thinking_language": "cs"})
    assert "thinking_language" not in out


def test_apply_is_idempotent(fake_hf):
    filter_patch.apply()
    assert filter_patch.is_applied() is True
    filter_patch.apply()
    assert filter_patch.is_applied() is True
    filter_patch.unapply()


def test_unapply_restores_original(fake_hf):
    mod, _ = fake_hf
    original = mod.resolve_chat_template_kwargs
    filter_patch.apply()
    filter_patch.unapply()
    assert mod.resolve_chat_template_kwargs is original
    assert filter_patch.is_applied() is False


def test_unapply_without_apply_is_safe(fake_hf):
    filter_patch.unapply()
    assert filter_patch.is_applied() is False


def test_missing_symbol_is_not_fatal(fake_hf, monkeypatch):
    """A vLLM version without this symbol must not break plugin load."""
    fake_hf
    import vllm.renderers.hf as hf

    monkeypatch.delattr(hf, "resolve_chat_template_kwargs", raising=False)
    filter_patch.apply()
    assert filter_patch.is_applied() is False
    filter_patch.unapply()