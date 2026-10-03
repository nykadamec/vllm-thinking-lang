import pathlib

import pytest

from vllm_thinking_lang import template_patch


class FakeTokenizer:
    def __init__(self, chat_template="ORIGINAL"):
        self.chat_template = chat_template


def test_disabled_by_default():
    assert template_patch.patch_template(FakeTokenizer()) is False


def test_patch_replaces_template(monkeypatch, tmp_path):
    path = tmp_path / "t.jinja"
    path.write_text("PATCHED")
    monkeypatch.setenv("VLLM_THINKING_LANG_TEMPLATE", str(path))
    tok = FakeTokenizer()
    assert template_patch.patch_template(tok) is True
    assert tok.chat_template == "PATCHED"


def test_patch_is_idempotent(monkeypatch, tmp_path):
    path = tmp_path / "t.jinja"
    path.write_text("PATCHED")
    monkeypatch.setenv("VLLM_THINKING_LANG_TEMPLATE", str(path))
    tok = FakeTokenizer()
    template_patch.patch_template(tok)
    template_patch.patch_template(tok)
    assert tok.chat_template == "PATCHED"


def test_missing_file_is_not_fatal(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_THINKING_LANG_TEMPLATE", str(tmp_path / "nope.jinja"))
    tok = FakeTokenizer()
    assert template_patch.patch_template(tok) is False
    assert tok.chat_template == "ORIGINAL"


def test_apply_is_noop_when_disabled():
    template_patch.apply()
    template_patch.unapply()


def test_bundled_template_declares_the_variable():
    text = pathlib.Path(template_patch.__file__).with_name("thinking.jinja").read_text()
    assert "thinking_language" in text


def test_bundled_template_has_no_open_think_tag():
    text = pathlib.Path(template_patch.__file__).with_name("thinking.jinja").read_text()
    assert "<think>" not in text


def test_existing_template_with_variable_is_left_alone(monkeypatch, tmp_path):
    """If the operator's own template already handles it, do not clobber it."""
    monkeypatch.setenv("VLLM_THINKING_LANG_TEMPLATE", str(tmp_path / "unused.jinja"))
    (tmp_path / "unused.jinja").write_text("REPLACEMENT")
    own = "{% if thinking_language %}OWN{% endif %}"
    tok = FakeTokenizer(chat_template=own)
    assert template_patch.patch_template(tok) is False
    assert tok.chat_template == own