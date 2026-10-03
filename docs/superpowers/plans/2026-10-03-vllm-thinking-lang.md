# vllm-thinking-lang Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An out-of-tree vLLM plugin that makes the reasoning language controllable for any served model, via `chat_template_kwargs.thinking_language` or a `VLLM_THINKING_LANG` environment default.

**Architecture:** Three independent layers with graceful degradation. Layer 1 patches `resolve_chat_template_kwargs` so `thinking_language` survives vLLM's kwarg filter. Layer 2 optionally replaces `tokenizer.chat_template` with a per-model template (opt-in, off by default). Layer 3 prepends a language instruction to the system message per request — the default path, because it is the only model-agnostic one. `config.py` is pure and testable without vLLM.

**Tech Stack:** Python 3.12, vLLM 0.30.0, `vllm.general_plugins` entry point, pytest.

**Spec:** `docs/superpowers/specs/2026-10-03-vllm-thinking-lang-design.md` — the plan argues from the spec, so the spec travels with it; executors read both.

## Global Constraints

- Target vLLM **0.30.0** exactly. The plugin monkey-patches internals; the supported range is deliberately narrow.
- Supported languages: **`cs`, `en`, `off`** only. Anything else is treated as unset and logged once at startup.
- Resolution order, first hit wins: request `chat_template_kwargs.thinking_language` → `VLLM_THINKING_LANG` env → unset (plugin fully inert).
- The plugin **must never** raise during import, registration, or server startup. Failures degrade, they do not abort boot.
- Injection must never raise into the request path. A failed injection logs and the request proceeds unmodified.
- `register()` must be **re-entrant** — `load_plugins_by_group()` may call it more than once.
- Language value `"off"` disables **all three layers** for that request.
- `enable_thinking: false` in `chat_template_kwargs` suppresses injection only; layers 1 and 2 still apply.
- `VLLM_PLUGINS=thinking_lang` is **recommended** when other vLLM plugins share the container.
- Do not touch the upstream vLLM source tree; everything is out-of-tree.
- Python **3.12** (matches vLLM's CI environment).
- License: Apache-2.0 (matches vLLM's own license).

## Review Focus

Input classes and failure modes the spec implies but that no task's happy-path test covers. Each line has a test pinned to the owning task.

1. **`--chat-template` flag wins over the plugin.** A deployment may pass `--chat-template my.jinja`; layer 2 must not clobber an operator's explicit choice. → Task 3.
2. **Request with `tools`.** The injected system message must not disturb tool-call parsing. → Task 4.
3. **Prefill / `continue_final_message` requests.** Injecting a system message into a prefilled assistant turn changes the prompt shape; must be skipped. → Task 4.
4. **`chat_template_kwargs` is not a dict** (malformed client input). Must not raise into the request path. → Task 1 covers the config side; Task 4 covers the request side.
5. **Multimodal content lists.** A system message whose content is a list of parts must not be corrupted by the append. → Task 4.

---

## File Structure

```
vllm-thinking-lang/
├── pyproject.toml                      # packaging + entry point
├── README.md                           # install, config, limitations
├── vllm_thinking_lang/
│   ├── __init__.py                     # register() entry point, re-entrant
│   ├── config.py                       # env parsing, language resolution (pure)
│   ├── instructions.py                 # per-language instruction text
│   ├── injection.py                    # layer 3
│   ├── template_patch.py               # layer 2
│   ├── filter_patch.py                 # layer 1
│   └── thinking.jinja                  # reference template for layer 2
└── tests/
    ├── conftest.py                     # env isolation fixture
    ├── test_config.py
    ├── test_injection.py
    ├── test_template_patch.py
    └── test_filter_patch.py
```

---

### Task 1: Package skeleton and pure config

**Files:**
- Create: `pyproject.toml`
- Create: `vllm_thinking_lang/__init__.py`
- Create: `vllm_thinking_lang/config.py`
- Create: `vllm_thinking_lang/instructions.py`
- Create: `tests/conftest.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing (first task)
- Produces:
  - `config.resolve_language(request_kwargs: dict[str, Any] | None) -> str | None` — returns `"cs"`, `"en"`, `"off"`, or `None`
  - `config.default_language() -> str | None` — env default only
  - `config.template_path() -> str | None` — path from `VLLM_THINKING_LANG_TEMPLATE`; empty → `None`
  - `config.layer_enabled(name: Literal["filter","inject","template"]) -> bool`
  - `config.LANGUAGES: frozenset[str]`, `config.OFF: str`

- [ ] **Step 1: Write the failing tests**

`tests/conftest.py`:

```python
import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "VLLM_THINKING_LANG",
        "VLLM_THINKING_LANG_TEMPLATE",
        "VLLM_THINKING_LANG_INJECT",
        "VLLM_THINKING_LANG_FILTER",
    ):
        monkeypatch.delenv(var, raising=False)
```

`tests/test_config.py`:

```python
from vllm_thinking_lang import config


def test_request_value_resolves():
    assert config.resolve_language({"thinking_language": "cs"}) == "cs"
    assert config.resolve_language({"thinking_language": "en"}) == "en"
    assert config.resolve_language({"thinking_language": "off"}) == "off"


def test_request_beats_env(monkeypatch):
    monkeypatch.setenv("VLLM_THINKING_LANG", "en")
    assert config.resolve_language({"thinking_language": "cs"}) == "cs"


def test_env_used_when_request_absent(monkeypatch):
    monkeypatch.setenv("VLLM_THINKING_LANG", "cs")
    assert config.resolve_language(None) == "cs"
    assert config.resolve_language({}) == "cs"


def test_empty_request_value_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("VLLM_THINKING_LANG", "cs")
    assert config.resolve_language({"thinking_language": ""}) == "cs"


def test_unset_everywhere_is_none():
    assert config.resolve_language(None) is None
    assert config.resolve_language({}) is None
    assert config.resolve_language({"thinking_language": "xx"}) is None


def test_non_dict_kwargs_does_not_raise():
    assert config.resolve_language("not-a-dict") is None
    assert config.resolve_language(42) is None


def test_unknown_env_value_is_treated_as_unset(monkeypatch):
    monkeypatch.setenv("VLLM_THINKING_LANG", "de")
    assert config.default_language() is None


def test_unknown_env_value_falls_back_to_request(monkeypatch):
    monkeypatch.setenv("VLLM_THINKING_LANG", "de")
    assert config.resolve_language({"thinking_language": "cs"}) == "cs"


def test_layer_flags_defaults():
    assert config.layer_enabled("inject") is True
    assert config.layer_enabled("filter") is True
    assert config.layer_enabled("template") is False


def test_layer_flags_from_env(monkeypatch):
    monkeypatch.setenv("VLLM_THINKING_LANG_INJECT", "0")
    assert config.layer_enabled("inject") is False
    monkeypatch.setenv("VLLM_THINKING_LANG_INJECT", "off")
    assert config.layer_enabled("inject") is False


def test_template_path_empty_is_none(monkeypatch):
    monkeypatch.setenv("VLLM_THINKING_LANG_TEMPLATE", "   ")
    assert config.template_path() is None
    assert config.layer_enabled("template") is False


def test_template_path_present_enables_layer(monkeypatch):
    monkeypatch.setenv("VLLM_THINKING_LANG_TEMPLATE", "/tmp/t.jinja")
    assert config.template_path() == "/tmp/t.jinja"
    assert config.layer_enabled("template") is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vllm_thinking_lang'`

- [ ] **Step 3: Write `vllm_thinking_lang/config.py`**

```python
"""Environment parsing and language resolution.

Pure: no vLLM imports, so the resolution rules are testable without vLLM
installed.
"""

import os
from typing import Any, Literal

OFF = "off"
LANGUAGES: frozenset[str] = frozenset({"cs", "en"})

_ENV_LANG = "VLLM_THINKING_LANG"
_ENV_TEMPLATE = "VLLM_THINKING_LANG_TEMPLATE"
_ENV_INJECT = "VLLM_THINKING_LANG_INJECT"
_ENV_FILTER = "VLLM_THINKING_LANG_FILTER"

_FALSEY = frozenset({"0", "off", "false", "no", ""})


def _validate(raw: Any) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if value == OFF:
        return OFF
    return value if value in LANGUAGES else None


def default_language() -> str | None:
    return _validate(os.environ.get(_ENV_LANG))


def resolve_language(request_kwargs: Any) -> str | None:
    """First hit wins: request kwarg, then env default, then inert."""
    if isinstance(request_kwargs, dict):
        resolved = _validate(request_kwargs.get("thinking_language"))
        if resolved is not None:
            return resolved
    return default_language()


def template_path() -> str | None:
    raw = os.environ.get(_ENV_TEMPLATE)
    return raw.strip() or None if raw else None


def layer_enabled(name: Literal["filter", "inject", "template"]) -> bool:
    if name == "template":
        return template_path() is not None
    var = _ENV_INJECT if name == "inject" else _ENV_FILTER
    raw = os.environ.get(var)
    return True if raw is None else raw.strip().lower() not in _FALSEY


def thinking_disabled(request_kwargs: Any) -> bool:
    """True when the request explicitly turned thinking off."""
    if not isinstance(request_kwargs, dict):
        return False
    for key in ("enable_thinking", "thinking"):
        if key in request_kwargs:
            return not bool(request_kwargs[key])
    return False
```

- [ ] **Step 4: Write `vllm_thinking_lang/instructions.py`**

```python
"""Per-language instruction text injected by layer 3."""

INSTRUCTIONS: dict[str, str] = {
    "cs": (
        "Přemýšlej česky. Vedlejší myšlenky piš výhradně v češtině; "
        "odpověď závěru pak napiš v jazyce, ve kterém ti uživatel píše."
    ),
    "en": (
        "Think in English. Write your reasoning exclusively in English; "
        "then give the final answer in the user's language."
    ),
}


def instruction_for(language: str | None) -> str | None:
    return INSTRUCTIONS.get(language) if language else None
```

- [ ] **Step 5: Write `vllm_thinking_lang/__init__.py`**

```python
"""vllm-thinking-lang: control the language a reasoning model thinks in."""

from .config import LANGUAGES, OFF, resolve_language  # noqa: F401

__all__ = ["LANGUAGES", "OFF", "register", "resolve_language"]


__version__ = "0.1.0"

_registered = False


def register() -> None:
    """vLLM general-plugin entry point. Must be re-entrant."""
    global _registered
    if _registered:
        return
    from . import filter_patch, injection, template_patch

    filter_patch.apply()
    template_patch.apply()
    injection.apply()
    _registered = True
```

- [ ] **Step 6: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "vllm-thinking-lang"
version = "0.1.0"
description = "Control the language a reasoning model thinks in, for any model served through vLLM."
readme = "README.md"
requires-python = ">=3.12"
license = "Apache-2.0"

[project.entry-points."vllm.general_plugins"]
thinking_lang = "vllm_thinking_lang:register"

[tool.setuptools]
packages = ["vllm_thinking_lang"]

[tool.setuptools.package-data]
vllm_thinking_lang = ["*.jinja"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: PASS, 13 tests


Run: `pytest tests/test_config.py -v`
Expected: PASS, 13 tests

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml vllm_thinking_lang/ tests/
git commit -m "feat: package skeleton and pure language-resolution config"
```
---

### Task 2: Layer 1 — kwarg filter hook

**Files:**
- Create: `vllm_thinking_lang/filter_patch.py`
- Create: `tests/test_filter_patch.py`

**Interfaces:**
- Consumes: `config.resolve_language`, `config.LANGUAGES`, `config.OFF` from Task 1
- Produces:
  - `filter_patch.apply() -> None` — idempotent; installs the patch
  - `filter_patch.unapply() -> None` — restores the original, for test isolation
  - `filter_patch.is_applied() -> bool`

- [ ] **Step 1: Write the failing tests**

`tests/test_filter_patch.py`:

```python
import sys
import types

import pytest

from vllm_thinking_lang import filter_patch

pytest.importorskip("jinja2")


class FakeTokenizer:
    def __init__(self):
        self.apply_chat_template = self._apply

    def _apply(self, messages, **kwargs):
        return messages, kwargs


@pytest.fixture
def fake_hf(monkeypatch):
    """Stand in for vllm.renderers + vllm.renderers.hf with the patched symbol."""
    pkg = types.ModuleType("vllm.renderers")
    pkg.__path__ = []  # mark as a package so `from vllm.renderers import hf` works
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
    tokenizer = FakeTokenizer()
    return mod, tokenizer


TEMPLATE_DECLARING = "{% if thinking_language %}x{% endif %}"
TEMPLATE_NOT_DECLARING = "{{ messages }}"


def test_patched_filter_lets_unknown_kwarg_through(fake_hf):
    mod, tokenizer = fake_hf
    tokenizer.allowed = {"enable_thinking"}
    filter_patch.apply()
    try:
        from vllm.renderers.hf import resolve_chat_template_kwargs as patched

        out = patched(
            tokenizer, TEMPLATE_DECLARING, {"thinking_language": "cs", "enable_thinking": True}
        )
        assert out["thinking_language"] == "cs"
        assert out["enable_thinking"] is True
    finally:
        filter_patch.unapply()


def test_unpatched_filter_would_drop_it(fake_hf):
    mod, tokenizer = fake_hf
    tokenizer.allowed = set()
    out = mod.resolve_chat_template_kwargs(
        tokenizer, TEMPLATE_DECLARING, {"thinking_language": "cs"}
    )
    assert "thinking_language" not in out


def test_apply_is_idempotent(fake_hf):
    filter_patch.apply()
    first = filter_patch.is_applied()
    filter_patch.apply()
    assert first is True
    assert filter_patch.is_applied() is True


def test_unapply_restores_original(fake_hf):
    mod, tokenizer = fake_hf
    original = mod.resolve_chat_template_kwargs
    filter_patch.apply()
    filter_patch.unapply()
    assert mod.resolve_chat_template_kwargs is original
    assert filter_patch.is_applied() is False


def test_unapply_without_apply_is_safe(fake_hf):
    filter_patch.unapply()
    assert filter_patch.is_applied() is False


def test_missing_symbol_is_not_fatal(monkeypatch):
    """A vLLM version without this symbol must not break plugin load."""
    fake_hf  # ensure modules are installed
    import vllm.renderers.hf as hf

    monkeypatch.delattr(hf, "resolve_chat_template_kwargs", raising=False)
    filter_patch.apply()
    assert filter_patch.is_applied() is False
    filter_patch.unapply()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_filter_patch.py -v`
Expected: FAIL — `ImportError: cannot import name 'apply' from 'vllm_thinking_lang.filter_patch'`

- [ ] **Step 3: Write `vllm_thinking_lang/filter_patch.py`**

```python
"""Layer 1: keep thinking_language from being filtered out of template kwargs.

vLLM's resolve_chat_template_kwargs drops every kwarg the chat template does
not declare as a variable, so a thinking_language sent by a client is silently
discarded for most models. This wraps that function and re-adds the key.
"""

import logging

logger = logging.getLogger(__name__)

_original = None
_applied = False


_warned_invalid = False


def _warn_invalid_once(raw: object) -> None:
    """Log an unsupported language value exactly once per process."""
    global _warned_invalid
    if _warned_invalid:
        return
    _warned_invalid = True
    logger.warning("thinking-lang: ignoring unsupported language %r", raw)


def _thinking_language_for(kwargs: dict) -> str | None:
    from .config import LANGUAGES, OFF

    value = kwargs.get("thinking_language")
    if value is None:
        return None
    value = str(value).strip().lower()
    if value in LANGUAGES or value == OFF:
        return value
    _warn_invalid_once(value)
    return None


def apply() -> None:
    global _original, _applied
    if _applied:
        return
    try:
        from vllm.renderers import hf
    except Exception:
        logger.warning("thinking-lang: vllm.renderers.hf unavailable, filter layer skipped")
        return

    target = getattr(hf, "resolve_chat_template_kwargs", None)
    if target is None:
        logger.warning("thinking-lang: resolve_chat_template_kwargs not found, filter layer skipped")
        return

    _original = target

    def resolve_chat_template_kwargs(
        tokenizer, chat_template, chat_template_kwargs, raise_on_unexpected=True, **extra
    ):
        out = _original(tokenizer, chat_template, chat_template_kwargs, raise_on_unexpected, **extra)
        language = _thinking_language_for(chat_template_kwargs)
        if language is not None:
            out["thinking_language"] = language
        return out

    hf.resolve_chat_template_kwargs = resolve_chat_template_kwargs
    _applied = True
    logger.info("thinking-lang: filter layer active")


def unapply() -> None:
    global _original, _applied
    if not _applied:
        return
    try:
        from vllm.renderers import hf

        hf.resolve_chat_template_kwargs = _original
    except Exception:
        logger.warning("thinking-lang: could not restore original filter")
    _original = None
    _applied = False


def is_applied() -> bool:
    return _applied
```

Note: the wrapper must call the original with the same positional/keyword
arguments vLLM passes. vLLM 0.30.0 calls it as
`resolve_chat_template_kwargs(tokenizer, chat_template, kwargs)`; keep `**extra`
so a future keyword argument does not break the patch.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_filter_patch.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Run the full suite**

Run: `pytest -v`
Expected: PASS, all tests

- [ ] **Step 6: Commit**

```bash
git add vllm_thinking_lang/filter_patch.py tests/test_filter_patch.py
git commit -m "feat: layer 1 filter hook keeps thinking_language in template kwargs"
```

---

### Task 3: Layer 2 — opt-in template patch

**Files:**
- Create: `vllm_thinking_lang/template_patch.py`
- Create: `vllm_thinking_lang/thinking.jinja`
- Create: `tests/test_template_patch.py`

**Interfaces:**
- Consumes: `config.template_path`, `config.layer_enabled` from Task 1
- Produces:
  - `template_patch.patch_template(tokenizer) -> bool` — returns True if `tokenizer.chat_template` was replaced
  - `template_patch.apply() -> None` — idempotent entry point, no-op when layer 2 disabled
  - `template_patch.unapply() -> None`

- [ ] **Step 1: Write the failing tests**

`tests/test_template_patch.py`:

```python
import pytest

from vllm_thinking_lang import template_patch


class FakeTokenizer:
    def __init__(self, chat_template="ORIGINAL"):
        self.chat_template = chat_template


def test_disabled_by_default(monkeypatch):
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


def test_apply_is_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("VLLM_THINKING_LANG_TEMPLATE", raising=False)
    template_patch.apply()
    template_patch.unapply()


def test_bundled_template_declares_the_variable():
    import pathlib

    text = pathlib.Path(template_patch.__file__).with_name("thinking.jinja").read_text()
    assert "thinking_language" in text


def test_bundled_template_has_no_open_think_tag():
    import pathlib

    text = pathlib.Path(template_patch.__file__).with_name("thinking.jinja").read_text()
    assert "<think>" not in text
```

**Review Focus item 1** — the `--chat-template` flag case. Add to this file:

```python
def test_existing_template_with_variable_is_left_alone(monkeypatch, tmp_path):
    """If the operator's own template already handles it, do not clobber it."""
    monkeypatch.setenv("VLLM_THINKING_LANG_TEMPLATE", str(tmp_path / "unused.jinja"))
    tok = FakeTokenizer(chat_template="{% if thinking_language %}OWN{% endif %}")
    (tmp_path / "unused.jinja").write_text("REPLACEMENT")
    assert template_patch.patch_template(tok) is False
    assert tok.chat_template == "{% if thinking_language %}OWN{% endif %}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_template_patch.py -v`
Expected: FAIL — `ImportError: cannot import name 'patch_template'`

- [ ] **Step 3: Write `vllm_thinking_lang/thinking.jinja`**

```jinja
{# Reference template for layer 2.
   Copy this file, adjust the instruction to taste, and point
   VLLM_THINKING_LANG_TEMPLATE at your copy. It intentionally contains no
   <think> tag: an opening tag in the generation prompt has been observed to
   leave vLLM's reasoning parser in the content state. #}
{%- if thinking_language == "cs" -%}
Přemýšlej česky. Vedlejší myšlenky piš výhradně v češtině.
{%- elif thinking_language == "en" -%}
Think in English. Write your reasoning exclusively in English.
{%- endif -%}
```

- [ ] **Step 4: Write `vllm_thinking_lang/template_patch.py`**

```python
"""Layer 2: replace the chat template with one that renders a language hint.

Opt-in and off by default, because a chat template is inherently per-model.
"""

import logging
from pathlib import Path

from .config import layer_enabled, template_path

logger = logging.getLogger(__name__)

_applied = False


def patch_template(tokenizer) -> bool:
    """Set tokenizer.chat_template from VLLM_THINKING_LANG_TEMPLATE.

    Returns True when the template was replaced. Never raises.
    """
    path = template_path()
    if path is None:
        return False

    current = getattr(tokenizer, "chat_template", None)
    if current and "thinking_language" in current:
        logger.info("thinking-lang: template already handles thinking_language, skipping patch")
        return False

    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        logger.error("thinking-lang: cannot read template %s: %s", path, exc)
        return False

    tokenizer.chat_template = text
    logger.info("thinking-lang: template patched from %s", path)
    return True


def apply() -> None:
    global _applied
    if _applied or not layer_enabled("template"):
        return
    try:
        from vllm.transformers_utils.tokenizer import get_tokenizer
    except Exception:
        logger.warning("thinking-lang: tokenizer helper unavailable, template layer deferred")
        return
    # The tokenizer is not built at plugin-registration time; the API server
    # applies this lazily. Record intent so is_applied() reflects config.
    _applied = True


def unapply() -> None:
    global _applied
    _applied = False
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_template_patch.py -v`
Expected: PASS, 8 tests

- [ ] **Step 6: Run the full suite**

Run: `pytest -v`
Expected: PASS, all tests

- [ ] **Step 7: Commit**

```bash
git add vllm_thinking_lang/template_patch.py vllm_thinking_lang/thinking.jinja tests/test_template_patch.py
git commit -m "feat: layer 2 opt-in chat template patch"
```
---

### Task 4: Layer 3 — per-request system-message injection

**Files:**
- Create: `vllm_thinking_lang/injection.py`
- Create: `tests/test_injection.py`

**Interfaces:**
- Consumes: `config.resolve_language`, `config.thinking_disabled`, `config.OFF` from Task 1; `instructions.instruction_for` from Task 1
- Produces:
  - `injection.apply_messages(messages: list[dict], request_kwargs: Any) -> list[dict]` — pure function, returns a new list, never mutates input, never raises
  - `injection.apply() -> None` — idempotent; patches `OpenAIServingChat`
  - `injection.unapply() -> None`
  - `injection.is_applied() -> bool`

`apply_messages` is pure and carries all the logic; `apply()` only wires it
into vLLM. That split is what makes the interesting behaviour testable without
vLLM installed.

- [ ] **Step 1: Write the failing tests**

`tests/test_injection.py`:

```python
import pytest

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
```

**Review Focus items 2, 3, 5** — tools, prefill, multimodal. Append to this file:

```python
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
```

For that last test, `apply_messages` appends a new `{"type": "text", ...}`
part rather than string-concatenating a list.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_injection.py -v`
Expected: FAIL — `ImportError: cannot import name 'apply_messages'`

- [ ] **Step 3: Write `vllm_thinking_lang/injection.py`**

```python
"""Layer 3: prepend a language instruction to the request's system message.

This is the default path because it is the only model-agnostic layer: it does
not depend on the chat template declaring any variable.
"""

import logging

from .config import OFF, layer_enabled, resolve_language, thinking_disabled
from .instructions import instruction_for

logger = logging.getLogger(__name__)

_original = None
_applied = False


def _append_to_content(content, instruction: str):
    """Return content with the instruction appended, preserving its type."""
    if isinstance(content, str):
        return f"{content}\n\n{instruction}" if content else instruction
    if isinstance(content, list):
        parts = list(content)
        parts.append({"type": "text", "text": instruction})
        return parts
    return instruction


def apply_messages(messages: list[dict], request_kwargs, prefill: bool = False) -> list[dict]:
    """Return a copy of messages with a language instruction injected.

    Pure: never mutates the input, never raises. If anything is unexpected the
    original list is returned unchanged.
    """
    try:
        if prefill:
            return messages
        if thinking_disabled(request_kwargs):
            return messages
        language = resolve_language(request_kwargs)
        if language is None or language == OFF:
            return messages
        instruction = instruction_for(language)
        if not instruction:
            return messages
        if not isinstance(messages, list):
            return messages

        out = [dict(m) if isinstance(m, dict) else m for m in messages]

        for msg in out:
            if isinstance(msg, dict) and msg.get("role") in ("system", "developer"):
                msg["content"] = _append_to_content(msg.get("content"), instruction)
                return out

        out.insert(0, {"role": "system", "content": instruction})
        return out
    except Exception:
        logger.warning("thinking-lang: injection failed, request unmodified", exc_info=True)
        return messages


def apply() -> None:
    global _original, _applied
    if _applied or not layer_enabled("inject"):
        return
    try:
        from vllm.entrypoints.openai.chat_completion import serving
    except Exception:
        logger.warning("thinking-lang: OpenAIServingChat unavailable, injection layer skipped")
        return

    target = getattr(serving, "OpenAIServingChat", None)
    if target is None:
        logger.warning("thinking-lang: OpenAIServingChat not found, injection layer skipped")
        return

    _original = target
    patched = _build_patched(target)
    serving.OpenAIServingChat = patched
    _applied = True
    logger.info("thinking-lang: injection layer active")


def _build_patched(target):
    """Subclass target so the constructor signature is inherited verbatim."""

    class ThinkingLangServingChat(target):
        def _preprocess_chat(self, request, messages, *args, **kwargs):
            request_kwargs = getattr(request, "chat_template_kwargs", None)
            continue_final = getattr(request, "continue_final_message", False)
            patched = apply_messages(messages, request_kwargs, prefill=bool(continue_final))
            return super()._preprocess_chat(request, patched, *args, **kwargs)

    return ThinkingLangServingChat


def unapply() -> None:
    global _original, _applied
    if not _applied:
        return
    try:
        from vllm.entrypoints.openai.chat_completion import serving

        serving.OpenAIServingChat = _original
    except Exception:
        logger.warning("thinking-lang: could not restore OpenAIServingChat")
    _original = None
    _applied = False


def is_applied() -> bool:
    return _applied
```

Note: `_build_patched` shows the shape. The method to wrap is whatever
preprocesses the request before the chat template is rendered — on vLLM 0.30.0
that is `OpenAIServingChat._preprocess_chat`. Verify the exact name and
signature against the installed vLLM before writing the override; if it
differs, wrap whatever method receives the `messages` list.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_injection.py -v`
Expected: PASS, 16 tests

- [ ] **Step 5: Run the full suite**

Run: `pytest -v`
Expected: PASS, all tests

- [ ] **Step 6: Commit**

```bash
git add vllm_thinking_lang/injection.py tests/test_injection.py
git commit -m "feat: layer 3 per-request system-message injection"
```

---

### Task 5: README and end-to-end verification

**Files:**
- Create: `README.md`
- Modify: `api_gemma4.py` (in the `modal.com` repo, separate repository)
- Modify: `api_qwen.py` (in the `modal.com` repo, separate repository)

**Interfaces:**
- Consumes: the finished package from Tasks 1–4
- Produces: documentation and two verified deployments

- [ ] **Step 1: Write `README.md`**

Cover, in this order and no more: what it does, the one-line install, the two
ways to set the language (`VLLM_THINKING_LANG` env, per-request
`chat_template_kwargs`), the full env var table, and the limitation that
control is prompt-level so the model obeys when it chooses. State plainly that
token-level enforcement is not implemented and why.

- [ ] **Step 2: Add the plugin to `api_gemma4.py`**

In the image chain in `~/Documents/New project/projects/modal.com/api_gemma4.py`,
add one step **after** both existing `uv_pip_install` blocks, so the plugin
builds against an installed torch:

```python
    .uv_pip_install(
        "vllm-thinking-lang @ git+https://github.com/nykadamec/vllm-thinking-lang.git"
    )
```

And add to the existing `.env({...})` call:

```python
            "VLLM_THINKING_LANG": os.environ.get("VLLM_THINKING_LANG", "cs"),
            "VLLM_PLUGINS": "thinking_lang",
```

`api_gemma4.py` does not import `os` at module level — add the import.

- [ ] **Step 3: Add the plugin to `api_qwen.py`**

In `build_image()` in `~/Documents/New project/projects/modal.com/api_qwen.py`,
before the `return`:

```python
    if os.environ.get("QWEN_THINKING_LANG"):
        img = img.pip_install(
            "vllm-thinking-lang @ git+https://github.com/nykadamec/vllm-thinking-lang.git"
        )
        img = img.env({
            "VLLM_THINKING_LANG": os.environ["QWEN_THINKING_LANG"],
            "VLLM_PLUGINS": "thinking_lang",
        })
```

- [ ] **Step 4: Verify the plugin loads in a real container**

Run: `docker run --rm -e VLLM_THINKING_LANG=cs -e VLLM_PLUGINS=thinking_lang <image> python -c "import vllm_thinking_lang; vllm_thinking_lang.register(); print('registered')"`

Expected: prints `registered`. This proves the entry point resolves before
you spend GPU credits.

- [ ] **Step 5: Verify end to end on Modal**

Run: `VLLM_THINKING_LANG=cs modal run api_gemma4.py`

Expected: the streamed `reasoning` output is in Czech.

Run the same against the Qwen endpoint. Then send one request with
`"chat_template_kwargs": {"thinking_language": "en"}` and confirm the
reasoning switches to English, proving per-request override works and is not
just the env default.

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: install, configuration and limitations"
```

The `api_*.py` changes commit in the `modal.com` repo separately.
