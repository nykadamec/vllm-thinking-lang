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