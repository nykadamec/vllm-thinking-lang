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