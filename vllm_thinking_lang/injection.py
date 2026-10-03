"""Layer 3: prepend a language instruction to the request's system message.

This is the default path because it is the only model-agnostic layer: it does
not depend on the chat template declaring any variable.

Patch target is OnlineRenderer.preprocess_chat -- the last point where
`messages` is still a plain list, immediately before the chat template renders.
OpenAIServingChat delegates there (chat_completion/serving.py:240 ->
online_renderer.py:333).
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
        from vllm.renderers import online_renderer
    except Exception:
        logger.warning("thinking-lang: vllm.renderers.online_renderer unavailable, injection skipped")
        return

    target = getattr(online_renderer, "OnlineRenderer", None)
    preprocess = getattr(target, "preprocess_chat", None) if target else None
    if preprocess is None:
        logger.warning("thinking-lang: OnlineRenderer.preprocess_chat not found, injection skipped")
        return

    _original = preprocess

    async def preprocess_chat(self, request, messages, *args, **kwargs):
        request_kwargs = getattr(request, "chat_template_kwargs", None)
        continue_final = getattr(request, "continue_final_message", False)
        patched = apply_messages(messages, request_kwargs, prefill=bool(continue_final))
        return await _original(self, request, patched, *args, **kwargs)

    target.preprocess_chat = preprocess_chat
    _applied = True
    logger.info("thinking-lang: injection layer active")


def unapply() -> None:
    global _original, _applied
    if not _applied:
        return
    try:
        from vllm.renderers import online_renderer

        online_renderer.OnlineRenderer.preprocess_chat = _original
    except Exception:
        logger.warning("thinking-lang: could not restore preprocess_chat")
    _original = None
    _applied = False


def is_applied() -> bool:
    return _applied