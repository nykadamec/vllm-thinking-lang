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
    """Record intent; the tokenizer does not exist at plugin-registration time."""
    global _applied
    if _applied or not layer_enabled("template"):
        return
    _applied = True
    logger.info("thinking-lang: template layer enabled")


def unapply() -> None:
    global _applied
    _applied = False