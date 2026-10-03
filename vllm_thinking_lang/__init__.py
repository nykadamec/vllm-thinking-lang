"""vllm-thinking-lang: control the language a reasoning model thinks in."""

from .config import LANGUAGES, OFF, resolve_language

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