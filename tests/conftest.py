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