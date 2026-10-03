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