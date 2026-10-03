# vllm-thinking-lang

Control the language a reasoning model thinks in, for any model served through
vLLM.

Reasoning models think in whatever language their training data pushed them
towards, regardless of the language the user writes in. This plugin makes that
language controllable — globally, or per request.

## Why a plugin is needed

vLLM already has a control channel: `chat_template_kwargs`. Passing
`reasoning_effort` works out of the box, because `ChatCompletionRequest`
forwards it into the template kwargs.

But vLLM filters those kwargs. `resolve_chat_template_kwargs` drops any kwarg
the chat template does not declare as a variable, so a `thinking_language`
value you send is **silently discarded** for most models. That single filter
is the reason this plugin exists.

## Install

```bash
uv pip install "vllm-thinking-lang @ git+https://github.com/nykadamec/vllm-thinking-lang.git"
```

The plugin registers itself via the `vllm.general_plugins` entry point; vLLM
loads it in every process it creates.

## Set the language

Globally, for the whole server:

```bash
VLLM_THINKING_LANG=cs vllm serve <model> --reasoning-parser qwen3
```

Or per request, which wins over the environment default:

```json
{
  "messages": [{"role": "user", "content": "9.11 vs 9.8?"}],
  "chat_template_kwargs": {"thinking_language": "cs"}
}
```

Set `"thinking_language": "off"` to disable the plugin for a single request.

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `VLLM_THINKING_LANG` | unset | default language for all requests |
| `VLLM_THINKING_LANG_TEMPLATE` | unset | path to a `.jinja` to install as the chat template |
| `VLLM_THINKING_LANG_INJECT` | `on` | enable system-message injection |
| `VLLM_THINKING_LANG_FILTER` | `on` | enable the kwarg filter patch |
| `VLLM_PLUGINS` | unset | when set, vLLM loads only the named plugins |

Supported languages: `cs`, `en`. Anything else is ignored and logged once at
startup.

With no `VLLM_THINKING_LANG` and no `thinking_language` in any request, the
plugin is completely inert.

## How it works

Three layers, each independently switchable, degrading rather than failing.

**Filter patch (on by default).** Wraps
`vllm.renderers.hf.resolve_chat_template_kwargs` so `thinking_language`
survives the kwarg filter for templates that declare it.

**Template patch (off by default).** Installs a chat template from
`VLLM_THINKING_LANG_TEMPLATE`. Off by default because a chat template is
inherently per-model. If the model's existing template already mentions
`thinking_language`, yours is not installed — an explicit `--chat-template`
choice is never clobbered. A reference template ships in
`vllm_thinking_lang/thinking.jinja`; copy it and point the variable at your
copy.

**System injection (on by default, and the one that works everywhere).**
Appends a language instruction to the request's system message, or prepends a
system message when there is none. This is the only model-agnostic layer, which
is why it is the default. An existing system prompt is appended to, never
replaced. Requests with `enable_thinking: false` are skipped, as are prefilled
continuations and requests carrying multimodal system content.

## Limitations

**Control is prompt-level. The model obeys when it chooses to.** This works on
Qwen3.x and Gemma 4. It is a strong instruction, not a guarantee.

Token-level enforcement is deliberately not implemented. Forcing a language
across the reasoning phase would break reasoning markup and tool-call tags,
because a token-level filter has to operate across tag boundaries and would
produce reasoning that is partly in one language and partly in another —
worse than no control at all. vLLM also has no general-plugin hook for logits
processors.

If prompt-level control proves insufficient for a model, the next step is a
different tool, not a deeper version of this plugin.

## Compatibility

Tested against **vLLM 0.30.0** and **Python 3.12**. The plugin patches
internals, so the supported version range is deliberately narrow. Widening it
requires running the test suite against each new version.

## License

Apache-2.0