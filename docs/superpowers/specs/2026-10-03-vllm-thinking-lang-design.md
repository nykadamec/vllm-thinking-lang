# vllm-thinking-lang — Design Spec

Date: 2026-10-03
Status: approved, pending implementation
Target: vLLM 0.30.0

## Problem

We want to control the language in which a reasoning model does its thinking,
for any model served through vLLM. Two concrete failures motivate this:

1. **Effort does not reach the model.** Hermes sends `reasoning_effort` as a
   top-level field. As of vLLM 0.30.0 this now works: `ChatCompletionRequest`
   declares `reasoning_effort` (chat_completion/protocol.py:245) and
   `build_chat_params` forwards it into `chat_template_kwargs` (line 589).
   This is resolved upstream and is *not* part of this plugin's job — but it
   establishes that `chat_template_kwargs` is the working control channel.

2. **Thinking language cannot be controlled at all.** `chat_template_kwargs`
   is the right channel, but `resolve_chat_template_kwargs`
   (vllm/renderers/hf.py:709) filters out every kwarg the chat template does
   not declare as a variable. A `thinking_language` kwarg sent to a model whose
   template has no such variable is silently discarded. Most models have no
   such variable, so per-request language control is impossible today.

## Goal

A `vllm.general_plugins` out-of-tree plugin that makes thinking language
controllable for **any** model served through vLLM, without requiring
per-model template authoring.

Explicit non-goal: enforcing the language at the token level. See
[Non-goals](#non-goals).

## Approach

Three independent layers with graceful degradation. Each layer can be disabled
independently; each produces correct behaviour on its own.

| Layer | Works when | Mechanism | Default |
|---|---|---|---|
| 1. Filter hook | template declares the variable | patch `resolve_chat_template_kwargs` to add `thinking_language` to the accept set | on |
| 2. Template patch | you supply a `.jinja` path | set `tokenizer.chat_template` to a patched template | **off** |
| 3. System injection | **always, any model** | patch `OpenAIServingChat`, prepend a system message per request | on |

Layer 3 is the default path because it is the only one that is model-agnostic.
Layers 1 and 2 exist because they are strictly better when available: a
template-level instruction reaches the model through the channel the model
was trained on, whereas an injected system message is an ordinary user-space
prompt.

### Layer 1 — filter hook

Monkey-patch `vllm.renderers.hf.resolve_chat_template_kwargs` so that
`thinking_language` is added to `accept_vars`. Without this, a request carrying
`chat_template_kwargs={"thinking_language": "cs"}` is silently stripped by the
filter before the template ever sees it.

The patch wraps the original function and post-extends its result. It does not
reimplement the filter, so upstream changes to the accept-set logic keep
working.

### Layer 2 — template patch

Off by default, because a template is inherently per-model and a universal
repository cannot ship a patched template for every model family.

When enabled with an explicit path, the plugin loads that `.jinja` file and
assigns it to `tokenizer.chat_template`, idempotently: patching twice yields the
same result. If the loaded template already declares `thinking_language`, the
patch is a no-op.

The template is expected to render a short instruction at the start of the
generation prompt, keyed on `thinking_language`.

### Layer 3 — system injection

Patch `OpenAIServingChat` so that, per request, a system message is prepended
to `messages` carrying a language instruction.

Rules, all of which are load-bearing:

- If the request already has a system message, the instruction is **appended**
  to it. Never replaced.
- If `chat_template_kwargs["enable_thinking"]` is false, no injection happens.
  On models like Gemma 4 that disables reasoning entirely, so an instruction
  about thinking would be out of scope for that request.
- If the resolved language is `off`, nothing is injected and layers 1 and 2 are
  also suppressed for that request.

## Configuration

Language resolution order, first hit wins:

1. `chat_template_kwargs["thinking_language"]` in the request
2. `VLLM_THINKING_LANG` environment variable
3. unset — the plugin is completely inert and vLLM behaves as if it were not
   installed

Supported values: `cs`, `en`, `off`. Unknown values are ignored (treated as
unset) and logged once at startup, never per request.

Layer toggles:

| Variable | Default | Effect |
|---|---|---|
| `VLLM_THINKING_LANG` | unset | default language |
| `VLLM_THINKING_LANG_TEMPLATE` | `off` | path to a `.jinja` to patch in (layer 2) |
| `VLLM_THINKING_LANG_INJECT` | `on` | enable layer 3 |
| `VLLM_THINKING_LANG_FILTER` | `on` | enable layer 1 |
| `VLLM_PLUGINS` | unset | if set, vLLM loads only the named plugins |

## Package layout

```
vllm-thinking-lang/
├── vllm_thinking_lang/
│   ├── __init__.py          # register() entry point, re-entrant
│   ├── config.py            # env parsing, language resolution
│   ├── injection.py         # layer 3
│   ├── template_patch.py    # layer 2
│   ├── filter_patch.py      # layer 1
│   └── thinking.jinja       # reference template for layer 2
├── tests/
│   ├── test_config.py
│   ├── test_filter_patch.py
│   ├── test_template_patch.py
│   └── test_injection.py
├── pyproject.toml
└── README.md
```

Each module has one responsibility and can be understood without reading the
others. `config.py` in particular is pure: no vLLM imports, no patching, so the
resolution rules are testable without vLLM installed.

Entry point:

```toml
[project.entry-points."vllm.general_plugins"]
thinking_lang = "vllm_thinking_lang:register"
```

`register()` must be re-entrant: `load_plugins_by_group()` may call it more than
once, and the vLLM plugin guidelines require it.

## Installation and deployment

```bash
uv pip install "vllm-thinking-lang @ git+https://github.com/nykadamec/vllm-thinking-lang.git"
```

For a vLLM serve deployment, set `VLLM_THINKING_LANG` in the container
environment. No deployment script needs to read it or pass any vLLM CLI flag;
the plugin reads the environment itself. Setting `VLLM_PLUGINS=thinking_lang`
is recommended when other vLLM plugins are present in the same container, so
that only this plugin's registration runs.

## Error handling

- Unknown language value: log once at startup, treat as unset. Never raise — a
  bad env var must not prevent the server from starting.
- Missing layer-2 template file: log an error, disable layer 2, continue with
  layers 1 and 3. Never raise at import or startup time.
- A vLLM version where a patched symbol does not exist: log a warning naming the
  layer that was skipped, continue with the remaining layers. The plugin must
  never be the reason a server fails to boot.
- Injection must never raise into the request path. Failures are logged and the
  request proceeds unmodified.

## Testing

Unit tests run without a GPU and, where possible, without vLLM installed.

- `config.py`: full resolution order, `off`, unknown values, env parsing.
- `filter_patch.py`: fake tokenizer; a template that declares
  `thinking_language` passes the filter, one that does not is stripped; the
  patch composes with upstream filter behaviour rather than replacing it.
- `template_patch.py`: idempotence, `cs`/`en`/unset branches, and that an
  existing generation block and tool-call markup are not corrupted.
- `injection.py`: appends to an existing system message rather than replacing
  it; injects when no system message exists; does not inject when
  `enable_thinking` is false or language is `off`.
- Re-entrancy: calling `register()` twice changes nothing.

End-to-end verification, run once on Modal after the unit tests pass:
Qwen3.8 and Gemma 4 served through the plugin, request with
`thinking_language=cs`, assert the `reasoning` field is in Czech, and assert
prefix-cache behaviour is unchanged.

## Non-goals

**Token-level enforcement is out of scope.** Forcing Czech tokens during the
reasoning phase would break reasoning markup (`<think>`, tool-call tags,
structured-output schemas in `content`), because a token-level filter has to
operate across tag boundaries and would produce reasoning that is partly Czech
and partly English — worse than no control at all. It is also not a plugin
hook: vLLM has no general-plugin group for logits processors.

Consequently **layer 3 is prompt-level. The model obeys when it chooses to.**
On Qwen3.8 and Gemma 4 it does. If it proves insufficient, the next step is a
different tool, not a deeper version of this plugin.

## Distribution and versioning

The repository is standalone and model-agnostic, a sibling of
`vllm-gguf-plugin`. It declares a supported vLLM version range and is tested
against vLLM 0.30.0. Because it monkey-patches internals, the supported range
is deliberately narrow; widening it requires running the unit tests against
each new version.

## Open questions

None blocking implementation. One detail is deliberately deferred: whether
layer 3 patches `OpenAIServingChat` directly or subclasses it and monkey-patches
the serving-class factory. Both are viable; the choice is made during
implementation based on which one avoids duplicating upstream constructor
arguments.