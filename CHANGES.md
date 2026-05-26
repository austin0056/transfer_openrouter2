# transfer_openrouter2 — Changes vs upstream

This fork adds two features on top of [`austin0056/transfer_openrouter`](https://github.com/austin0056/transfer_openrouter):

1. **Native four-field cache billing** — emit `cache_creation_input_tokens` /
   `cache_read_input_tokens` so dispatch layers (LiteLLM / New-API / One-API)
   that already model the four canonical token fields can apply their own
   per-field pricing correctly, including a separate "cache write" line item
   in the billing UI.
2. **Reasoning-effort suffix routing** — clients can pick `<model>-low`,
   `-medium`, `-high`, `-xhigh`, `-max` to drive OpenRouter `reasoning.effort`
   or Anthropic `thinking.budget_tokens` without touching the upstream model id.

---

## 1. Native cache billing (`cache_billing_mode`)

### Why

OpenRouter usage looks like:

```jsonc
{
  "prompt_tokens": 8200,                    // fresh + cache_read + cache_write
  "completion_tokens": 6,
  "prompt_tokens_details": {
    "cached_tokens": 0,                     // cache read
    "cache_write_tokens": 6555              // cache write
  }
}
```

Most dispatch layers (LiteLLM-style) expect:

```jsonc
{
  "prompt_tokens": 1645,                    // fresh only
  "completion_tokens": 6,
  "cache_creation_input_tokens": 6555,
  "cache_read_input_tokens":     0
}
```

with a price table such as:

```jsonc
{
  "input_cost_per_token":              5e-6,
  "output_cost_per_token":              25e-6,
  "cache_creation_input_token_cost":  6.25e-6,
  "cache_read_input_token_cost":         5e-7
}
```

The original gateway only knew the legacy *additive* trick (fold `cache_write`
into `prompt_tokens` at a 1.25× / 2× premium), which means
`cache_creation_input_token_cost` was silently ignored and the billing UI
never reflected cache write activity.

### What changes

`Settings.cache_billing_mode` selects the output shape. Default is `"native"`:

| Mode       | prompt_tokens | cache_creation_input_tokens | cache_read_input_tokens |
| ---------- | ------------- | --------------------------- | ----------------------- |
| `native`   | fresh         | cache_write                 | cache_read              |
| `additive` | fresh + ⌊cw × multiplier⌋ | (not emitted)   | (not emitted)           |

Switch via the admin UI under **缓存与压缩 → 缓存计费模式** or with
`CONFIG_FILE`:

```jsonc
{ "cache_billing_mode": "native" }
```

### Verification

Tested against `https://transfer111.zeabur.app/v1` with `anthropic/claude-opus-4.7`:

| Phase  | Upstream `usage.cost` | Dispatch (native, prices match upstream) | Δ      |
| ------ | --------------------: | ---------------------------------------: | -----: |
| Write  | $0.04114875           | $0.04114875                              |  $0.00 |
| Read   | $0.00345750           | $0.00345750                              |  $0.00 |

(If your dispatch-side `cache_creation_input_token_cost` is set higher than
the upstream cache-write rate the dispatch total will be higher by design —
that's the resale margin, not a calculation error.)

---

## 2. Reasoning-effort suffix (`reasoning_suffix_enabled`)

### Why

Cursor (and many OpenAI-compatible clients) only let users pick a model from
the `/v1/models` list. There is no first-class way to switch the reasoning
budget per request without proliferating upstream model ids.

### What changes

When `reasoning_suffix_enabled` is on (default), every advertised model is
expanded into 6 variants:

```
claude-opus-4-7
claude-opus-4-7-low
claude-opus-4-7-medium
claude-opus-4-7-high
claude-opus-4-7-xhigh
claude-opus-4-7-max
```

When a request lands with a suffixed model, the gateway:

1. Strips the suffix and forwards the **original** upstream model id (so
   billing, caching, and routing all stay correct).
2. Maps the suffix to the right reasoning knob:

| Suffix    | OpenRouter `reasoning`                          | Anthropic `thinking.budget_tokens` |
| --------- | ----------------------------------------------- | ---------------------------------: |
| (none)    | not injected                                    | falls back to `thinking_budget_tokens` |
| `low`     | `{effort: "low"}`                               | 2 048                              |
| `medium`  | `{effort: "medium"}`                            | 8 192                              |
| `high`    | `{effort: "high"}`                              | 16 384                             |
| `xhigh`   | `{effort: "high", max_tokens: 32 768}`          | 32 768                             |
| `max`     | `{effort: "high", max_tokens: 65 536}`          | 65 536                             |

Anthropic budgets are configurable per-suffix in the admin UI under
**模型列表 → 推理等级后缀**.

If the client already sends an explicit `reasoning_effort` / `reasoning`
field, the suffix is honored for routing only and the explicit value wins.

---

## Files changed

```
app/config.py       +12 lines   cache_billing_mode + reasoning_budget_*
app/sse_stream.py   ~25 lines   convert_usage_to_additive(mode=...)
app/upstream.py     +85 lines   parse_reasoning_suffix() + builders
app/main.py         ~10 lines   /v1/models variant expansion + mode passthrough
app/admin.py        +50 lines   admin UI for the new options
```

No database migration. Defaults preserve existing behaviour for the
reasoning suffix (off by env override → matches upstream) but flip cache
billing to `native` — set `cache_billing_mode=additive` in `CONFIG_FILE`
to keep the legacy behaviour.
