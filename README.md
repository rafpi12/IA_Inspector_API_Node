# IA Inspector API Node

Custom ComfyUI nodes for Gemini, OpenAI and Grok API access, with token usage and credit display.

## Nodes

- **Gemini API (IA Inspector)**: Google Gemini, text + images
- **OpenAI API (IA Inspector)**: OpenAI GPT models, text + images
- **Grok API (IA Inspector)**: xAI Grok, text + images
- **API Credit (IA Inspector)**: credit / spend for the three providers
- **Grok Balance (IA Inspector)**: xAI prepaid credit only (kept for older workflows)

The three API nodes share the same image inputs and have two outputs: `text` and `usage`
(model, input / output tokens of the call, estimated cost when a price is set).

## Installation

```bash
cd /workspace/ComfyUI/custom_nodes
git clone https://github.com/rafpi12/IA_Inspector_API_Node
pip install google-genai openai pillow
```

## API keys

Keys are read from `api_keys.json` in the node folder first, then from pod environment variables.
The file is read on every run: no ComfyUI restart needed after editing it. It is in `.gitignore`.

```bash
cd /workspace/ComfyUI/custom_nodes/IA_Inspector_API_Node
cp api_keys.example.json api_keys.json
nano api_keys.json
```

| `api_keys.json` | Environment variable | Used for |
|-----------------|----------------------|----------|
| `gemini_api_key` | `GEMINI_API` | Gemini node |
| `openai_api_key` | `OPENAI_API` or `OPENAI_API_KEY` | OpenAI node |
| `grok_api_key` | `GROK_API` | Grok node |
| `openai_admin_key` | `OPENAI_ADMIN_KEY` | OpenAI spend in the Credit node |
| `grok_management_key` | `GROK_MANAGEMENT_KEY` | xAI balance |
| `grok_team_id` | `GROK_TEAM_ID` | xAI balance |

## Credit display

What each provider allows:

| Provider | What the Credit node shows | Source |
|----------|----------------------------|--------|
| Grok | Real prepaid balance | xAI Management API (`grok_management_key` + `grok_team_id`) |
| OpenAI | Real spend since a date, and credit left if you declare your credit | OpenAI Costs API (`openai_admin_key`, an admin key from platform.openai.com). OpenAI has no balance endpoint. |
| Gemini | Estimate only: tokens logged by this node, cost if prices are set | Google has no balance API. Real balance: AI Studio > Billing. |

`credit` block: `amount` is the credit you bought (USD), `since` the date it started (OpenAI only, defaults to the 1st of the month).

`prices` block: `"model-id": [input, output]` in USD per 1M tokens. Used for the per-call estimate and for the
Gemini estimate (and OpenAI when no admin key is set). Take the values from the provider pricing page.

Token counters are stored in `usage_log.json` in the node folder. Delete the file to reset them.

## Models

The model list of each node is fetched from the provider when the ComfyUI page loads (needs the API key),
so new models appear without updating the node. Without a key or network, a built-in list is used.
Add ids manually with `extra_models` in `api_keys.json`.

A model saved in an older workflow stays accepted even if it left the list; the provider answers with an error if it was shut down.

## Image inputs (all three API nodes)

- `image`: one image, or a video frame batch. With a batch, `max_frames` frames are sent, evenly spaced and in time order.
- `image2`: an extra image sent after `image` (e.g. a character reference).
- `resize_image_to`: 768 is a good default for video frames (fewer tokens).
- `reasoning_effort` (OpenAI, Grok): `default` sends nothing; other values set it explicitly.

OpenAI: reasoning tokens count against `max_tokens`. If the answer comes back empty, raise it.
