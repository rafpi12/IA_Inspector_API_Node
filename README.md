# IA Inspector API Node

Custom ComfyUI nodes for Gemini and Grok API access.  
API keys are read directly from pod environment variables — no JSON config files.

## Nodes

- **Gemini API (IA Inspector)** — Google Gemini, supports image input
- **Grok Balance (IA Inspector)** — shows the xAI prepaid credit (Management API)
- **Grok API (IA Inspector)** — xAI Grok, text + images (single image or video frame batch)

## Installation

Clone into your ComfyUI custom_nodes folder:
```bash
cd /workspace/ComfyUI/custom_nodes
git clone https://github.com/rafpi12/IA_Inspector_API_Node
```

## Environment Variables

Set these in your RunPod pod template:

| Variable | Description |
|----------|-------------|
| `GEMINI_API` | Google AI Studio API key |
| `GROK_API` | xAI console API key |
| `GROK_MANAGEMENT_KEY` | xAI management key (Grok Balance node only) |
| `GROK_TEAM_ID` | xAI team ID (Grok Balance node only) |

## Dependencies

```bash
pip install google-genai openai pillow
```

## Gemini Models

| Model | Notes |
|-------|-------|
| `gemini-3.1-flash-lite-preview` | Fastest, recommended |
| `gemini-2.5-flash-lite-preview-09-2025` | Stable, 1000 req/day free |
| `gemini-2.5-flash-preview-09-2025` | Better quality |
| `gemini-2.0-flash` | Previous gen stable |
| `gemini-2.0-flash-lite` | Previous gen lite |

## Grok Models

| Model | Notes |
|-------|-------|
| `grok-4.7` | Flagship, recommended |
| `grok-4.3` | Cheaper ($1.25 / $2.50 per 1M tokens) |
| `grok-4.7-latest` | Always the newest version |

grok-3, grok-4-fast and grok-4-1-fast were retired on May 15, 2026 (xAI silently redirects them to grok-4.3).

## Grok image inputs

- `image`: one image, or a video frame batch. With a batch, `max_frames` frames are sent, evenly spaced and in time order.
- `image2`: an extra image sent after `image` (e.g. a character reference).
- `resize_image_to`: 768 is a good default for video frames (fewer tokens).
- `reasoning_effort`: `default` sends nothing; `none` / `low` / `medium` / `high` set it explicitly.
