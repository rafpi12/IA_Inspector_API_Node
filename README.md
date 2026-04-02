# IA Inspector API Node

Custom ComfyUI nodes for Gemini and Grok API access.  
API keys are read directly from pod environment variables — no JSON config files.

## Nodes

- **Gemini API (IA Inspector)** — Google Gemini, supports image input
- **Grok API (IA Inspector)** — xAI Grok, text only

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
| `grok-3-fast` | Fast, recommended for most tasks |
| `grok-3` | Standard quality |
| `grok-4-fast` | Latest gen fast |
| `grok-4` | Latest gen flagship |
| `grok-3-mini-fast` | Lightest option |
| `grok-3-mini` | Mini standard |
