import os
import json
from PIL import Image
import numpy as np

# ── API keys from pod environment variables ─────────────────────
def get_gemini_key():
    return os.environ.get("GEMINI_API", "").strip()

def get_grok_key():
    return os.environ.get("GROK_API", "").strip()


# ── Safety map (Gemini only) ────────────────────────────────────
SAFETY_THRESHOLD_MAP = {
    "Block None":   "BLOCK_NONE",
    "Block Low":    "BLOCK_LOW_AND_ABOVE",
    "Block Medium": "BLOCK_MEDIUM_AND_ABOVE",
    "Block High":   "BLOCK_HIGH_AND_ABOVE",
}

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite-preview",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-exp",
]

# grok-3 / grok-4-fast etc. were retired on May 15, 2026 (silently redirected to grok-4.3).
GROK_MODELS = [
    "grok-4.7",
    "grok-4.3",
    "grok-4.7-latest",
]

GROK_REASONING = ["default", "none", "low", "medium", "high"]


def _frames_to_jpeg_b64(image, max_frames, resize_to):
    """Return a list of base64 JPEGs from an IMAGE batch, sampling up to max_frames evenly."""
    import base64, io
    total = int(image.shape[0])
    n = max(1, min(int(max_frames), total))
    if n == 1:
        idx = [0]
    else:
        idx = sorted({round(k * (total - 1) / (n - 1)) for k in range(n)})
    out = []
    for i in idx:
        arr = 255. * image[i].cpu().numpy()
        img = Image.fromarray(np.uint8(np.clip(arr, 0, 255)))
        if resize_to != "None":
            t = int(resize_to)
            img.thumbnail((t, t), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        out.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return out


# ══════════════════════════════════════════════════════════════════
#  GEMINI NODE
# ══════════════════════════════════════════════════════════════════
class IAInspectorGemini:
    CATEGORY  = "IA_Inspector/API"
    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("text",)
    FUNCTION  = "generate_text"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (GEMINI_MODELS,),
                "max_output_tokens": ("INT",   {"default": 1024, "min": 1, "max": 8192}),
                "temperature":       ("FLOAT", {"default": 0.9,  "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p":             ("FLOAT", {"default": 0.9,  "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k":             ("INT",   {"default": 50,   "min": 1,   "max": 100}),
                "seed":              ("INT",   {"default": 0,    "min": 0,   "max": 0xffffffffffffffff}),
            },
            "optional": {
                "user_instructions": ("STRING", {"multiline": True, "default": ""}),
                "image":             ("IMAGE",),
                "resize_image_to":   (["None", "512", "768", "1024"], {"default": "None"}),
                "thinking_mode":     (["disable", "enable"],           {"default": "disable"}),
                "safety_threshold":  (list(SAFETY_THRESHOLD_MAP.keys()), {"default": "Block None"}),
            }
        }

    def generate_text(self, system_prompt, model, max_output_tokens, temperature,
                      top_p, top_k, seed,
                      user_instructions="", image=None, resize_image_to="None",
                      thinking_mode="disable", safety_threshold="Block None"):

        api_key = get_gemini_key()
        if not api_key:
            return ("Error: GEMINI_API environment variable not set.",)

        try:
            from google import genai
            from google.genai import types
        except ImportError:
            return ("Error: google-genai package not installed. Run: pip install google-genai",)

        try:
            client = genai.Client(api_key=api_key)
        except Exception as e:
            return (f"Error initializing Gemini client: {e}",)

        # Build contents
        contents = []
        if user_instructions and user_instructions.strip():
            contents.append(user_instructions.strip())

        if image is not None:
            try:
                i = 255. * image[0].cpu().numpy()
                img = Image.fromarray(np.uint8(i))
                if resize_image_to != "None":
                    target_size = int(resize_image_to)
                    img.thumbnail((target_size, target_size), Image.LANCZOS)
                contents.append(img)
            except Exception as e:
                return (f"Error processing image: {e}",)

        if not contents:
            if system_prompt and system_prompt.strip():
                contents.append("Please respond based on your system instructions.")
            else:
                return ("Error: No input provided.",)

        # Safety settings
        safety_settings = [
            types.SafetySetting(category=cat, threshold=SAFETY_THRESHOLD_MAP[safety_threshold])
            for cat in [
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
            ]
        ]

        generate_config = {
            "system_instruction": system_prompt if system_prompt.strip() else None,
            "temperature":        temperature,
            "top_p":              top_p,
            "top_k":              top_k,
            "max_output_tokens":  max_output_tokens,
            "safety_settings":    safety_settings,
        }

        if thinking_mode == "enable":
            generate_config["thinking_config"] = {"include_thoughts": True}

        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**generate_config),
            )
            return (response.text,)
        except Exception as e:
            return (f"Gemini API Error: {str(e)}",)


# ══════════════════════════════════════════════════════════════════
#  GROK NODE
# ══════════════════════════════════════════════════════════════════
class IAInspectorGrok:
    CATEGORY  = "IA_Inspector/API"
    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("text",)
    FUNCTION  = "generate_text"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "model":         (GROK_MODELS,),
                "max_tokens":    ("INT",   {"default": 1024, "min": 1,   "max": 16384}),
                "temperature":   ("FLOAT", {"default": 0.9,  "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p":         ("FLOAT", {"default": 0.9,  "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed":          ("INT",   {"default": 0,    "min": 0,   "max": 2147483647}),
            },
            "optional": {
                "user_instructions": ("STRING", {"multiline": True, "default": ""}),
                # image: a single image OR a video frame batch (sampled with max_frames)
                "image":             ("IMAGE",),
                "max_frames":        ("INT", {"default": 1, "min": 1, "max": 32,
                                              "tooltip": "If 'image' is a batch (video frames), send up to this many frames, evenly spaced, in time order."}),
                # image2: extra image(s) sent AFTER 'image' (e.g. the character reference)
                "image2":            ("IMAGE",),
                "resize_image_to":   (["None", "512", "768", "1024"], {"default": "None"}),
                "reasoning_effort":  (GROK_REASONING, {"default": "default"}),
            }
        }

    def generate_text(self, system_prompt, model, max_tokens, temperature, top_p, seed,
                      user_instructions="", image=None, max_frames=1, image2=None,
                      resize_image_to="None", reasoning_effort="default"):

        api_key = get_grok_key()
        if not api_key:
            return ("Error: GROK_API environment variable not set.",)

        try:
            from openai import OpenAI
        except ImportError:
            return ("Error: openai package not installed. Run: pip install openai",)

        try:
            client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1", timeout=600)
        except Exception as e:
            return (f"Error initializing Grok client: {e}",)

        user_text = user_instructions.strip() if user_instructions and user_instructions.strip() \
                    else "Please respond based on your system instructions."

        images_b64 = []
        try:
            if image is not None:
                images_b64 += _frames_to_jpeg_b64(image, max_frames, resize_image_to)
            if image2 is not None:
                images_b64 += _frames_to_jpeg_b64(image2, 1, resize_image_to)
        except Exception as e:
            return (f"Error processing image: {e}",)

        if images_b64:
            user_content = [{"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{b}", "detail": "high"}}
                            for b in images_b64]
            user_content.append({"type": "text", "text": user_text})
        else:
            user_content = user_text

        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": user_content})

        kwargs = dict(model=model, messages=messages, max_tokens=max_tokens,
                      temperature=temperature, top_p=top_p, seed=seed)
        if reasoning_effort != "default":
            kwargs["reasoning_effort"] = reasoning_effort

        try:
            response = client.chat.completions.create(**kwargs)
            return (response.choices[0].message.content or "",)
        except Exception as e:
            return (f"Grok API Error: {str(e)}",)


# ══════════════════════════════════════════════════════════════════
#  MAPPINGS
# ══════════════════════════════════════════════════════════════════
NODE_CLASS_MAPPINGS = {
    "IAInspectorGemini": IAInspectorGemini,
    "IAInspectorGrok":   IAInspectorGrok,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAInspectorGemini": "Gemini API (IA Inspector)",
    "IAInspectorGrok":   "Grok API (IA Inspector)",
}
