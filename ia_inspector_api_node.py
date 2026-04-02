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
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash-lite-preview-09-2025",
    "gemini-2.5-flash-preview-09-2025",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-exp",
]

GROK_MODELS = [
    "grok-3-fast",
    "grok-3",
    "grok-4-fast",
    "grok-4",
    "grok-3-mini-fast",
    "grok-3-mini",
]


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
                "max_tokens":    ("INT",   {"default": 1024, "min": 1,   "max": 8192}),
                "temperature":   ("FLOAT", {"default": 0.9,  "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p":         ("FLOAT", {"default": 0.9,  "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed":          ("INT",   {"default": 0,    "min": 0,   "max": 0xffffffffffffffff}),
            },
            "optional": {
                "user_instructions": ("STRING", {"multiline": True, "default": ""}),
                "image":             ("IMAGE",),
                "resize_image_to":   (["None", "512", "768", "1024"], {"default": "None"}),
            }
        }

    def generate_text(self, system_prompt, model, max_tokens, temperature, top_p, seed,
                      user_instructions="", image=None, resize_image_to="None"):

        api_key = get_grok_key()
        if not api_key:
            return ("Error: GROK_API environment variable not set.",)

        try:
            from openai import OpenAI
        except ImportError:
            return ("Error: openai package not installed. Run: pip install openai",)

        import base64, io

        try:
            client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        except Exception as e:
            return (f"Error initializing Grok client: {e}",)

        # Build user message content
        user_text = user_instructions.strip() if user_instructions and user_instructions.strip()                     else "Please respond based on your system instructions."

        if image is not None:
            try:
                i = 255. * image[0].cpu().numpy()
                img = Image.fromarray(np.uint8(i))
                if resize_image_to != "None":
                    target_size = int(resize_image_to)
                    img.thumbnail((target_size, target_size), Image.LANCZOS)
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=90)
                b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            except Exception as e:
                return (f"Error processing image: {e}",)

            user_content = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]
        else:
            user_content = user_text

        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": user_content})

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
            )
            return (response.choices[0].message.content,)
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
