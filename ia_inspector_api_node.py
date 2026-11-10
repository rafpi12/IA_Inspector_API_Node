import os
import re
import json
import time
import datetime
from PIL import Image
import numpy as np

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(NODE_DIR, "api_keys.json")
USAGE_PATH = os.path.join(NODE_DIR, "usage_log.json")


# ══════════════════════════════════════════════════════════════════
#  CONFIG: api_keys.json in the node folder, then pod env variables
# ══════════════════════════════════════════════════════════════════
# name -> (key in api_keys.json, env variables tried in order)
KEY_SOURCES = {
    "gemini":           ("gemini_api_key",      ["GEMINI_API", "GEMINI_API_KEY"]),
    "openai":           ("openai_api_key",      ["OPENAI_API", "OPENAI_API_KEY"]),
    "grok":             ("grok_api_key",        ["GROK_API", "XAI_API_KEY"]),
    "openai_admin":     ("openai_admin_key",    ["OPENAI_ADMIN_KEY"]),
    "grok_management":  ("grok_management_key", ["GROK_MANAGEMENT_KEY"]),
    "grok_team":        ("grok_team_id",        ["GROK_TEAM_ID"]),
}


def load_config():
    """Read api_keys.json on every call, so edits apply without restarting ComfyUI."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"[IA Inspector] api_keys.json could not be read ({e}), using env variables only.")
        return {}


def get_key(name):
    json_key, env_names = KEY_SOURCES[name]
    value = str(load_config().get(json_key, "") or "").strip()
    if value:
        return value
    for env in env_names:
        value = os.environ.get(env, "").strip()
        if value:
            return value
    return ""


def missing_key_message(name):
    json_key, env_names = KEY_SOURCES[name]
    return (f"Error: no key found. Set \"{json_key}\" in {CONFIG_PATH} "
            f"or the {env_names[0]} environment variable.")


# ══════════════════════════════════════════════════════════════════
#  MODEL LISTS: live from each provider, fallback to these defaults
# ══════════════════════════════════════════════════════════════════
FALLBACK_MODELS = {
    "gemini": [
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.6-flash",
        "gemini-3.8-flash",
        "gemini-3.1-pro-preview",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro",
        # kept so older workflows still load (Google lists it as shut down)
        "gemini-3.1-flash-lite-preview",
    ],
    "openai": [
        "gpt-5.4-mini",
        "gpt-5.6-luna",
        "gpt-5.6-terra",
        "gpt-5.6-sol",
        "gpt-5.5",
        "gpt-5.4",
        "chat-latest",
    ],
    # grok-3 / grok-4-fast etc. were retired on May 15, 2026 (silently redirected to grok-4.3).
    "grok": [
        "grok-4.7",
        "grok-4.3",
        "grok-4.7-latest",
    ],
}

_EXCLUDE = {
    "gemini": ["tts", "image", "live", "embedding", "audio", "robotics", "computer-use",
               "transcribe", "omni", "banana", "translate", "native", "aqa", "gemma"],
    "openai": ["audio", "realtime", "transcribe", "tts", "image", "embedding", "moderation",
               "search", "codex", "instruct", "oss", "cyber", "rosalind", "whisper", "dall",
               "sora", "davinci", "babbage", "computer-use", "deep-research"],
    "grok":   ["imagine", "image", "video", "tts", "voice"],
}
_DATED = re.compile(r"-\d{4}-\d{2}-\d{2}$|-\d{4}$|-\d{3}$")
_MODEL_CACHE = {}      # provider -> (timestamp, [ids])
_MODEL_TTL = 3600


def _http_json(url, headers, timeout=20):
    import urllib.request
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _keep_model(provider, model_id):
    low = model_id.lower()
    if provider == "gemini" and not low.startswith("gemini-"):
        return False
    if provider == "openai" and not low.startswith(("gpt-", "chat", "o1", "o3", "o4")):
        return False
    if provider == "grok" and not low.startswith("grok-"):
        return False
    if any(word in low for word in _EXCLUDE[provider]):
        return False
    if provider != "grok" and _DATED.search(low):
        return False   # hide dated snapshots, keep the aliases
    return True


def _fetch_live_models(provider):
    key = get_key(provider)
    if not key:
        return []
    if provider == "gemini":
        data = _http_json("https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000",
                          {"x-goog-api-key": key}, timeout=5)
        ids = [m.get("name", "").split("/")[-1] for m in data.get("models", [])
               if "generateContent" in m.get("supportedGenerationMethods", [])]
    else:
        base = "https://api.openai.com/v1" if provider == "openai" else "https://api.x.ai/v1"
        data = _http_json(f"{base}/models", {"Authorization": f"Bearer {key}"}, timeout=5)
        ids = [m.get("id", "") for m in data.get("data", [])]
    return sorted({i for i in ids if i and _keep_model(provider, i)}, reverse=True)


def model_choices(provider):
    """Models shown in the node: extra_models from api_keys.json, then defaults, then the live list."""
    extra = load_config().get("extra_models", {})
    extra = [str(m) for m in extra.get(provider, [])] if isinstance(extra, dict) else []

    cached = _MODEL_CACHE.get(provider)
    if cached and time.time() - cached[0] < _MODEL_TTL:
        live = cached[1]
    else:
        try:
            live = _fetch_live_models(provider)
        except Exception as e:
            print(f"[IA Inspector] live {provider} model list unavailable ({e}), using built-in list.")
            live = []
        _MODEL_CACHE[provider] = (time.time(), live)

    out = []
    for m in extra + FALLBACK_MODELS[provider] + live:
        if m not in out:
            out.append(m)
    return out


# ══════════════════════════════════════════════════════════════════
#  USAGE LOG (tokens per provider / model) + cost estimate
# ══════════════════════════════════════════════════════════════════
def _load_usage():
    try:
        with open(USAGE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _price(model):
    """(input, output) USD per 1M tokens from the "prices" block of api_keys.json, or None."""
    prices = load_config().get("prices", {})
    p = prices.get(model) if isinstance(prices, dict) else None
    try:
        return float(p[0]), float(p[1])
    except Exception:
        return None


def _cost(model, prompt_tokens, completion_tokens):
    p = _price(model)
    if p is None:
        return None
    return (prompt_tokens * p[0] + completion_tokens * p[1]) / 1_000_000


def record_usage(provider, model, prompt_tokens, completion_tokens):
    """Add one call to usage_log.json and return a one-line summary."""
    prompt_tokens = int(prompt_tokens or 0)
    completion_tokens = int(completion_tokens or 0)
    try:
        data = _load_usage()
        prov = data.setdefault(provider, {})
        prov.setdefault("since", datetime.date.today().isoformat())
        m = prov.setdefault("models", {}).setdefault(model, {"calls": 0, "prompt": 0, "completion": 0})
        m["calls"] += 1
        m["prompt"] += prompt_tokens
        m["completion"] += completion_tokens
        tmp = USAGE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, USAGE_PATH)
    except Exception as e:
        print(f"[IA Inspector] usage_log.json not updated: {e}")

    line = f"{model} | {prompt_tokens} in / {completion_tokens} out tokens"
    c = _cost(model, prompt_tokens, completion_tokens)
    if c is not None:
        line += f" | est. ${c:.4f}"
    return line


def _local_usage_summary(provider):
    """Totals from usage_log.json: (since, calls, prompt, completion, est_cost or None, unpriced models)."""
    prov = _load_usage().get(provider, {})
    calls = prompt = completion = 0
    cost, unpriced = 0.0, []
    for model, m in prov.get("models", {}).items():
        calls += m.get("calls", 0)
        prompt += m.get("prompt", 0)
        completion += m.get("completion", 0)
        c = _cost(model, m.get("prompt", 0), m.get("completion", 0))
        if c is None:
            unpriced.append(model)
        else:
            cost += c
    has_price = calls > 0 and len(unpriced) < len(prov.get("models", {}))
    return prov.get("since", "?"), calls, prompt, completion, (cost if has_price else None), unpriced


def _credit_setting(provider):
    """(amount, since_date or None) from the "credit" block of api_keys.json."""
    credit = load_config().get("credit", {})
    c = credit.get(provider, {}) if isinstance(credit, dict) else {}
    try:
        amount = float(c.get("amount", 0) or 0)
    except Exception:
        amount = 0.0
    since = None
    try:
        if c.get("since"):
            since = datetime.date.fromisoformat(str(c["since"]))
    except Exception:
        since = None
    return amount, since


# ══════════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ══════════════════════════════════════════════════════════════════
SAFETY_THRESHOLD_MAP = {
    "Block None":   "BLOCK_NONE",
    "Block Low":    "BLOCK_LOW_AND_ABOVE",
    "Block Medium": "BLOCK_MEDIUM_AND_ABOVE",
    "Block High":   "BLOCK_HIGH_AND_ABOVE",
}

GROK_REASONING = ["default", "none", "low", "medium", "high"]
OPENAI_REASONING = ["default", "none", "minimal", "low", "medium", "high", "xhigh"]


def _sample_frames(image, max_frames, resize_to):
    """Return a list of PIL images from an IMAGE batch, sampling up to max_frames evenly, in time order."""
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
        out.append(img)
    return out


def _frames_to_jpeg_b64(image, max_frames, resize_to):
    """Return a list of base64 JPEGs from an IMAGE batch, sampling up to max_frames evenly."""
    import base64, io
    out = []
    for img in _sample_frames(image, max_frames, resize_to):
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=90)
        out.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return out


def _chat_completion(tag, provider, client, model, system_prompt, user_instructions,
                     image, max_frames, image2, resize_image_to,
                     max_tokens, temperature, top_p, seed, reasoning_effort, attempt_order):
    """OpenAI-compatible chat completion (xAI and OpenAI). Returns (text, usage_line)."""
    user_text = user_instructions.strip() if user_instructions and user_instructions.strip() \
                else "Please respond based on your system instructions."

    images_b64 = []
    try:
        if image is not None:
            images_b64 += _frames_to_jpeg_b64(image, max_frames, resize_image_to)
        if image2 is not None:
            images_b64 += _frames_to_jpeg_b64(image2, 1, resize_image_to)
    except Exception as e:
        return (f"Error processing image: {e}", "")

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

    base = dict(model=model, messages=messages)
    sampling = dict(temperature=temperature, top_p=top_p)
    if seed and seed > 0:  # xAI rejects seed = 0
        sampling["seed"] = seed
    if reasoning_effort != "default":
        sampling["reasoning_effort"] = reasoning_effort
    print(f"[{tag}] model={model} | system_prompt={len(system_prompt.strip()) if system_prompt else 0} chars"
          f" | user_instructions={len(user_text)} chars | images={len(images_b64)}")

    # Providers reject some parameters depending on the model: try full, then progressively leaner requests.
    variants = {
        "full":                  {**base, "max_tokens": max_tokens, **sampling},
        "no sampling params":    {**base, "max_tokens": max_tokens},
        "full (max_completion_tokens)": {**base, "max_completion_tokens": max_tokens, **sampling},
        "max_completion_tokens": {**base, "max_completion_tokens": max_tokens},
        "minimal":               dict(base),
    }
    errors = []
    for label in attempt_order:
        try:
            response = client.chat.completions.create(**variants[label])
            choice = response.choices[0]
            text = choice.message.content or ""
            usage_line = ""
            u = getattr(response, "usage", None)
            if u is not None:
                pt = getattr(u, "prompt_tokens", 0) or 0
                ct = getattr(u, "completion_tokens", 0) or 0
                print(f"[{tag}] tokens: prompt={pt} completion={ct} total={getattr(u, 'total_tokens', '?')}")
                usage_line = record_usage(provider, model, pt, ct)
            if errors:
                print(f"[{tag}] OK with '{label}' after: " + " | ".join(errors))
            if not text.strip():
                reason = getattr(choice, "finish_reason", "?")
                text = (f"{tag} returned an empty answer (finish_reason={reason}). "
                        f"With reasoning models, raise max_tokens: reasoning tokens count against it.")
            return (text, usage_line)
        except Exception as e:
            status = getattr(e, "status_code", None)
            body = getattr(e, "body", None)
            if body is None and getattr(e, "response", None) is not None:
                try:
                    body = e.response.text
                except Exception:
                    body = None
            msg = f"[{label}] {status}: {body if body else e}"
            print(f"[{tag}] {msg}")
            errors.append(msg)
            if status not in (400, 422):
                break
    return (f"{tag} API Error: " + " | ".join(str(m).replace("\n", " ") for m in errors), "")


# ══════════════════════════════════════════════════════════════════
#  GEMINI NODE
# ══════════════════════════════════════════════════════════════════
class IAInspectorGemini:
    CATEGORY  = "IA_Inspector/API"
    RETURN_TYPES  = ("STRING", "STRING")
    RETURN_NAMES  = ("text", "usage")
    FUNCTION  = "generate_text"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (model_choices("gemini"),),
                "max_output_tokens": ("INT",   {"default": 1024, "min": 1, "max": 65536}),
                "temperature":       ("FLOAT", {"default": 0.9,  "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p":             ("FLOAT", {"default": 0.9,  "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k":             ("INT",   {"default": 50,   "min": 1,   "max": 100}),
                "seed":              ("INT",   {"default": 0,    "min": 0,   "max": 0xffffffffffffffff}),
            },
            "optional": {
                "user_instructions": ("STRING", {"multiline": True, "default": ""}),
                # image: a single image OR a video frame batch (sampled with max_frames)
                "image":             ("IMAGE",),
                "resize_image_to":   (["None", "512", "768", "1024"], {"default": "None"}),
                "thinking_mode":     (["disable", "enable"],           {"default": "disable"}),
                "safety_threshold":  (list(SAFETY_THRESHOLD_MAP.keys()), {"default": "Block None"}),
                "max_frames":        ("INT", {"default": 1, "min": 1, "max": 32,
                                              "tooltip": "If 'image' is a batch (video frames), send up to this many frames, evenly spaced, in time order."}),
                # image2: extra image sent AFTER 'image' (e.g. the character reference)
                "image2":            ("IMAGE",),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, model):
        return True  # accept a model saved in an older workflow even if it left the list

    def generate_text(self, system_prompt, model, max_output_tokens, temperature,
                      top_p, top_k, seed,
                      user_instructions="", image=None, resize_image_to="None",
                      thinking_mode="disable", safety_threshold="Block None",
                      max_frames=1, image2=None):

        api_key = get_key("gemini")
        if not api_key:
            return (missing_key_message("gemini"), "")

        try:
            from google import genai
            from google.genai import types
        except ImportError:
            return ("Error: google-genai package not installed. Run: pip install google-genai", "")

        try:
            client = genai.Client(api_key=api_key)
        except Exception as e:
            return (f"Error initializing Gemini client: {e}", "")

        # Build contents
        contents = []
        if user_instructions and user_instructions.strip():
            contents.append(user_instructions.strip())

        n_images = 0
        try:
            if image is not None:
                frames = _sample_frames(image, max_frames, resize_image_to)
                contents += frames
                n_images += len(frames)
            if image2 is not None:
                frames = _sample_frames(image2, 1, resize_image_to)
                contents += frames
                n_images += len(frames)
        except Exception as e:
            return (f"Error processing image: {e}", "")

        if not contents:
            if system_prompt and system_prompt.strip():
                contents.append("Please respond based on your system instructions.")
            else:
                return ("Error: No input provided.", "")

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

        print(f"[IA Inspector Gemini] model={model} | system_prompt={len(system_prompt.strip())} chars"
              f" | images={n_images}")
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**generate_config),
            )
        except Exception as e:
            return (f"Gemini API Error: {str(e)}", "")

        usage_line = ""
        u = getattr(response, "usage_metadata", None)
        if u is not None:
            pt = getattr(u, "prompt_token_count", 0) or 0
            # thinking tokens are billed as output
            ct = (getattr(u, "candidates_token_count", 0) or 0) + (getattr(u, "thoughts_token_count", 0) or 0)
            print(f"[IA Inspector Gemini] tokens: prompt={pt} completion={ct} "
                  f"total={getattr(u, 'total_token_count', '?')}")
            usage_line = record_usage("gemini", model, pt, ct)

        text = response.text
        if not text:
            reason = "?"
            try:
                reason = response.candidates[0].finish_reason
            except Exception:
                pass
            feedback = getattr(response, "prompt_feedback", None)
            text = f"Gemini returned an empty answer (finish_reason={reason}, prompt_feedback={feedback})."
        return (text, usage_line)


# ══════════════════════════════════════════════════════════════════
#  GROK NODE
# ══════════════════════════════════════════════════════════════════
class IAInspectorGrok:
    CATEGORY  = "IA_Inspector/API"
    RETURN_TYPES  = ("STRING", "STRING")
    RETURN_NAMES  = ("text", "usage")
    FUNCTION  = "generate_text"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "model":         (model_choices("grok"),),
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

    @classmethod
    def VALIDATE_INPUTS(cls, model):
        return True

    def generate_text(self, system_prompt, model, max_tokens, temperature, top_p, seed,
                      user_instructions="", image=None, max_frames=1, image2=None,
                      resize_image_to="None", reasoning_effort="default"):

        api_key = get_key("grok")
        if not api_key:
            return (missing_key_message("grok"), "")

        try:
            from openai import OpenAI
        except ImportError:
            return ("Error: openai package not installed. Run: pip install openai", "")

        try:
            client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1", timeout=600)
        except Exception as e:
            return (f"Error initializing Grok client: {e}", "")

        return _chat_completion(
            "IA Inspector Grok", "grok", client, model, system_prompt, user_instructions,
            image, max_frames, image2, resize_image_to,
            max_tokens, temperature, top_p, seed, reasoning_effort,
            attempt_order=["full", "no sampling params", "max_completion_tokens", "minimal"])


# ══════════════════════════════════════════════════════════════════
#  OPENAI NODE
# ══════════════════════════════════════════════════════════════════
class IAInspectorOpenAI:
    CATEGORY  = "IA_Inspector/API"
    RETURN_TYPES  = ("STRING", "STRING")
    RETURN_NAMES  = ("text", "usage")
    FUNCTION  = "generate_text"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "model":         (model_choices("openai"),),
                "max_tokens":    ("INT",   {"default": 4096, "min": 1,   "max": 65536,
                                            "tooltip": "Output budget. Reasoning tokens count against it."}),
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
                # image2: extra image sent AFTER 'image' (e.g. the character reference)
                "image2":            ("IMAGE",),
                "resize_image_to":   (["None", "512", "768", "1024"], {"default": "None"}),
                "reasoning_effort":  (OPENAI_REASONING, {"default": "default"}),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, model):
        return True

    def generate_text(self, system_prompt, model, max_tokens, temperature, top_p, seed,
                      user_instructions="", image=None, max_frames=1, image2=None,
                      resize_image_to="None", reasoning_effort="default"):

        api_key = get_key("openai")
        if not api_key:
            return (missing_key_message("openai"), "")

        try:
            from openai import OpenAI
        except ImportError:
            return ("Error: openai package not installed. Run: pip install openai", "")

        try:
            client = OpenAI(api_key=api_key, timeout=600)
        except Exception as e:
            return (f"Error initializing OpenAI client: {e}", "")

        # Current OpenAI models want max_completion_tokens and often refuse temperature / top_p.
        return _chat_completion(
            "IA Inspector OpenAI", "openai", client, model, system_prompt, user_instructions,
            image, max_frames, image2, resize_image_to,
            max_tokens, temperature, top_p, seed, reasoning_effort,
            attempt_order=["full (max_completion_tokens)", "max_completion_tokens", "full", "minimal"])


# ══════════════════════════════════════════════════════════════════
#  CREDIT / BALANCE
# ══════════════════════════════════════════════════════════════════
def _local_line(label, provider, hint):
    """Estimate from usage_log.json (providers without a balance API)."""
    amount, _ = _credit_setting(provider)
    since, calls, prompt, completion, cost, unpriced = _local_usage_summary(provider)
    if calls == 0:
        return f"{label}: no call logged by this node yet. {hint}"
    line = f"{label}: since {since}, {calls} calls, {prompt:,} in / {completion:,} out tokens"
    if cost is not None:
        line += f", est. ${cost:.4f} spent" if cost < 1 else f", est. ${cost:.2f} spent"
        if amount > 0:
            line += f", est. ${amount - cost:.2f} left of ${amount:.2f}"
        if unpriced:
            line += f" (no price set for: {', '.join(unpriced)})"
    else:
        line += " (add \"prices\" in api_keys.json for a cost estimate)"
    return f"{line}. {hint}"


def grok_credit_line():
    import urllib.error
    key = get_key("grok_management")
    team = get_key("grok_team")
    if not key or not team:
        return "xAI balance: set grok_management_key and grok_team_id in api_keys.json " \
               "(or GROK_MANAGEMENT_KEY and GROK_TEAM_ID on the pod)."
    url = f"https://management-api.x.ai/v1/billing/teams/{team}/prepaid/balance"
    try:
        data = _http_json(url, {"Authorization": f"Bearer {key}"})
        cents = int(float(data.get("total", {}).get("val", 0)))
        # xAI reports prepaid credit as a negative number of USD cents
        return f"xAI prepaid credit: ${abs(cents) / 100:.2f}"
    except urllib.error.HTTPError as e:
        return f"xAI balance error {e.code}: {e.read().decode('utf-8', 'ignore')[:300]}"
    except Exception as e:
        return f"xAI balance error: {e}"


def openai_credit_line():
    """OpenAI has no balance endpoint: real spend from the Costs API (admin key), minus the credit you declare."""
    import urllib.error, urllib.parse
    hint = "Real balance: platform.openai.com > Billing."
    admin = get_key("openai_admin")
    if not admin:
        return _local_line("OpenAI (estimate, no openai_admin_key)", "openai", hint)

    amount, since = _credit_setting("openai")
    today = datetime.datetime.now(datetime.timezone.utc).date()
    start = since or today.replace(day=1)
    start_ts = int(datetime.datetime(start.year, start.month, start.day,
                                     tzinfo=datetime.timezone.utc).timestamp())
    spent, page = 0.0, None
    try:
        for _ in range(20):
            params = {"start_time": start_ts, "bucket_width": "1d", "limit": 180}
            if page:
                params["page"] = page
            data = _http_json("https://api.openai.com/v1/organization/costs?" + urllib.parse.urlencode(params),
                              {"Authorization": f"Bearer {admin}"})
            for bucket in data.get("data", []):
                for result in bucket.get("results", []):
                    spent += float((result.get("amount") or {}).get("value", 0) or 0)
            page = data.get("next_page")
            if not data.get("has_more") or not page:
                break
    except urllib.error.HTTPError as e:
        return f"OpenAI costs error {e.code}: {e.read().decode('utf-8', 'ignore')[:300]}"
    except Exception as e:
        return f"OpenAI costs error: {e}"

    line = f"OpenAI spent since {start.isoformat()}: ${spent:.2f}"
    if amount > 0:
        line += f" | credit left: ${amount - spent:.2f} of ${amount:.2f}"
    return line


def gemini_credit_line():
    return _local_line("Gemini (estimate, Google has no balance API)", "gemini",
                       "Real balance: aistudio.google.com > Billing.")


CREDIT_FUNCS = {"grok": grok_credit_line, "openai": openai_credit_line, "gemini": gemini_credit_line}


class IAInspectorCredit:
    """Credit / spend for Grok, OpenAI and Gemini in one node."""
    CATEGORY = "IA_Inspector/API"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("credit",)
    FUNCTION = "get_credit"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"provider": (["all", "grok", "openai", "gemini"], {"default": "all"})},
                "optional": {"trigger": ("*",)}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # re-check on every run

    def get_credit(self, provider="all", trigger=None):
        names = list(CREDIT_FUNCS) if provider == "all" else [provider]
        text = "\n".join(CREDIT_FUNCS[n]() for n in names)
        for line in text.split("\n"):
            print(f"[IA Inspector Credit] {line}")
        return {"ui": {"text": [text]}, "result": (text,)}


class IAInspectorGrokBalance:
    """Prepaid credit balance of the xAI team (kept for existing workflows, see IAInspectorCredit)."""
    CATEGORY = "IA_Inspector/API"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("balance",)
    FUNCTION = "get_balance"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {"trigger": ("*",)}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # re-check on every run

    def get_balance(self, trigger=None):
        text = grok_credit_line()
        print(f"[IA Inspector Grok] {text}")
        return {"ui": {"text": [text]}, "result": (text,)}


# ══════════════════════════════════════════════════════════════════
#  MAPPINGS
# ══════════════════════════════════════════════════════════════════
NODE_CLASS_MAPPINGS = {
    "IAInspectorGemini": IAInspectorGemini,
    "IAInspectorGrok":   IAInspectorGrok,
    "IAInspectorOpenAI": IAInspectorOpenAI,
    "IAInspectorCredit": IAInspectorCredit,
    "IAInspectorGrokBalance": IAInspectorGrokBalance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAInspectorGemini": "Gemini API (IA Inspector)",
    "IAInspectorGrok":   "Grok API (IA Inspector)",
    "IAInspectorOpenAI": "OpenAI API (IA Inspector)",
    "IAInspectorCredit": "API Credit (IA Inspector)",
    "IAInspectorGrokBalance": "Grok Balance (IA Inspector)",
}
