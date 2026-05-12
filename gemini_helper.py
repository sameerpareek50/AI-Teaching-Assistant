"""
Unified AI helper with multi-provider support and automatic fallback.

Priority order for TEXT generation:
  1. Groq (Llama 3.3 70B) — fast, 100K tokens/day free
  2. Gemini (rotating models + keys) — cloud fallback
  3. Ollama (Gemma 3 4B) — local, no rate limits

Priority order for AUDIO TRANSCRIPTION:
  1. Groq Whisper (whisper-large-v3-turbo) — 8 hrs/day free, Hindi→English
  2. Gemini (native audio) — cloud fallback
  3. Local Whisper (base model) — offline, no limits

Priority order for VISION (video frames / images):
  1. Groq Vision (Llama 4 Scout) — 1000 req/day free
  2. Gemini (native video upload) — cloud fallback
  3. Ollama (Gemma 3 4B vision) — local fallback
"""

import os
import re
import json
import time
import base64
import tempfile
import subprocess

from dotenv import load_dotenv
load_dotenv()

# --- Gemini ---
from google import genai

GEMINI_API_KEYS = [
    k for k in [
        os.environ.get("GEMINI_API_KEY_1"),
        os.environ.get("GEMINI_API_KEY_2"),
    ] if k
]
_current_key_index = 0
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

# --- Groq ---
GROQ_API_KEYS = [
    k for k in [
        os.environ.get("GROQ_API_KEY_1"),
        os.environ.get("GROQ_API_KEY_2"),
        os.environ.get("GROQ_API_KEY_3"),
    ] if k
]
_current_groq_key_index = 0
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_WHISPER_MODEL = "whisper-large-v3"
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# --- Ollama ---
OLLAMA_MODEL = "gemma3:4b"

_gemini_clients = {}
_groq_clients = {}


def _switch_groq_key():
    global _current_groq_key_index
    _current_groq_key_index = (_current_groq_key_index + 1) % len(GROQ_API_KEYS)


def _get_groq_client():
    if not GROQ_API_KEYS:
        raise Exception("No Groq API keys found. Add GROQ_API_KEY_1 to .env file.")
    from groq import Groq
    key = GROQ_API_KEYS[_current_groq_key_index]
    if key not in _groq_clients:
        _groq_clients[key] = Groq(api_key=key)
    return _groq_clients[key]


def _switch_gemini_key():
    global _current_key_index
    _current_key_index = (_current_key_index + 1) % len(GEMINI_API_KEYS)


def get_gemini_client():
    if not GEMINI_API_KEYS:
        raise Exception("No Gemini API keys found. Add GEMINI_API_KEY_1 to .env file.")
    key = GEMINI_API_KEYS[_current_key_index]
    if key not in _gemini_clients:
        _gemini_clients[key] = genai.Client(api_key=key)
    return _gemini_clients[key]


def get_client():
    return get_gemini_client()


def _is_rate_limit(error_str):
    return any(k in error_str for k in ["429", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "rate", "quota"])


# ============================================================
# Groq — Text
# ============================================================

def _groq_generate(prompt, status_callback=None):
    last_error = None
    for _ in range(len(GROQ_API_KEYS)):
        try:
            client = _get_groq_client()
            if status_callback:
                status_callback(f"Using Groq ({GROQ_MODEL})...")
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=4096,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            if _is_rate_limit(str(e)):
                _switch_groq_key()
            else:
                raise
    raise Exception(f"GROQ_RATE_LIMITED: All Groq keys at quota. {last_error}")


# ============================================================
# Groq — Whisper Audio Transcription / Translation
# ============================================================

def groq_transcribe(audio_path, translate=False, status_callback=None):
    """Transcribe or translate audio using Groq Whisper API with key rotation.
    Returns list of segments: [{"text": ..., "start": ..., "end": ...}]
    Max file size: 25MB. For Hindi→English, set translate=True."""
    last_error = None
    audio_data = open(audio_path, "rb").read()
    filename = os.path.basename(audio_path)

    for _ in range(len(GROQ_API_KEYS)):
        try:
            client = _get_groq_client()
            if status_callback:
                status_callback(f"Using Groq Whisper ({GROQ_WHISPER_MODEL})...")

            if translate:
                response = client.audio.translations.create(
                    file=(filename, audio_data),
                    model=GROQ_WHISPER_MODEL,
                    response_format="verbose_json",
                    temperature=0.0,
                )
            else:
                response = client.audio.transcriptions.create(
                    file=(filename, audio_data),
                    model=GROQ_WHISPER_MODEL,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                    temperature=0.0,
                )

            segments = []
            if hasattr(response, "segments") and response.segments:
                for seg in response.segments:
                    s = seg if isinstance(seg, dict) else seg.__dict__ if hasattr(seg, '__dict__') else {}
                    segments.append({
                        "text": s.get("text", ""),
                        "start": s.get("start", 0),
                        "end": s.get("end", 0),
                        "no_speech_prob": s.get("no_speech_prob", 0),
                        "avg_logprob": s.get("avg_logprob", 0),
                    })
            elif hasattr(response, "text") and response.text:
                segments.append({"text": response.text, "start": 0, "end": 0, "no_speech_prob": 1.0, "avg_logprob": -1.0})

            return segments
        except Exception as e:
            last_error = e
            if _is_rate_limit(str(e)):
                _switch_groq_key()
            else:
                raise

    raise Exception(f"GROQ_RATE_LIMITED: All Groq keys at quota. {last_error}")


def is_transcription_useful(segments):
    """Check if transcription actually contains meaningful speech content.
    Uses Whisper's no_speech_prob and avg_logprob to detect hallucination.
    Returns False if it's mostly silence, music, or hallucinated text."""
    if not segments:
        return False

    total_text = " ".join(s["text"].strip() for s in segments)

    if len(total_text.strip()) < 20:
        return False

    # Whisper-specific hallucination detection (from Groq verbose_json)
    segments_with_prob = [s for s in segments if "no_speech_prob" in s]
    if segments_with_prob:
        avg_no_speech = sum(s["no_speech_prob"] for s in segments_with_prob) / len(segments_with_prob)
        avg_logprob = sum(s["avg_logprob"] for s in segments_with_prob) / len(segments_with_prob)

        # High no_speech_prob = Whisper thinks there's no speech
        if avg_no_speech > 0.5:
            return False

        # Very low logprob = Whisper is not confident in its transcription (hallucinating)
        if avg_logprob < -0.7:
            return False

        # Check if majority of segments are no-speech
        no_speech_count = sum(1 for s in segments_with_prob if s["no_speech_prob"] > 0.6)
        if no_speech_count > len(segments_with_prob) * 0.5:
            return False

    # Text-based fallback detection (for non-Whisper transcriptions)
    noise_patterns = [
        r"^[\s\.\,\!\?\-\*\[\]\(\)]*$",
        r"(?i)^(music|applause|laughter|silence|noise|♪|🎵|♫|\[.*\]|\(.*\))\s*$",
    ]
    meaningful = 0
    for seg in segments:
        text = seg["text"].strip()
        is_noise = any(re.match(p, text) for p in noise_patterns)
        if text and not is_noise and len(text) > 5:
            meaningful += 1

    return meaningful >= max(2, len(segments) * 0.3)


# ============================================================
# Groq — Vision
# ============================================================

def groq_vision(image_paths, prompt, status_callback=None):
    """Describe images using Groq Vision (Llama 4 Scout) with key rotation.
    Accepts up to 5 images per request. Images must be < 4MB base64."""
    content = [{"type": "text", "text": prompt}]
    for img_path in image_paths[:5]:
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(img_path)[1].lower().strip(".")
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"}
        })

    last_error = None
    for _ in range(len(GROQ_API_KEYS)):
        try:
            client = _get_groq_client()
            if status_callback:
                status_callback(f"Using Groq Vision ({GROQ_VISION_MODEL})...")
            response = client.chat.completions.create(
                model=GROQ_VISION_MODEL,
                messages=[{"role": "user", "content": content}],
                max_completion_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            if _is_rate_limit(str(e)):
                _switch_groq_key()
            else:
                raise
    raise Exception(f"GROQ_RATE_LIMITED: All Groq keys at quota. {last_error}")


# ============================================================
# Ollama
# ============================================================

def _ollama_available():
    try:
        import ollama as ollama_lib
        ollama_lib.list()
        return True
    except Exception:
        return False


def _ollama_has_model():
    try:
        import ollama as ollama_lib
        models = ollama_lib.list()
        for m in models.models:
            if OLLAMA_MODEL in m.model:
                return True
        return False
    except Exception:
        return False


def _ollama_generate(prompt, status_callback=None):
    import ollama as ollama_lib
    if not _ollama_has_model():
        raise Exception(f"Ollama model {OLLAMA_MODEL} not available")
    if status_callback:
        status_callback(f"Using Ollama ({OLLAMA_MODEL}) locally...")
    response = ollama_lib.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.message.content


def _ollama_vision(image_paths, prompt, status_callback=None):
    import ollama as ollama_lib
    if not _ollama_has_model():
        raise Exception(f"Ollama model {OLLAMA_MODEL} not available")
    if status_callback:
        status_callback(f"Using Ollama ({OLLAMA_MODEL}) for visual analysis...")
    response = ollama_lib.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt, "images": image_paths}],
    )
    return response.message.content


# ============================================================
# Gemini
# ============================================================

def _gemini_generate(prompt, status_callback=None):
    last_error = None
    for key_attempt in range(len(GEMINI_API_KEYS)):
        client = get_gemini_client()
        for model in GEMINI_MODELS:
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                return response.text
            except Exception as e:
                last_error = e
                if not _is_rate_limit(str(e)):
                    raise
        if status_callback:
            status_callback(f"Key {_current_key_index + 1} exhausted, switching to next key...")
        _switch_gemini_key()
    raise Exception(f"GEMINI_RATE_LIMITED: All keys and models at quota. {last_error}")


# ============================================================
# Unified API — Text
# ============================================================

def generate(prompt, status_callback=None):
    """Generate text. Priority: Groq → Gemini → Ollama"""
    errors = []

    try:
        return _groq_generate(prompt, status_callback)
    except Exception as e:
        errors.append(f"Groq: {e}")

    try:
        return _gemini_generate(prompt, status_callback)
    except Exception as e:
        errors.append(f"Gemini: {e}")

    if _ollama_available():
        try:
            return _ollama_generate(prompt, status_callback=status_callback)
        except Exception as e:
            errors.append(f"Ollama: {e}")

    raise Exception(f"All providers failed:\n" + "\n".join(errors))


# ============================================================
# Unified API — File-based (Gemini only — native file upload)
# ============================================================

def generate_with_file(file_path, prompt_text, status_callback=None):
    """Generate content from a file (audio/video) using Gemini."""
    last_error = None
    uploaded_files = []

    try:
        for key_attempt in range(len(GEMINI_API_KEYS)):
            client = get_gemini_client()
            uploaded_file = client.files.upload(file=file_path)
            uploaded_files.append((client, uploaded_file))

            for model in GEMINI_MODELS:
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=[uploaded_file, prompt_text]
                    )
                    return response.text
                except Exception as e:
                    last_error = e
                    if not _is_rate_limit(str(e)):
                        raise

            if status_callback:
                status_callback(f"Key {_current_key_index + 1} exhausted, switching to next key...")
            _switch_gemini_key()

        raise Exception(f"GEMINI_RATE_LIMITED: All keys and models at quota. {last_error}")
    finally:
        for cli, uf in uploaded_files:
            try:
                cli.files.delete(name=uf.name)
            except:
                pass


# ============================================================
# Unified API — Vision (frame analysis)
# ============================================================

def _get_video_duration(video_path):
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ], capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except Exception:
        return None


def _get_frame_interval(duration_seconds):
    """Calculate frame extraction interval based on video duration.
    Tighter intervals for shorter videos to capture more detail."""
    if duration_seconds is None:
        return 5
    if duration_seconds < 120:
        return 2
    elif duration_seconds < 300:
        return 3
    elif duration_seconds < 900:
        return 5
    elif duration_seconds < 1800:
        return 8
    else:
        return 10


def describe_video_frames(video_path, prompt_text, status_callback=None):
    """Analyze video frames. Priority: Groq Vision → Gemini → Ollama."""
    duration = _get_video_duration(video_path)
    interval = _get_frame_interval(duration)

    if status_callback:
        dur_str = f"{int(duration)}s" if duration else "unknown duration"
        expected_frames = int(duration / interval) if duration else "?"
        status_callback(f"Extracting 1 frame every {interval}s ({expected_frames} frames for {dur_str} video)...")

    frames_dir = tempfile.mkdtemp()
    try:
        subprocess.run([
            "ffmpeg", "-i", video_path,
            "-vf", f"fps=1/{interval}",
            "-q:v", "5",
            os.path.join(frames_dir, "frame_%04d.jpg")
        ], capture_output=True, timeout=300)

        frame_files = sorted([
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir) if f.endswith(".jpg")
        ])

        if not frame_files:
            raise Exception("No frames extracted")

        # Try Groq Vision first (5 images per request)
        try:
            if status_callback:
                status_callback(f"Analyzing {len(frame_files)} frames with Groq Vision...")
            descriptions = []
            batch_size = 5
            for i in range(0, len(frame_files), batch_size):
                batch = frame_files[i:i + batch_size]
                start_sec = i * interval
                end_sec = min((i + batch_size) * interval, len(frame_files) * interval)
                batch_prompt = (
                    f"These are frames from a video (timestamps {start_sec}s - {end_sec}s). "
                    f"Describe what is shown: content, text on screen, concepts, images, and actions. "
                    f"Be detailed and specific."
                )
                desc = groq_vision(batch, batch_prompt, status_callback=None)
                descriptions.append({"text": desc, "start": start_sec, "end": end_sec})
            return descriptions
        except Exception as e:
            if status_callback:
                status_callback(f"Groq Vision failed ({e}), trying Gemini...")

        # Try Gemini native video
        try:
            if status_callback:
                status_callback("Analyzing video with Gemini...")
            return generate_with_file(video_path, prompt_text, status_callback)
        except Exception as e:
            if status_callback:
                status_callback(f"Gemini failed ({e}), trying local vision...")

        # Ollama fallback
        if _ollama_available() and _ollama_has_model():
            if status_callback:
                status_callback(f"Analyzing {len(frame_files)} frames with local vision model...")
            descriptions = []
            batch_size = 4
            for i in range(0, len(frame_files), batch_size):
                batch = frame_files[i:i + batch_size]
                start_sec = i * interval
                end_sec = min((i + batch_size) * interval, len(frame_files) * interval)
                batch_prompt = (
                    f"These are frames from a video (timestamps {start_sec}s - {end_sec}s). "
                    f"Describe what is shown: content, text on screen, concepts, and actions. "
                    f"Be detailed and specific."
                )
                desc = _ollama_vision(batch, batch_prompt, status_callback=None)
                descriptions.append({"text": desc, "start": start_sec, "end": end_sec})

            if descriptions:
                return descriptions

        raise Exception("All vision providers failed")
    finally:
        for f_path in os.listdir(frames_dir):
            try:
                os.unlink(os.path.join(frames_dir, f_path))
            except:
                pass
        try:
            os.rmdir(frames_dir)
        except:
            pass


# Keep MODELS exported for backward compatibility
MODELS = GEMINI_MODELS
