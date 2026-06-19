import streamlit as st
import chromadb
import json
import os
import re
import sys
import subprocess
import time
import tempfile
from youtube_transcript_api import YouTubeTranscriptApi

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from gemini_helper import generate, generate_with_file, describe_video_frames, groq_transcribe, is_transcription_useful
from theme import get_theme, get_common_css
CHROMA_PATH = "./chroma_db"
MERGE_SIZE = 5
# Use the same Python that's running this script (ensures venv python is used)
PYTHON = sys.executable

st.set_page_config(page_title="Process Video | Videx", page_icon="🎬", layout="wide")

t = get_theme()

st.markdown(get_common_css(t) + f"""
<style>
    .page-header {{
        animation: fadeIn 0.8s ease;
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(-15px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .process-card {{
        background: {t['bg_card']};
        border: 1px solid {t['border_card']};
        border-radius: 18px;
        padding: 2rem;
        margin: 1rem 0;
        animation: fadeIn 0.6s ease;
    }}

    .step-indicator {{
        display: flex;
        gap: 0.5rem;
        align-items: center;
        margin-bottom: 1rem;
    }}

    .step-dot {{
        width: 32px; height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: 700;
        color: white;
        flex-shrink: 0;
    }}

    .step-dot-active {{
        background: linear-gradient(135deg, {t['accent']}, {t['accent2']});
        box-shadow: 0 0 15px {t['code_bg']};
    }}

    .step-dot-done {{
        background: {t['success']};
    }}

    .step-dot-pending {{
        background: {t['bg_card']};
        color: {t['text_muted']};
    }}

    .step-line {{
        height: 2px;
        flex: 1;
        background: {t['border_card']};
    }}

    .step-line-done {{
        background: linear-gradient(90deg, {t['success']}, {t['accent']});
    }}

    .summary-box {{
        background: {t['card_gradient']};
        border: 1px solid {t['border_card']};
        border-radius: 14px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: {t['text_heading']};
        line-height: 1.7;
    }}

    .stat-pill {{
        display: inline-block;
        background: {t['code_bg']};
        border: 1px solid {t['border_card']};
        border-radius: 25px;
        padding: 4px 14px;
        font-size: 0.8rem;
        color: {t['text_heading']};
        margin: 0.3rem 0.3rem 0.3rem 0;
    }}

    .stat-pill-green {{
        background: {t['success_bg']};
        border-color: {t['success_border']};
        color: {t['success']};
    }}

    .transcript-preview {{
        background: {t['bg_card']};
        border: 1px solid {t['border_card']};
        border-radius: 12px;
        padding: 1rem 1.2rem;
        max-height: 300px;
        overflow-y: auto;
        font-size: 0.85rem;
        color: {t['text_explanation']};
        line-height: 1.8;
    }}

    .success-banner {{
        background: {t['success_bg']};
        border: 1px solid {t['success_border']};
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }}

    .success-banner h3 {{
        color: {t['success']} !important;
        margin: 0 0 0.5rem;
    }}

    .success-banner p {{
        color: {t['text_explanation']} !important;
        margin: 0;
    }}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 1rem 0;">
        <div style="font-size:2rem;">🎓</div>
        <div style="font-size:1.1rem; font-weight:700; color:#ffffff;">VIDEX</div>
        <div style="font-size:0.7rem; color:#7aa8cc; letter-spacing:2px;">PROCESS VIDEO</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="color:#e2eeff; font-size:1rem; font-weight:700; margin-bottom:0.6rem;">How it works</div>
    <div style="font-size:0.85rem; line-height:1.9;">
        <div style="color:#b8d4f0; margin-bottom:0.3rem;">1. Paste any YouTube URL</div>
        <div style="color:#b8d4f0; margin-bottom:0.2rem;">2. Smart transcript extraction:</div>
        <div style="color:#7aa8cc; padding-left:1rem; margin-bottom:0.2rem;">· English captions (instant)</div>
        <div style="color:#7aa8cc; padding-left:1rem; margin-bottom:0.2rem;">· Auto-translate other languages</div>
        <div style="color:#7aa8cc; padding-left:1rem; margin-bottom:0.3rem;">· AI audio transcription (fallback)</div>
        <div style="color:#b8d4f0; margin-bottom:0.3rem;">3. AI generates a summary</div>
        <div style="color:#b8d4f0; margin-bottom:0.3rem;">4. Content is indexed for search</div>
        <div style="color:#b8d4f0;">5. Ask questions in Chat page</div>
    </div>
    """, unsafe_allow_html=True)


def extract_video_id(url):
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r'(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'(?:embed/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_youtube_title(video_id):
    """Fetch the actual title of a YouTube video using yt-dlp."""
    try:
        result = subprocess.run(
            [PYTHON, "-m", "yt_dlp", "--no-check-certificates",
             "--get-title", "--no-download",
             f"https://www.youtube.com/watch?v={video_id}"],
            capture_output=True, text=True, timeout=30
        )
        title = result.stdout.strip()
        if title:
            return title
    except Exception:
        pass
    return None


def _download_audio(video_id, tmpdir, status_callback=None):
    """Download audio from YouTube. Returns audio path."""
    audio_path = os.path.join(tmpdir, "audio.mp3")
    if status_callback:
        status_callback("Downloading audio from YouTube...")
    subprocess.run([
        PYTHON, "-m", "yt_dlp",
        "--no-check-certificates",
        "-x", "--audio-format", "mp3",
        "--audio-quality", "5",
        "-o", audio_path,
        f"https://www.youtube.com/watch?v={video_id}"
    ], capture_output=True, text=True, timeout=120)
    if not os.path.exists(audio_path):
        for f in os.listdir(tmpdir):
            if f.endswith(".mp3"):
                audio_path = os.path.join(tmpdir, f)
                break
    if not os.path.exists(audio_path):
        raise Exception("Audio download failed")
    return audio_path


def _download_video(video_id, tmpdir, status_callback=None):
    """Download video from YouTube (lowest quality for speed). Returns video path."""
    video_path = os.path.join(tmpdir, "video.mp4")
    if status_callback:
        status_callback("Downloading video for visual analysis...")
    subprocess.run([
        PYTHON, "-m", "yt_dlp",
        "--no-check-certificates",
        "-f", "worst[ext=mp4]/worst",
        "-o", video_path,
        f"https://www.youtube.com/watch?v={video_id}"
    ], capture_output=True, text=True, timeout=180)
    if not os.path.exists(video_path):
        for f in os.listdir(tmpdir):
            if f.endswith((".mp4", ".webm", ".mkv")):
                video_path = os.path.join(tmpdir, f)
                break
    if not os.path.exists(video_path):
        raise Exception("Video download failed")
    return video_path


def get_transcript(video_id, status_callback=None):
    """Fetch transcript with smart fallback chain.
    1. YouTube captions (English or translated)
    2. Groq Whisper (fast cloud, 8hrs/day free, Hindi→English)
    3. Gemini audio transcription
    4. Local Whisper (offline fallback)
    If audio transcription produces no useful speech → auto-switch to vision analysis.
    5. Groq Vision (frame analysis, 1000 req/day free)
    6. Gemini Vision
    Returns (segments, method_used)"""
    from gemini_helper import groq_transcribe, is_transcription_useful

    api = YouTubeTranscriptApi()

    # --- Attempt 1: YouTube English captions ---
    try:
        if status_callback:
            status_callback("Trying English transcript...")
        transcript = api.fetch(video_id, languages=['en', 'en-US', 'en-GB'])
        segments = [{"text": s.text, "start": s.start, "end": s.start + s.duration} for s in transcript]
        return segments, "english_transcript"
    except Exception:
        pass

    # --- Attempt 2: YouTube captions in other languages → translate ---
    try:
        if status_callback:
            status_callback("No English transcript. Checking other languages...")
        transcript_list = api.list(video_id)
        for t in transcript_list:
            if t.is_translatable:
                if status_callback:
                    status_callback(f"Found {t.language} transcript. Translating to English...")
                translated = t.translate('en').fetch()
                segments = [{"text": s.text, "start": s.start, "end": s.start + s.duration} for s in translated]
                return segments, f"translated_from_{t.language_code}"
    except Exception:
        pass

    # --- Download audio once, reuse for multiple attempts ---
    errors = {}
    audio_path = None
    tmpdir = tempfile.mkdtemp()

    try:
        if status_callback:
            status_callback("No captions found. Downloading audio...")
        audio_path = _download_audio(video_id, tmpdir, status_callback)
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)

        # --- Attempt 3: Groq Whisper (fast, free, handles Hindi) ---
        no_speech_detected = False
        if file_size_mb <= 25:
            try:
                if status_callback:
                    status_callback("Transcribing with Groq Whisper (cloud)...")
                segments = groq_transcribe(audio_path, translate=True, status_callback=status_callback)
                if is_transcription_useful(segments):
                    return segments, "groq_whisper"
                else:
                    no_speech_detected = True
                    if status_callback:
                        status_callback("Groq Whisper detected no meaningful speech — skipping to visual analysis...")
            except Exception as e:
                errors["Groq Whisper"] = str(e)
        else:
            errors["Groq Whisper"] = f"File too large ({file_size_mb:.0f}MB > 25MB limit)"

        # If Groq Whisper confidently detected no speech, skip other audio methods
        if not no_speech_detected:
            # --- Attempt 4: Gemini audio transcription ---
            try:
                if status_callback:
                    status_callback(f"Trying Gemini audio transcription ({file_size_mb:.1f}MB)...")
                response_text = generate_with_file(
                    audio_path,
                    """Transcribe this audio into English text. Output ONLY a JSON array where each element has:
- "text": the transcribed sentence/phrase
- "start": approximate start time in seconds
- "end": approximate end time in seconds

Group the transcript into segments of roughly 5-10 seconds each.
Output valid JSON only, no markdown or explanation.""",
                    status_callback=status_callback
                )
                response_text = response_text.strip()
                if response_text.startswith("```"):
                    response_text = re.sub(r'^```\w*\n?', '', response_text)
                    response_text = re.sub(r'\n?```$', '', response_text)
                segments = json.loads(response_text)
                if is_transcription_useful(segments):
                    return segments, "gemini_audio_transcription"
                else:
                    if status_callback:
                        status_callback("Gemini found no meaningful speech...")
            except Exception as e:
                errors["Gemini audio"] = str(e)

            # --- Attempt 5: Local Whisper (offline) ---
            try:
                if status_callback:
                    status_callback("Trying local Whisper transcription...")
                import whisper
                model = whisper.load_model("base")
                result = model.transcribe(audio_path, task="translate", word_timestamps=False)
                segments = [{"text": s["text"].strip(), "start": s["start"], "end": s["end"]} for s in result["segments"]]
                if is_transcription_useful(segments):
                    return segments, "whisper_local_transcription"
                else:
                    if status_callback:
                        status_callback("Local Whisper found no meaningful speech...")
            except Exception as e:
                errors["Local Whisper"] = str(e)

    except Exception as e:
        errors["Audio download"] = str(e)

    # --- All audio methods failed or found no speech → visual analysis ---
    if status_callback:
        status_callback("No speech detected. Switching to visual analysis...")

    try:
        video_path = _download_video(video_id, tmpdir, status_callback)
        if status_callback:
            status_callback("Analyzing video frames with AI vision...")
        segments = describe_video_frames(video_path, "Describe what is shown in this video", status_callback)
        if isinstance(segments, str):
            segments = json.loads(segments)
        if isinstance(segments, list) and segments:
            return segments, "vision_analysis"
    except Exception as e:
        errors["Vision analysis"] = str(e)

    # Clean up
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    error_lines = "\n".join(f"- {k}: {v}" for k, v in errors.items())
    raise Exception(f"All methods failed:\n{error_lines}\n\nMake sure the video exists and is accessible.")


def merge_segments(segments, merge_size=MERGE_SIZE):
    """Merge small transcript segments into larger semantic chunks."""
    merged = []
    for i in range(0, len(segments), merge_size):
        group = segments[i:i + merge_size]
        merged.append({
            "start": group[0]["start"],
            "end": group[-1]["end"],
            "text": " ".join(s["text"].replace("\n", " ").strip() for s in group)
        })
    return merged


def generate_summary(full_text, title):
    """Generate AI summary using Gemini."""
    prompt = f"""Summarize the following video transcript in a clear, structured way.
Include:
- A brief overview (2-3 sentences)
- Key topics covered (bullet points)
- Key takeaways

Video title: {title}
Transcript:
{full_text[:8000]}"""

    return generate(prompt)


def generate_srt(segments):
    """Convert transcript segments to SRT subtitle format.
    Long segments are split by sentence so subtitles change throughout the video.
    """
    import re

    MAX_SUB_DURATION = 4.0  # max seconds per subtitle entry

    def fmt_time(seconds):
        seconds = max(0.0, float(seconds))
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int(round((seconds % 1) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def wrap_line(text, max_chars=50, max_lines=2):
        words = text.split()
        lines, current = [], ""
        for word in words:
            if current and len(current) + 1 + len(word) > max_chars:
                lines.append(current)
                if len(lines) >= max_lines:
                    break
                current = word
            else:
                current = (current + " " + word).strip()
        if current and len(lines) < max_lines:
            lines.append(current)
        return "\n".join(lines) if lines else text

    def split_segment(seg):
        """Split a long segment into sentence-sized subtitle entries."""
        text = seg["text"].strip()
        start = float(seg["start"])
        end = float(seg["end"])
        duration = end - start

        # Split into sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text) if s.strip()]
        if not sentences:
            sentences = [text]

        # If short enough already, return as-is
        if duration <= MAX_SUB_DURATION and len(sentences) <= 1:
            return [{"text": text, "start": start, "end": max(end, start + 0.5)}]

        # Distribute duration evenly across sentences
        time_per = duration / len(sentences)
        result = []
        for i, sentence in enumerate(sentences):
            s = start + i * time_per
            e = min(s + time_per, end)
            if e <= s:
                e = s + 0.5
            result.append({"text": sentence, "start": s, "end": e})
        return result

    # Expand all segments into sentence-level entries
    expanded = []
    for seg in segments:
        expanded.extend(split_segment(seg))

    # Render as SRT
    blocks = []
    for i, seg in enumerate(expanded, 1):
        text = wrap_line(seg["text"])
        if text:
            blocks.append(f"{i}\n{fmt_time(seg['start'])} --> {fmt_time(seg['end'])}\n{text}")

    return "\n\n".join(blocks) + "\n"


def burn_subtitles(video_path, srt_path, output_path):
    """Burn SRT subtitles into a video using imageio_ffmpeg (has libass/subtitles filter built in)."""
    import imageio_ffmpeg
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    srt_escaped = srt_path.replace("\\", "/").replace(":", "\\:")
    cmd = [
        ffmpeg_bin, "-y",
        "-i", video_path,
        "-vf", (
            f"subtitles='{srt_escaped}':force_style='"
            "FontName=Arial,FontSize=14,"
            "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
            "BackColour=&H80000000,BorderStyle=3,"
            "Outline=1,Shadow=0,Alignment=2,MarginV=15'"
        ),
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "fast",
        "-c:a", "aac",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise Exception(f"FFmpeg error: {result.stderr[-500:]}")
    return output_path


def check_video_exists(video_id):
    """Check if a YouTube video is already indexed in ChromaDB.
    Returns (exists, title, chunks, full_text) or (False, None, None, None)."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection("video_chunks")
        results = collection.get(
            where={"source": "youtube"},
            include=["documents", "metadatas"]
        )

        matched_docs = []
        matched_title = None
        for i, meta in enumerate(results["metadatas"]):
            chunk_id = results["ids"][i]
            if f"yt_{video_id}_" in chunk_id:
                matched_docs.append({
                    "text": results["documents"][i],
                    "start": meta["start"],
                    "end": meta["end"],
                })
                matched_title = meta["title"]

        if matched_docs:
            matched_docs.sort(key=lambda x: x["start"])
            full_text = " ".join(d["text"] for d in matched_docs)
            return True, matched_title, len(matched_docs), full_text

    except Exception:
        pass

    return False, None, None, None


def add_to_chromadb(video_id, title, merged_chunks, source="youtube"):
    """Add processed video chunks to ChromaDB."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection("video_chunks")

    existing_count = collection.count()
    ids = []
    texts = []
    metadatas = []

    for i, chunk in enumerate(merged_chunks):
        chunk_id = existing_count + i
        ids.append(f"yt_{video_id}_{i}")
        texts.append(chunk["text"])
        metadatas.append({
            "number": video_id[:6],
            "title": title,
            "start": float(chunk["start"]),
            "end": float(chunk["end"]),
            "chunk_id": chunk_id,
            "source": source
        })

    collection.add(ids=ids, documents=texts, metadatas=metadatas)
    return len(ids)


# --- Main Content ---
st.markdown("""
<div class="page-header">
    <div class="page-title">Process New Video</div>
    <div class="page-subtitle">Add any video to the knowledge base — YouTube URL or file upload</div>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "process_state" not in st.session_state:
    st.session_state.process_state = "input"
if "video_data" not in st.session_state:
    st.session_state.video_data = {}

# Two input modes
tab_youtube, tab_upload = st.tabs(["YouTube URL", "Upload Audio/Video File"])

url = None
uploaded_file = None
process_btn = False
generate_subtitles_cb = False  # safe default before tab renders

with tab_youtube:
    url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Works best with videos that have captions/subtitles"
    )
    col1, col2 = st.columns([1, 5])
    with col1:
        yt_btn = st.button("Process YouTube Video", type="primary", use_container_width=True)
    with col2:
        if st.button("Reset", key="reset_yt"):
            st.session_state.process_state = "input"
            st.session_state.video_data = {}
            st.session_state.video_chat = []
            st.rerun()
    if yt_btn and url:
        process_btn = True

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload an audio or video file",
        type=["mp3", "mp4", "wav", "m4a", "webm", "ogg"],
        help="Upload any audio/video file — Gemini AI will transcribe it"
    )
    upload_title = st.text_input("Video/Lecture Title", placeholder="e.g., Machine Learning Lecture 1")

    # Subtitle checkbox — only for video files, not audio-only
    _upload_ext = uploaded_file.name.split('.')[-1].lower() if uploaded_file else ""
    _is_video_upload = _upload_ext in ("mp4", "webm", "mkv", "avi", "mov")
    if uploaded_file and _is_video_upload:
        generate_subtitles_cb = st.checkbox(
            "Generate subtitled video",
            help="Burns the transcript as subtitles into a downloadable copy of your video"
        )

    col3, col4 = st.columns([1, 5])
    with col3:
        upload_btn = st.button("Process Uploaded File", type="primary", use_container_width=True)
    with col4:
        if st.button("Reset", key="reset_upload"):
            st.session_state.process_state = "input"
            st.session_state.video_data = {}
            st.session_state.video_chat = []
            st.rerun()

# --- Process uploaded file ---
if upload_btn and uploaded_file:
    generate_subtitles = generate_subtitles_cb  # capture before rerun
    total_steps = 5 if generate_subtitles else 4

    step1_status = st.empty()
    step1_status.info(f"Step 1/{total_steps}: Processing uploaded file...")

    from gemini_helper import groq_transcribe, is_transcription_useful, generate_with_file, describe_video_frames

    try:
        start_time = time.time()
        file_bytes = uploaded_file.read()  # keep bytes for subtitle burning later
        file_ext = uploaded_file.name.split('.')[-1].lower()
        with tempfile.NamedTemporaryFile(suffix=f".{file_ext}", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        file_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        title = upload_title if upload_title else uploaded_file.name
        video_id = f"upload_{int(time.time())}"
        segments = None
        method_label = None

        # --- Try 1: Groq Whisper (fast, free, ≤25MB) ---
        no_speech_detected = False
        if file_size_mb <= 25:
            try:
                step1_status.info(f"Step 1/{total_steps}: Transcribing with Groq Whisper ({file_size_mb:.1f}MB)...")
                segments = groq_transcribe(tmp_path, translate=True)
                if is_transcription_useful(segments):
                    method_label = "Groq Whisper (cloud)"
                else:
                    step1_status.info(f"Step 1/{total_steps}: No speech detected — skipping to visual analysis...")
                    no_speech_detected = True
                    segments = None
            except Exception:
                segments = None

        # If Groq Whisper detected no speech, skip other audio methods
        if segments is None and not no_speech_detected:
            # --- Try 2: Gemini audio ---
            try:
                step1_status.info(f"Step 1/{total_steps}: Trying Gemini audio transcription ({file_size_mb:.1f}MB)...")
                response_text = generate_with_file(
                    tmp_path,
                    """Transcribe this audio into English text. Output ONLY a JSON array where each element has:
- "text": the transcribed sentence/phrase
- "start": approximate start time in seconds
- "end": approximate end time in seconds

Group the transcript into segments of roughly 5-10 seconds each.
Output valid JSON only, no markdown or explanation."""
                )
                response_text = response_text.strip()
                if response_text.startswith("```"):
                    response_text = re.sub(r'^```\w*\n?', '', response_text)
                    response_text = re.sub(r'\n?```$', '', response_text)
                segments = json.loads(response_text)
                if is_transcription_useful(segments):
                    method_label = "Gemini AI transcription"
                else:
                    segments = None
            except Exception:
                segments = None

            # --- Try 3: Local Whisper ---
            if segments is None:
                try:
                    step1_status.info(f"Step 1/{total_steps}: Trying local Whisper transcription...")
                    import whisper
                    whisper_model = whisper.load_model("base")
                    whisper_result = whisper_model.transcribe(tmp_path, task="translate", word_timestamps=False)
                    segments = [{"text": s["text"].strip(), "start": s["start"], "end": s["end"]} for s in whisper_result["segments"]]
                    if is_transcription_useful(segments):
                        method_label = "Whisper local transcription"
                    else:
                        segments = None
                except Exception:
                    segments = None

        # --- Try 4: Vision analysis (for videos with no speech) ---
        if segments is None and file_ext in ("mp4", "webm", "mkv", "avi", "mov"):
            try:
                step1_status.info(f"Step 1/{total_steps}: No speech detected. Analyzing video visually...")
                result = describe_video_frames(tmp_path, "Describe what is shown in this video")
                if isinstance(result, str):
                    result = json.loads(result)
                if isinstance(result, list) and result:
                    segments = result
                    method_label = "AI visual analysis (no speech detected)"
            except Exception:
                segments = None

        if segments is None:
            os.unlink(tmp_path)
            step1_status.error(f"Step 1/{total_steps}: All transcription methods failed. Try a different file.")
            st.stop()

        transcript_time = time.time() - start_time
        full_text = " ".join(s["text"].replace("\n", " ") for s in segments)
        os.unlink(tmp_path)
        step1_status.success(f"Step 1/{total_steps}: {len(segments)} segments via {method_label} ({transcript_time:.1f}s)")

    except Exception as e:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        step1_status.error(f"Processing failed: {e}")
        st.stop()

    # Prevent YouTube path from running
    process_btn = False

    # Step 2: Merge
    step2_status = st.empty()
    merged = merge_segments(segments)
    step2_status.success(f"Step 2/{total_steps}: Created {len(merged)} semantic chunks ({len(segments)} → {len(merged)})")

    # Step 3: Summary
    step3_status = st.empty()
    step3_status.info(f"Step 3/{total_steps}: Generating AI summary...")
    try:
        summary_start = time.time()
        summary = generate_summary(full_text, title)
        step3_status.success(f"Step 3/{total_steps}: Summary generated ({time.time()-summary_start:.1f}s)")
    except Exception:
        summary = "Summary generation skipped (rate limit). Try again shortly."
        step3_status.warning(f"Step 3/{total_steps}: Summary skipped (rate limit)")

    # Step 4: Index
    step4_status = st.empty()
    try:
        chunks_added = add_to_chromadb(video_id, title, merged, source="upload")
        step4_status.success(f"Step 4/{total_steps}: Added {chunks_added} chunks to ChromaDB")
    except Exception as e:
        step4_status.error(f"Indexing failed: {e}")
        st.stop()

    # Step 5: Generate subtitled video (only for video files when checkbox is checked)
    subtitled_video_bytes = None
    subtitle_filename = None
    if generate_subtitles and file_ext in ("mp4", "webm", "mkv", "avi", "mov"):
        import shutil
        step5_status = st.empty()
        step5_status.info("Step 5/5: Generating subtitled video (this may take a minute)...")
        sub_tmpdir = tempfile.mkdtemp()
        try:
            orig_video = os.path.join(sub_tmpdir, f"input.{file_ext}")
            with open(orig_video, "wb") as f:
                f.write(file_bytes)

            srt_content = generate_srt(segments)
            srt_path = os.path.join(sub_tmpdir, "subs.srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_content)

            output_path = os.path.join(sub_tmpdir, "subtitled.mp4")
            burn_subtitles(orig_video, srt_path, output_path)

            with open(output_path, "rb") as f:
                subtitled_video_bytes = f.read()

            size_mb = len(subtitled_video_bytes) / (1024 * 1024)
            subtitle_filename = f"{title}_subtitled.mp4"
            step5_status.success(f"Step 5/5: Subtitled video ready ({size_mb:.1f} MB)")
        except Exception as e:
            step5_status.error(f"Step 5/5: Subtitle generation failed — {e}")
        finally:
            shutil.rmtree(sub_tmpdir, ignore_errors=True)

    st.session_state.video_data = {
        "video_id": video_id, "segments": len(segments), "chunks": len(merged),
        "summary": summary, "full_text": full_text, "transcript_time": transcript_time,
        "method": method_label, "title": title,
        "subtitled_video": subtitled_video_bytes,
        "subtitle_filename": subtitle_filename,
    }
    st.session_state.process_state = "done"
    st.rerun()


# --- Process YouTube URL ---
if process_btn and url:
    video_id = extract_video_id(url)

    if not video_id:
        st.error("Invalid YouTube URL. Please check and try again.")
    else:
        # Check if video is already indexed
        exists, existing_title, existing_chunks, existing_text = check_video_exists(video_id)

        if exists:
            st.markdown(f"""
            <div class="success-banner">
                <h3>Video Already Indexed!</h3>
                <p>This video is already in the knowledge base — no need to process again</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="text-align:center; margin: 1rem 0;">
                <span class="stat-pill">{existing_title}</span>
                <span class="stat-pill">{existing_chunks} chunks indexed</span>
                <span class="stat-pill stat-pill-green">Already in ChromaDB</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("### Video Preview")
                st.video(f"https://www.youtube.com/watch?v={video_id}")

            with col2:
                st.markdown("### AI Summary")
                with st.spinner("Generating summary from existing content..."):
                    try:
                        summary = generate_summary(existing_text, existing_title)
                        st.markdown(f"""<div class="summary-box">{summary}</div>""", unsafe_allow_html=True)
                    except Exception:
                        st.info("Summary generation skipped (rate limit). Content is still searchable in Chat.")

            with st.expander("View Full Transcript", expanded=False):
                st.markdown(f"""<div class="transcript-preview">{existing_text}</div>""", unsafe_allow_html=True)

            st.session_state.video_data = {
                "video_id": video_id, "title": existing_title,
                "segments": existing_chunks, "chunks": existing_chunks,
                "summary": "Already indexed", "full_text": existing_text,
                "transcript_time": 0, "method": "Already indexed",
            }
            st.session_state.process_state = "done"
            st.stop()

        # Step 1: Extract transcript (with smart fallback)
        step1_status = st.empty()
        step1_status.info("Step 1/4: Extracting transcript from YouTube...")

        try:
            start_time = time.time()

            def update_status(msg):
                step1_status.info(f"Step 1/4: {msg}")

            segments, method = get_transcript(video_id, status_callback=update_status)
            transcript_time = time.time() - start_time

            full_text = " ".join(s["text"].replace("\n", " ") for s in segments)

            # Fetch actual YouTube video title
            title = get_youtube_title(video_id)
            if not title:
                title = full_text[:80].strip() + "..."

            method_labels = {
                "english_transcript": "English captions",
                "groq_whisper": "Groq Whisper (cloud)",
                "gemini_audio_transcription": "Gemini AI transcription",
                "whisper_local_transcription": "Whisper local transcription",
                "vision_analysis": "AI visual analysis (no speech detected)",
            }
            method_label = method_labels.get(method, f"Translated ({method.replace('translated_from_', '')})")
            step1_status.success(f"Step 1/4: Extracted {len(segments)} segments via {method_label} ({transcript_time:.1f}s)")

        except Exception as e:
            step1_status.error(f"Could not extract transcript from YouTube.\n\n{e}\n\n**Tip:** Try the 'Upload Audio/Video File' tab instead — you can download the video manually and upload it here.")
            st.stop()

        # Step 2: Merge chunks
        step2_status = st.empty()
        step2_status.info("Step 2/4: Creating semantic chunks...")

        merged = merge_segments(segments)
        step2_status.success(f"Step 2/4: Created {len(merged)} semantic chunks ({len(segments)} → {len(merged)})")

        # Step 3: Generate summary
        step3_status = st.empty()
        step3_status.info("Step 3/4: Generating AI summary...")

        try:
            start_time = time.time()
            summary = generate_summary(full_text, title)
            summary_time = time.time() - start_time
            step3_status.success(f"Step 3/4: Summary generated ({summary_time:.1f}s)")
        except Exception as e:
            summary = "Summary generation failed due to rate limits. Try again in a minute."
            step3_status.warning(f"Step 3/4: Summary skipped (rate limit)")

        # Step 4: Add to ChromaDB
        step4_status = st.empty()
        step4_status.info("Step 4/4: Indexing in vector database...")

        try:
            chunks_added = add_to_chromadb(video_id, title, merged)
            step4_status.success(f"Step 4/4: Added {chunks_added} chunks to ChromaDB")
        except Exception as e:
            step4_status.error(f"Step 4/4: Indexing failed: {e}")
            st.stop()

        # Store results
        st.session_state.video_data = {
            "video_id": video_id,
            "title": title,
            "segments": len(segments),
            "chunks": len(merged),
            "summary": summary,
            "full_text": full_text,
            "transcript_time": transcript_time,
            "method": method_label,
        }
        st.session_state.process_state = "done"
        st.rerun()

# Show results
if st.session_state.process_state == "done" and st.session_state.video_data:
    data = st.session_state.video_data

    # Success banner
    st.markdown(f"""
    <div class="success-banner">
        <h3>Video Processed Successfully!</h3>
        <p>The video has been indexed and is now searchable in the Chat page</p>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    st.markdown(f"""
    <div style="text-align:center; margin: 1rem 0;">
        <span class="stat-pill">{data.get('title', data['video_id'])}</span>
        <span class="stat-pill">{data['segments']} segments extracted</span>
        <span class="stat-pill">{data['chunks']} semantic chunks</span>
        <span class="stat-pill">Method: {data.get('method', 'auto')}</span>
        <span class="stat-pill stat-pill-green">Indexed in ChromaDB</span>
    </div>
    """, unsafe_allow_html=True)

    # Video embed / title
    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Video Preview")
        if data['video_id'].startswith("upload_"):
            st.markdown(f"**{data.get('title', 'Uploaded File')}**")
            st.info("Uploaded file — no video preview available")
        else:
            st.video(f"https://www.youtube.com/watch?v={data['video_id']}")

    with col2:
        st.markdown("### AI Summary")
        st.markdown(f"""<div class="summary-box">{data['summary']}</div>""", unsafe_allow_html=True)

    # Subtitled video section (only shown if generated)
    if data.get("subtitled_video"):
        st.markdown("---")
        st.markdown(f"""
        <div style="text-align:center; margin: 0.5rem 0 1rem;">
            <span style="font-size:1.2rem; font-weight:700; color:{t['text_heading']};">Subtitled Video</span>
            <p style="color:{t['text_secondary']}; font-size:0.85rem; margin-top:0.3rem;">
                Your video with transcript subtitles burned in at the bottom
            </p>
        </div>
        """, unsafe_allow_html=True)
        col_vid, col_dl = st.columns([3, 1])
        with col_vid:
            st.video(data["subtitled_video"])
        with col_dl:
            size_mb = len(data["subtitled_video"]) / (1024 * 1024)
            st.markdown(f"""
            <div style="background:{t['bg_card']};border:1px solid {t['border_card']};
                border-radius:14px;padding:1.2rem;text-align:center;margin-bottom:0.8rem;">
                <div style="font-size:1.8rem;">🎬</div>
                <div style="color:{t['text_heading']};font-weight:600;margin:0.4rem 0;">Ready to Download</div>
                <div style="color:{t['text_muted']};font-size:0.8rem;">{size_mb:.1f} MB &nbsp;·&nbsp; MP4</div>
            </div>
            """, unsafe_allow_html=True)
            st.download_button(
                label="Download Subtitled Video",
                data=data["subtitled_video"],
                file_name=data.get("subtitle_filename", "subtitled_video.mp4"),
                mime="video/mp4",
                use_container_width=True,
                type="primary",
            )

    # Transcript preview
    st.markdown("---")
    with st.expander("View Full Transcript", expanded=False):
        st.markdown(f"""<div class="transcript-preview">{data['full_text']}</div>""", unsafe_allow_html=True)

    # --- Inline Q&A Section ---
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center; margin: 1rem 0 0.5rem;">
        <span style="font-size:1.3rem; font-weight:700; color:{t['text_heading']};">Ask Questions About This Video</span>
        <p style="color:{t['text_secondary']}; font-size:0.85rem; margin-top:0.3rem;">Powered by RAG — answers are grounded in the video content</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize video chat history
    if "video_chat" not in st.session_state:
        st.session_state.video_chat = []

    # Display chat history for this video
    for msg in st.session_state.video_chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("View Source Chunks", expanded=False):
                    for src in msg["sources"]:
                        similarity = max(0, 1 - src.get('distance', 0))
                        st.markdown(f"""<div class="source-card" style="background:{t['card_gradient']};border:1px solid {t['border_card']};border-radius:12px;padding:1rem;margin:0.5rem 0;color:{t['text_explanation']};">
                            <b style="color:{t['text_heading']};">{src['title']}</b><br>
                            <span style="font-size:0.85rem;">"{src['text'][:200]}..."</span><br>
                            <span style="font-size:0.78rem;color:{t['text_heading']};">Timestamp: {int(src['start']//60)}m {int(src['start']%60)}s — {int(src['end']//60)}m {int(src['end']%60)}s</span>
                            <span style="font-size:0.78rem;color:{t['success']}; margin-left:0.5rem;">Relevance: {similarity:.0%}</span>
                        </div>""", unsafe_allow_html=True)

    # Chat input
    video_question = st.chat_input("Ask a question about this video...")

    if video_question:
        st.session_state.video_chat.append({"role": "user", "content": video_question})

        with st.chat_message("user"):
            st.markdown(video_question)

        with st.chat_message("assistant"):
            with st.spinner("Searching video content..."):
                try:
                    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
                    collection = chroma_client.get_collection("video_chunks")
                    current_title = data.get("title", "")
                    results = collection.query(
                        query_texts=[video_question],
                        n_results=5,
                        where={"title": current_title},
                        include=["documents", "metadatas", "distances"]
                    )

                    chunks_info = []
                    sources = []
                    for i in range(len(results['ids'][0])):
                        meta = results['metadatas'][0][i]
                        chunks_info.append({
                            "title": meta['title'], "number": meta['number'],
                            "start": meta['start'], "end": meta['end'],
                            "text": results['documents'][0][i]
                        })
                        sources.append({
                            "title": meta['title'], "start": meta['start'],
                            "end": meta['end'], "text": results['documents'][0][i],
                            "distance": results['distances'][0][i]
                        })

                    prompt = f'''You are an AI video assistant. The user is asking about the video: "{current_title}".
Here are relevant transcript chunks from THIS video only, with start time, end time, and text:

{json.dumps(chunks_info)}
---------------------------------
"{video_question}"
Answer based ONLY on the above transcript chunks from this specific video. Include timestamps when referencing specific parts. If the question is unrelated to this video's content, let them know politely.'''

                    answer = generate(prompt)

                    st.markdown(answer)

                    with st.expander("View Source Chunks", expanded=False):
                        for src in sources:
                            similarity = max(0, 1 - src.get('distance', 0))
                            st.markdown(f"""<div style="background:{t['card_gradient']};border:1px solid {t['border_card']};border-radius:12px;padding:1rem;margin:0.5rem 0;color:{t['text_explanation']};">
                                <b style="color:{t['text_heading']};">{src['title']}</b><br>
                                <span style="font-size:0.85rem;">"{src['text'][:200]}..."</span><br>
                                <span style="font-size:0.78rem;color:{t['text_heading']};">Timestamp: {int(src['start']//60)}m {int(src['start']%60)}s — {int(src['end']//60)}m {int(src['end']%60)}s</span>
                                <span style="font-size:0.78rem;color:{t['success']}; margin-left:0.5rem;">Relevance: {similarity:.0%}</span>
                            </div>""", unsafe_allow_html=True)

                    st.session_state.video_chat.append({"role": "assistant", "content": answer, "sources": sources})

                except Exception as e:
                    error_msg = f"Could not generate answer: {e}"
                    st.error(error_msg)
                    st.session_state.video_chat.append({"role": "assistant", "content": error_msg})

        st.rerun()
