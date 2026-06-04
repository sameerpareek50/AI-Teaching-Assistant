# VIDEX — Presentation Slides
> Convert this file to a PowerPoint. Use a dark, modern theme (dark navy/slate background, white text, purple/blue accents). One slide per `---` separator. Keep bullet points as-is — short and punchy.

---

# Slide 1 — Title Slide

## VIDEX
### An AI-Powered Video-Based Intelligent Teaching Assistant

**Using Retrieval-Augmented Generation (RAG)**

---

**Sameer Pareek** | Roll No. 22124087
B.Tech — Information Technology

**Mentor:** Dr. Simranjit Singh, Assistant Professor
**Department of IT** | Dr B R Ambedkar NIT Jalandhar
**June 2026**

---

# Slide 2 — The Problem

## Why Does This Problem Exist?

- Students re-watch entire lectures to find one concept
- No way to **ask questions** directly to a video
- Video content is **unsearchable and non-interactive**
- Learning pace is one-size-fits-all — no personalization
- Transcripts exist but are buried, never used intelligently

> **Gap:** Videos are the #1 learning resource — yet the dumbest format to learn from.

---

# Slide 3 — What is VIDEX?

## An Intelligent Layer Over Video Content

VIDEX transforms any educational video into an **interactive AI tutor** — you can:

- **Ask questions** and get timestamped answers from the video
- **Generate quizzes** to test yourself
- **Get summaries** of any topic in the video
- **Process silent videos** through vision AI frame analysis
- **Generate subtitled videos** with AI-generated captions
- **Track your learning** with adaptive difficulty

> Input: a video. Output: a personal AI tutor.

---

# Slide 4 — System Architecture

## High-Level Architecture

```
[Video Input]
     │
     ▼
[Transcription Layer]
 Groq Whisper → Gemini Audio → Local Whisper
 (hallucination detection via no_speech_prob)
     │
     ▼
[Chunking & Embedding]
 5-segment merging → all-MiniLM-L6-v2 → ChromaDB
     │
     ▼
[RAG Query Engine]
 User Query → Embed → Cosine Similarity → Top-K Chunks
     │
     ▼
[LLM Response]
 Groq Llama-3.3-70B → Gemini 2.5-flash → Ollama (fallback)
     │
     ▼
[Streamlit UI]
 Chat / Quiz / Summary / Learner Profile
```

---

# Slide 5 — Video Processing Pipeline

## How a Video Gets Processed

**Step 1 — Ingest**
- YouTube URL → yt-dlp download
- File upload (mp4, mkv, webm, avi, mov)

**Step 2 — Transcription**
- Primary: Groq Whisper large-v3 (cloud, fast)
- Fallback 1: Gemini audio understanding
- Fallback 2: Local OpenAI Whisper base
- Silent video: Vision AI (Groq Llama-4-Scout, frame analysis)

**Step 3 — Chunking**
- Raw segments → grouped into 5-segment merged chunks
- Preserves timestamps for source traceability

**Step 4 — Embedding & Storage**
- all-MiniLM-L6-v2 → 384-dim vectors → ChromaDB

---

# Slide 6 — RAG: How Q&A Works

## Retrieval-Augmented Generation

**Traditional approach:** Ask LLM → hallucinated answer

**VIDEX approach:**

1. User types a question
2. Question is embedded (same model as video chunks)
3. ChromaDB returns **Top-K most relevant chunks** (cosine similarity)
4. Chunks + question sent to LLM as **grounded context**
5. LLM answers **only from video content** — with timestamps

**Result:** Accurate, source-backed, timestamped answers

> No hallucinations. Every answer points back to the video.

---

# Slide 7 — Multi-LLM Orchestration

## Intelligent Fallback Chain

```
Primary:    Groq API → Llama-3.3-70B
                ↓ (if fails)
Fallback 1: Google Gemini → gemini-2.5-flash
                ↓ (if fails)
Fallback 2: Ollama → gemma3:4b (fully local, offline)
```

**Why this matters:**
- No single point of failure
- Cost optimization (local fallback = free)
- Works offline via Ollama
- Best quality when available (Llama 70B)

---

# Slide 8 — Key Features

## What VIDEX Can Do

| Feature | Description |
|---|---|
| **Video Q&A** | Ask anything, get timestamped answers |
| **Auto Summary** | Topic-wise summary of full video |
| **Quiz Generation** | MCQs with explanations, adaptive difficulty |
| **Multi-source** | YouTube, file upload, pre-processed local videos |
| **Silent Video Support** | Vision AI describes frames when no audio |
| **Subtitle Generation** | AI captions burned into downloadable video |
| **Video Library** | Browse, search, delete indexed videos |
| **Learner Profile** | Tracks performance, adjusts difficulty |

---

# Slide 9 — Adaptive Learning System

## Personalized Difficulty Adjustment

**Tracks last 3 quiz scores → auto-adjusts difficulty**

```
Score ≥ 80%  →  Promote to HARD
Score 50–80% →  Stay at MEDIUM
Score < 50%  →  Drop to EASY
```

**Learner Profile stores:**
- Total questions attempted
- Average score per difficulty level
- Progress trend over time
- Recommended next topic

> Every student gets a different experience based on their performance.

---

# Slide 10 — Subtitle Generation Feature

## AI-Generated Subtitles for Uploaded Videos

**Full pipeline:**

1. Transcription segments extracted from ASR
2. Long segments split by **sentence boundaries** (max 4s per subtitle)
3. SRT file generated with precise timestamps
4. Subtitles **burned into video** using FFmpeg (imageio_ffmpeg, libass)
5. Preview in browser + **one-click download**

**Result:** Any uploaded video → subtitled video in minutes

> Useful for silent recordings, screencasts, or multi-language content.

---

# Slide 11 — Evaluation Results

## Benchmark Performance

**Test Set:** 34 questions across 4 topic domains

| Metric | top-k=3 | top-k=5 (Optimal) |
|---|---|---|
| **Hit Rate** | 91.18% | **97.06%** |
| **MRR** | 0.843 | **0.858** |
| **Avg Latency** | 71.1ms | **69.5ms** |

**Topic-wise accuracy at top-k=5:**

| Topic | Accuracy |
|---|---|
| Setup & Tools | 100% |
| HTML | 91.67% |
| CSS | 100% |
| JavaScript | 100% |

**Embedding model:** all-MiniLM-L6-v2 | **LLM:** Gemini 2.5-flash

---

# Slide 12 — Tech Stack

## Technologies Used

**AI / ML**
- Groq API — Whisper large-v3, Llama-3.3-70B, Llama-4-Scout (vision)
- Google Gemini 2.5-flash
- Ollama + gemma3:4b (local offline fallback)
- sentence-transformers / all-MiniLM-L6-v2

**Storage & Retrieval**
- ChromaDB (persistent vector database)

**Video Processing**
- yt-dlp, OpenAI Whisper, moviepy, FFmpeg (imageio_ffmpeg with libass)

**Frontend / UI**
- Streamlit (multi-page, dark theme, session state caching)

**Language & Hardware**
- Python 3.11+ | MacBook M4 Air (Apple Silicon)

---

# Slide 13 — Demo Flow

## Live Demo Walkthrough

1. **Video Library** → show pre-indexed CSS/JS lecture videos
2. **Ask a question** → "How does CSS flexbox work?" → timestamped answer
3. **Generate a quiz** → MCQs appear, answer, see score
4. **Adaptive difficulty** → system adjusts based on score
5. **Upload a video** → enable subtitle generation
6. **Download subtitled video**
7. **Learner Profile** → performance graph and trends

> Tip: Keep a YouTube URL ready as backup if demo network is slow.

---

# Slide 14 — Conclusion & Future Work

## What We Built

- VIDEX turns passive videos into **interactive AI tutors**
- RAG ensures **factually grounded**, source-backed responses
- **97.06%** retrieval accuracy with **sub-70ms** latency
- Works across audio, silent, and uploaded videos
- **Fully adaptive** to individual learner performance

## Future Scope

- Multi-language subtitle support (Hindi, regional languages)
- Real-time classroom video streaming
- LMS integration (Moodle, Google Classroom)
- Mobile app wrapper
- Collaborative Q&A (multiple students, shared sessions)

---

# Slide 15 — Thank You

## Thank You

**VIDEX — Making Every Video a Smarter Learning Experience**

---

Sameer Pareek | Roll No. 22124087 | B.Tech IT

**Mentor:** Dr. Simranjit Singh, Assistant Professor

Department of Information Technology
Dr B R Ambedkar National Institute of Technology, Jalandhar

---

*Questions?*
