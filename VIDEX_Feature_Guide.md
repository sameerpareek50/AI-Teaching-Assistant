
# VIDEX AI Teaching Assistant — Complete Feature Guide

> Prepared for presentation reference. Covers every feature, how it is implemented, and what it does.

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Data Pipeline (Offline Scripts)](#2-data-pipeline-offline-scripts)
3. [RAG Chat Interface (Main Page)](#3-rag-chat-interface-main-page)
4. [Video Library](#4-video-library)
5. [Process New Video](#5-process-new-video)
6. [Adaptive Quiz Generator](#6-adaptive-quiz-generator)
7. [Smart Notes Generator](#7-smart-notes-generator)
8. [Video Comparison](#8-video-comparison)
9. [Knowledge Graph](#9-knowledge-graph)
10. [Learner Progress Dashboard](#10-learner-progress-dashboard)
11. [Multi-Provider AI System](#11-multi-provider-ai-system)
12. [Learner Profile & Adaptive Learning System](#12-learner-profile--adaptive-learning-system)
13. [Evaluation & Benchmarking Pipeline](#13-evaluation--benchmarking-pipeline)
14. [Subtitle Generation Feature](#14-subtitle-generation-feature)

---

## 1. System Architecture Overview

VIDEX is a **RAG (Retrieval-Augmented Generation)** based AI Teaching Assistant built with Streamlit. The system works in two phases:

### Offline Phase (Data Preparation)
```
Video Files → FFmpeg (audio) → Whisper (transcription) → JSON segments → ChromaDB vector database
```

### Online Phase (User Interaction)
```
User Question → Vector Embedding → ChromaDB Semantic Search → Top-K chunks → LLM Prompt → Answer
```

### Tech Stack
| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit (Python) |
| Vector Database | ChromaDB (persistent, local) |
| Embedding Model | all-MiniLM-L6-v2 (ChromaDB default, runs locally) |
| Primary LLM | Groq — Llama 3.3 70B (free, 100K tokens/day) |
| Fallback LLM 1 | Gemini 2.5 Flash / 2.0 Flash / 2.0 Flash-Lite |
| Fallback LLM 2 | Ollama — Gemma 3 4B (fully local) |
| Audio Transcription | Groq Whisper → Gemini Audio → Local Whisper |
| Vision Analysis | Groq Vision (Llama 4 Scout) → Gemini → Ollama |
| Audio Extraction | FFmpeg via yt-dlp |

---

## 2. Data Pipeline (Offline Scripts)

These are one-time setup scripts run before the app to index video content.

### Script 1: `1_video_to_mp3.py` — Video to Audio
**What it does:** Converts MP4 video files in the `videos/` folder into MP3 audio files saved to `audios/`.

**How it works:**
- Lists all files in the `videos/` directory
- Parses the filename to extract tutorial number and title (format: `Title_#Number.mp4`)
- Calls `ffmpeg` via subprocess to extract the audio stream as MP3
- Output filename: `{number}_{title}.mp3`

**Why this step:** Whisper transcription only needs audio, not video — extracting audio first reduces file size and processing time.

---

### Script 2: `2_mp3_to_json.py` — Audio to Transcript JSON
**What it does:** Transcribes each MP3 file using OpenAI Whisper `large-v2` and saves transcript segments as JSON.

**How it works:**
- Loads Whisper `large-v2` model (runs locally, no API needed)
- For each audio file, calls `model.transcribe()` with `task="translate"` (auto-translates Hindi/other languages to English)
- Whisper returns segments — each segment has: `start` time, `end` time, `text`
- Adds `number` (video number) and `title` metadata to each segment
- Saves two things per JSON: `chunks` (individual segments with metadata) and `text` (full transcript)

**Output format per JSON:**
```json
{
  "chunks": [
    {"number": "14", "title": "CSS", "start": 12.4, "end": 18.7, "text": "CSS stands for Cascading Style Sheets"}
  ],
  "text": "Full transcript text here..."
}
```

---

### Script 3: `3_preprocess_json.py` — Text to Embeddings (Old Method)
**What it does:** (Legacy script — replaced by ChromaDB in Script 5) Creates vector embeddings for each chunk using Ollama `bge-m3` model and saves as a `.joblib` file.

**How it works:**
- Reads all JSON files from `jsons/` directory
- Sends all chunk texts to local Ollama's `bge-m3` embedding model via REST API
- Assigns a unique `chunk_id` to each chunk
- Creates a pandas DataFrame: columns = number, title, start, end, text, embedding
- Saves as `embeddings.joblib` for fast loading

---

### Script 4: `4_process_incoming.py` — CLI Query Tool (Old Method)
**What it does:** (Legacy prototype) Command-line tool for querying the joblib embeddings using cosine similarity.

**How it works:**
- Loads `embeddings.joblib` into memory
- Takes user query from `input()`
- Creates embedding for the query using Ollama `bge-m3`
- Computes cosine similarity between query embedding and all chunk embeddings (using sklearn)
- Picks top-5 most similar chunks
- Builds a prompt with those chunks and calls local Ollama `llama3.2` for the answer
- Saves prompt and response to `.txt` files

---

### Script 5: `5_migrate_to_chromadb.py` — Migrate to ChromaDB
**What it does:** Reads all JSON files, merges small Whisper segments into larger semantic chunks, and stores them in ChromaDB vector database.

**How it works:**

**Chunk Merging Strategy:**
- Whisper produces very short segments (2–5 seconds each)
- Script merges every 5 consecutive segments into one chunk (`MERGE_SIZE = 5`)
- Merged chunk has: start time of first segment, end time of last, concatenated text
- This gives better semantic context for embedding (a single meaningful paragraph vs. a 3-word fragment)
- Result: roughly 5× compression (e.g., 500 raw segments → 100 semantic chunks)

**ChromaDB Setup:**
- Creates a persistent ChromaDB client at `./chroma_db`
- Creates collection `"video_chunks"` with the **default embedding function** (`all-MiniLM-L6-v2` — runs locally, no API key needed)
- ChromaDB auto-embeds documents when you call `collection.add()`

**Metadata stored per chunk:**
```python
{
  "number": "14",        # Video number
  "title": "CSS",        # Video title
  "start": 120.0,        # Start time in seconds
  "end": 145.3,          # End time in seconds
  "chunk_id": 42         # Unique chunk ID
}
```

---

## 3. RAG Chat Interface (Main Page)

**File:** `app.py`  
**What it does:** The primary user-facing feature. A chat interface where students ask questions about their course videos and get AI-generated answers with video references and timestamps.

### How RAG Works Here

1. **User asks a question** via `st.chat_input()`
2. **Vector Search:** The question is sent to ChromaDB's `collection.query()` which embeds it using `all-MiniLM-L6-v2` and finds the top-K most semantically similar chunks
3. **Prompt Building:** A structured prompt is assembled containing:
   - The retrieved chunk texts with metadata (video title, number, timestamps)
   - Last 3 exchanges of conversation history (for follow-up support)
   - Learner profile context (weak areas from quiz history)
4. **LLM Generation:** Prompt is sent to `gemini_generate()` → Gemini API returns the answer
5. **Answer Display:** Rendered as markdown with source chunk expander and performance metrics

### Key Features in Chat Page

**Conversation Memory:**
- Last 6 messages (3 exchanges) are injected into each prompt
- Allows follow-up questions like "Can you explain that more?"

**Adaptive Responses (Learner-Aware):**
- If the student has taken quizzes, their weak topics and recently missed questions are injected into the prompt
- LLM is told to "focus more on concepts the learner struggles with"

**Source Chunk Display:**
- Each answer shows expandable "View Source Chunks"
- Shows: Video number, title, text excerpt, timestamp badge, relevance % (calculated as `1 - cosine_distance`)

**Performance Bar:**
- Every response shows: Search time (ms), LLM time (seconds), number of chunks retrieved

**Feedback Buttons:**
- Thumbs up / Thumbs down per response
- Stored in session state

**Chat Export:**
- Sidebar download button exports full conversation as `.txt`

**Configurable Retrieval:**
- Sidebar slider lets user adjust `top_k` (1–10 chunks to retrieve)
- Checkbox to show/hide source chunks

---

## 4. Video Library

**File:** `pages/1_📚_Video_Library.py`  
**What it does:** Browse, search, and manage all videos indexed in the knowledge base.

### How it works

**Dual-Source Video Discovery:**
- Source 1: Reads local `jsons/` directory for pre-processed videos (from the offline pipeline)
- Source 2: Queries ChromaDB for videos added via YouTube or file upload (identified by `source` metadata = `"youtube"` or `"upload"`)
- Deduplicates by title

**Each video card shows:**
- Video number, title, first-chunk text preview
- Source badge (Local / YouTube / Uploaded) in different colors
- Number of transcript segments
- Duration in minutes

**Search:**
- Real-time text filter by video title using `st.text_input()`

**Video Deletion:**
- Two-step confirmation to prevent accidental deletion
- Deletes from ChromaDB by matching chunk IDs or title
- Also deletes the corresponding JSON file for local videos
- Reports exactly what was deleted

---

## 5. Process New Video

**File:** `pages/2_🎬_Process_Video.py`  
**What it does:** Add any new video to the knowledge base — either from YouTube URL or by uploading a file.

### YouTube URL Processing

**6-Method Transcript Fallback Chain:**
The system tries each method in order until one succeeds:

| Priority | Method | How |
|----------|--------|-----|
| 1 | YouTube English Captions | `youtube-transcript-api` — instant, no download |
| 2 | YouTube Captions (Translated) | Auto-translate any language captions to English |
| 3 | Groq Whisper (Cloud) | Downloads audio → Groq API (free, 8hrs/day, ≤25MB) |
| 4 | Gemini Audio | Uploads audio file to Gemini Files API for transcription |
| 5 | Local Whisper | OpenAI Whisper `base` model running locally (offline) |
| 6 | AI Vision Analysis | Downloads video → extracts frames → Groq/Gemini vision describes each |

**Smart "No Speech" Detection:**
- After Groq Whisper transcription, `is_transcription_useful()` checks:
  - `no_speech_prob` — Whisper's own confidence that there's no speech
  - `avg_logprob` — overall transcription confidence
  - If mostly silence/music → skip remaining audio methods and jump straight to visual analysis

**Processing Steps (shown live in UI):**
1. Extract transcript (with live status updates as fallback methods are tried)
2. Merge segments → semantic chunks (5 segments per chunk)
3. Generate AI summary using Gemini
4. Add to ChromaDB vector database

**Duplicate Detection:** Checks ChromaDB for existing `yt_{video_id}_*` IDs before processing. If already indexed, shows summary without re-processing.

### File Upload Processing

Same 4-method fallback as YouTube (Groq Whisper → Gemini Audio → Local Whisper → Vision), but reads from the uploaded file bytes.

**Subtitle Generation Option:**
- Only shown for video files (mp4, webm, mkv, avi, mov)
- If checked, runs an extra Step 5 that:
  1. Generates an SRT subtitle file from transcript segments
  2. Splits long segments sentence-by-sentence (max 4 seconds per subtitle entry)
  3. Burns subtitles into the video using FFmpeg via `imageio_ffmpeg`
  4. Offers a download button for the subtitled MP4

**Inline Q&A After Processing:**
- After a video is processed, a chat interface appears on the same page
- Questions are scoped to ONLY that specific video (ChromaDB `where={"title": ...}` filter)

---

## 6. Adaptive Quiz Generator

**File:** `pages/3_🧠_Quiz_Generator.py`  
**What it does:** Generates multiple-choice quizzes from video content. Difficulty automatically adjusts based on past quiz performance.

### How Quiz Generation Works

1. **Topic Selection:** User picks a specific video or "All Topics"
2. **Chunk Retrieval:** Fetches relevant transcript chunks from ChromaDB (either all chunks for a video, or top-20 semantically relevant ones for a topic)
3. **Prompt Engineering:** Sends chunk text to LLM with:
   - Requested difficulty (Easy/Medium/Hard)
   - Number of questions (5/10/15)
   - Previous wrong questions (to reinforce weak areas)
4. **LLM Response:** Returns JSON array — each question has: `question`, `options` (4 choices), `correct` (index 0-3), `explanation`, `concept` (topic tag)
5. **Rendering:** Streamlit form with radio buttons, submit button, and instant results

### Adaptive Difficulty System

**Algorithm (`compute_adaptive_difficulty`):**
- After each quiz, looks at the last 3 quiz scores
- Average score ≥ 80% → upgrade to Hard
- Average score ≥ 50% → stay at Medium
- Average score < 50% → downgrade to Easy
- Displayed as a badge and pre-selects the dropdown (user can override)

**Weak Area Reinforcement:**
- If user has quiz history, last 5 wrong questions are injected into the quiz prompt
- LLM generates similar questions on those concepts to force practice
- Wrong concepts shown as "Focus areas" after quiz results

**Profile Recording:**
After every quiz submission, results are saved to `learner_data.json`:
- Topic accuracy (correct/total per topic label)
- Video accuracy (correct/total per video title)
- Wrong questions stored with question text, user answer, correct answer, explanation
- Quiz history entry (timestamp, topic, score, difficulty)

---

## 7. Smart Notes Generator

**File:** `pages/4_📝_Smart_Notes.py`  
**What it does:** Generates structured study notes from any indexed video's transcript using AI.

### How it works

1. User selects a video from a dropdown (loads from both local JSON files and ChromaDB YouTube videos)
2. User picks a **note style:**

| Style | What it generates |
|-------|------------------|
| Comprehensive Study Notes | Full structured notes with headings, bullets, bold key terms, code examples |
| Quick Revision (Bullet Points) | Short, scannable key facts only |
| Key Concepts & Definitions | Glossary format — term: definition pairs |
| Code Examples & Syntax | Code snippets with brief explanations |
| Exam Prep (Q&A Format) | Potential exam questions with concise answers |
| Focus on Weak Areas *(adaptive)* | Only shown if you've taken quizzes — targets your weak topics |

**"Focus on Weak Areas" Mode:**
- Pulls top-5 weak topics from learner profile
- Pulls last 10 wrong quiz questions
- Tells the LLM to structure notes around:
  1. Key Gaps section
  2. Clear explanations of each weak concept with examples
  3. Common mistakes to avoid
  4. A quick 3-5 question self-check at the end

**Download:**
- Generated notes can be downloaded as a `.md` Markdown file

---

## 8. Video Comparison

**File:** `pages/5_🔀_Compare_Videos.py`  
**What it does:** Compares two selected videos side-by-side to find overlapping topics, unique content, and get a recommendation on watch order.

### How it works

1. User selects Video A and Video B from dropdowns
2. First 4000 characters of each video's transcript are sent to the LLM
3. LLM returns structured JSON:
   ```json
   {
     "video_a_summary": "...",
     "video_b_summary": "...",
     "overlapping_topics": ["HTML Basics", "Box Model"],
     "unique_to_a": ["Semantic Tags", "Forms"],
     "unique_to_b": ["Flexbox", "Grid"],
     "recommendation": "Watch Video A first as it covers foundational concepts...",
     "similarity_score": 0.45
   }
   ```
4. Displayed as:
   - Large similarity % score at top
   - Two-column cards for each video with unique topic badges
   - Centered row of overlapping topic badges
   - Recommendation card

---

## 9. Knowledge Graph

**File:** `pages/6_🕸️_Knowledge_Graph.py`  
**What it does:** Interactive visual graph showing how topics connect across all indexed videos. Built with Plotly.

### How it works

**Graph Construction (AI-powered):**
1. Fetches all chunks from ChromaDB and groups them by video title
2. Sends content summaries of all videos to Gemini in one prompt
3. LLM extracts 15–30 key topics and their connections, returning JSON:
   ```json
   {
     "nodes": [{"id": "CSS Flexbox", "group": "layout", "size": 25}],
     "edges": [{"from": "CSS", "to": "CSS Flexbox", "strength": 3}]
   }
   ```
4. Cached for 2 minutes (`@st.cache_data(ttl=120)`) to avoid re-calling the API

**Layout Algorithm:**
- Groups are arranged in a circle (angle = 2π / number of groups)
- Nodes within a group are placed around the group center with random offset
- **Force-directed relaxation** runs for 50 iterations: connected nodes attract, all nodes repel — produces natural-looking graph

**Visualization:**
- Built with Plotly `go.Scatter` (one trace per node, one per edge)
- Node size = topic importance (from LLM)
- Edge thickness = connection strength (1–3)
- Hover shows topic name and category
- Dynamic color palette per group category

---

## 10. Learner Progress Dashboard

**File:** `pages/7_📊_Learner_Progress.py`  
**What it does:** Tracks and visualizes quiz performance over time. Shows strengths, weaknesses, and personalized recommendations.

### Sections

**Overview Stats (4 metric cards):**
- Total quizzes taken
- Overall accuracy %
- Correct / Total answers count
- Current adaptive difficulty level badge

**Performance Trend Chart:**
- Line chart (Streamlit native) of score % per quiz over time
- Only shown if ≥ 2 quizzes taken

**Topic Performance (bar chart):**
- Accuracy per topic (from all quizzes)
- Color-coded: green ≥70%, orange ≥50%, red <50%
- Sorted highest to lowest

**Video Performance (bar chart):**
- Same but broken down by video title

**Personalized Recommendations:**
- Lists weak topics (<70% accuracy) with specific advice: take a focused quiz or generate weak-area notes
- Lists weak videos with advice to rewatch
- If overall accuracy ≥80%: suggests trying Hard difficulty
- If overall accuracy <50%: suggests focusing on one topic at a time

**Recent Quiz History:**
- Last 15 quizzes in reverse order
- Shows: topic, score, difficulty, date

**Recently Missed Questions:**
- Last 10 wrong questions with: question text, your answer, correct answer, explanation

**Reset Button:**
- Two-step confirmation to clear all quiz data from `learner_data.json`

---

## 11. Multi-Provider AI System

**File:** `gemini_helper.py`  
**What it does:** Unified AI layer with automatic fallback across multiple providers so the app never fails due to rate limits.

### Text Generation Priority Chain
```
Groq (Llama 3.3 70B)  →  Gemini (2.5 Flash → 2.0 Flash → 2.0 Flash-Lite)  →  Ollama (Gemma 3 4B, local)
```

- **Groq:** Tried first — fast, 100K tokens/day free. Rotates across multiple API keys if rate-limited.
- **Gemini:** Tried if Groq fails. Tries multiple model versions in order. Rotates across multiple API keys.
- **Ollama:** Final fallback — fully local, no API needed. Requires Ollama running locally with `gemma3:4b` model.

### Audio Transcription Priority Chain
```
Groq Whisper (whisper-large-v3)  →  Gemini Audio (file upload)  →  Local Whisper (base model)
```

### Vision Analysis Priority Chain
```
Groq Vision (Llama 4 Scout)  →  Gemini (native video file)  →  Ollama (Gemma 3 4B vision)
```

### API Key Rotation
```python
# When rate-limited, rotates to next key in the list
GROQ_API_KEYS = [GROQ_API_KEY_1, GROQ_API_KEY_2, GROQ_API_KEY_3]
GEMINI_API_KEYS = [GEMINI_API_KEY_1, GEMINI_API_KEY_2]
```

### File-Based Generation (Gemini only)
`generate_with_file()` — uploads audio/video directly to Gemini Files API for native multimodal processing. Automatically deletes uploaded files after response.

---

## 12. Learner Profile & Adaptive Learning System

**File:** `learner_profile.py`  
**What it does:** Persistent JSON-based learner data store. Tracks all quiz results and adapts difficulty automatically.

### Data Stored in `learner_data.json`
```json
{
  "quiz_history": [{"timestamp": "...", "topic": "CSS", "score": 4, "total": 5, "difficulty": "Medium"}],
  "topic_scores": {"CSS": {"correct": 12, "total": 15}},
  "video_scores": {"Video 14: CSS": {"correct": 8, "total": 10}},
  "weak_questions": [{"question": "...", "user_answer": "...", "correct_answer": "...", "explanation": "..."}],
  "current_difficulty": "Hard",
  "total_quizzes": 7,
  "total_correct": 42,
  "total_questions": 55
}
```

### Adaptive Difficulty Algorithm
After every quiz, looks at the last 3 quiz scores:
- Average ≥ 80% → **Hard**
- Average ≥ 50% → **Medium**  
- Average < 50% → **Easy**

### Cross-Feature Integration
The learner profile is used across all features:
- **Chat page:** Injects weak topics into RAG prompt so LLM explains those concepts more carefully
- **Quiz page:** Pre-selects adaptive difficulty, injects wrong questions for reinforcement
- **Smart Notes:** Enables "Focus on Weak Areas" note style
- **Progress page:** Full analytics dashboard
- **Sidebar (Chat):** Shows top-3 weak topics as suggested questions

---

## 13. Evaluation & Benchmarking Pipeline

**File:** `6_evaluate.py`  
**What it does:** Measures the quality of the RAG pipeline with 34 labeled test questions.

### Metrics Measured

**Hit Rate (Accuracy):**
- For each test question, checks if the expected video number appears in the top-K retrieved chunks
- Tested at K = 3, 5, 7, 10

**MRR (Mean Reciprocal Rank):**
- Rewards retrieving the correct video at a higher rank
- MRR = average of (1 / rank_of_first_correct_result) across all questions
- MRR of 1.0 = always retrieved at rank 1

**Topic-Wise Breakdown:**
- Breaks accuracy down by topic category: Setup, HTML, CSS, JavaScript

**End-to-End Evaluation:**
- Runs full RAG pipeline (retrieval + LLM) on first 5 questions
- Checks if the LLM response mentions the correct video number
- Reports LLM latency per question

**Test Set Coverage:**
- 34 questions covering: VS Code setup, HTML (headings, images, forms, tables, semantic tags), CSS (selectors, box model, Flexbox, Grid, animations, responsive), JavaScript (variables, arrays, arrow functions, DOM, events, Promises, fetch, localStorage, ES6)

**Results saved to:** `evaluation_results.json`

---

## 14. Subtitle Generation Feature

**Implemented in:** `pages/2_🎬_Process_Video.py`  
**What it does:** Burns transcript text as subtitles directly into a video file and provides a download link.

### How it works

**SRT Generation (`generate_srt()`):**
1. Takes the list of transcript segments (each with `text`, `start`, `end`)
2. Splits long segments sentence-by-sentence using regex (`(?<=[.!?])\s+`)
3. Distributes time evenly across split sentences (max 4 seconds per subtitle entry)
4. Wraps text at 50 characters, max 2 lines per subtitle card
5. Outputs standard SRT format:
   ```
   1
   00:00:12,400 --> 00:00:16,200
   CSS stands for
   Cascading Style Sheets
   ```

**Subtitle Burning (`burn_subtitles()`):**
1. Uses `imageio_ffmpeg` to get the FFmpeg binary (bundled, no system FFmpeg needed)
2. Calls FFmpeg with `subtitles=` filter (uses `libass` for rendering)
3. Font: Arial, Size 14, white text, black outline, semi-transparent background
4. Output: re-encoded MP4 with H.264 video + AAC audio
5. Presented as in-browser video preview + download button

---

## Quick Reference: Feature → File Mapping

| Feature | File |
|---------|------|
| Chat / RAG Q&A | `app.py` |
| Video Library | `pages/1_📚_Video_Library.py` |
| Add YouTube / Upload | `pages/2_🎬_Process_Video.py` |
| Subtitle Generation | `pages/2_🎬_Process_Video.py` |
| Adaptive Quiz | `pages/3_🧠_Quiz_Generator.py` |
| Smart Notes | `pages/4_📝_Smart_Notes.py` |
| Compare Videos | `pages/5_🔀_Compare_Videos.py` |
| Knowledge Graph | `pages/6_🕸️_Knowledge_Graph.py` |
| Learner Progress | `pages/7_📊_Learner_Progress.py` |
| Multi-Provider AI | `gemini_helper.py` |
| Adaptive Profile | `learner_profile.py` |
| RAG Evaluation | `6_evaluate.py` |
| Video → MP3 | `1_video_to_mp3.py` |
| MP3 → JSON | `2_mp3_to_json.py` |
| JSON → ChromaDB | `5_migrate_to_chromadb.py` |

---

## Key Technical Decisions to Highlight in Presentation

1. **ChromaDB over FAISS:** ChromaDB is persistent, has a built-in embedding model, and supports metadata filtering (`where={"title": ...}`) — critical for video-scoped Q&A.

2. **Chunk Merging (5 segments):** Raw Whisper segments are 2–5 seconds (a few words). Merging 5 creates a paragraph-length chunk with enough context for meaningful semantic search.

3. **Multi-Provider Fallback:** Groq's free tier has rate limits. Instead of failing, the system silently falls back to Gemini, then to a fully local Ollama model — zero downtime.

4. **Adaptive Learning Loop:** Quiz results feed back into both the chat responses AND future quiz generation, creating a personalized learning loop without any manual configuration.

5. **Vision Fallback for Silent Videos:** For tutorial videos that are screen recordings without narration, the system detects "no speech" and automatically switches from audio transcription to visual frame analysis.
