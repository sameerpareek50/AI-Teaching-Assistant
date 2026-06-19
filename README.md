# VIDEX — AI Teaching Assistant

An AI-powered teaching assistant that lets you ask questions about course videos, generate quizzes, create smart notes, and track your learning — all through a clean Streamlit interface.

Built on a RAG (Retrieval-Augmented Generation) pipeline: video content is transcribed, chunked, embedded into ChromaDB, and retrieved semantically at query time.

---

## Features

| Page | What it does |
|------|-------------|
| **Home** | Chat with your video library — ask questions, get timestamped answers with source chunks |
| **Video Library** | Browse and delete all indexed videos |
| **Process Video** | Add new videos via YouTube URL or file upload — transcription and indexing handled automatically |
| **Quiz Generator** | Adaptive quizzes that get harder/easier based on your performance |
| **Smart Notes** | Auto-generate structured study notes from any indexed video |
| **Compare Videos** | Find overlapping and unique content between two videos |
| **Knowledge Graph** | Visual map of how topics connect across all your videos |
| **Learner Progress** | Track accuracy trends, weak areas, and full quiz history |

---

## Tech Stack

- **Frontend** — Streamlit
- **Vector DB** — ChromaDB (persistent, local)
- **Embeddings** — `all-MiniLM-L6-v2` (via ChromaDB default)
- **LLM (text)** — Groq (Llama 3.3 70B) → Gemini (2.5 Flash) → Ollama (Gemma 3 4B)
- **Transcription** — Groq Whisper → Gemini Audio → Local Whisper
- **Vision** — Groq Vision (Llama 4 Scout) → Gemini → Ollama Vision
- **YouTube** — `yt-dlp` + `youtube-transcript-api`

The app automatically falls back through providers — you don't need all three set up, but having at least Groq or Gemini is recommended.

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd AI-Teaching-Assistant

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

Create a `.env` file in the project root:

```env
# Groq (recommended — fast and free tier)
GROQ_API_KEY_1=your_groq_api_key
GROQ_API_KEY_2=optional_second_groq_key   # for rate-limit rotation
GROQ_API_KEY_3=optional_third_groq_key

# Gemini (cloud fallback)
GEMINI_API_KEY_1=your_gemini_api_key
GEMINI_API_KEY_2=optional_second_gemini_key
```

You need **at least one key** from Groq or Gemini. Multiple keys enable automatic rotation when rate limits are hit.

Get free API keys:
- Groq: https://console.groq.com
- Gemini: https://aistudio.google.com

### 3. (First time only) Build the vector database

If you have existing JSON transcripts in the `jsons/` folder, populate ChromaDB once:

```bash
python 5_migrate_to_chromadb.py
```

Skip this if you're starting fresh — the **Process Video** page handles everything.

### 4. Run the app

```bash
streamlit run Home.py
```

Opens at `http://localhost:8501`.

---

## Adding Videos

From inside the app, go to **Process Video** and either:
- Paste a **YouTube URL** — the app fetches the transcript (or downloads and transcribes audio if no captions exist)
- **Upload a video file** — the app transcribes it using Whisper

The video is automatically chunked, embedded, and added to ChromaDB. No manual steps needed.

---

## Project Structure

```
AI-Teaching-Assistant/
├── Home.py                    # Main chat interface (RAG Q&A)
├── pages/
│   ├── 1_📚_Video_Library.py
│   ├── 2_🎬_Process_Video.py
│   ├── 3_🧠_Quiz_Generator.py
│   ├── 4_📝_Smart_Notes.py
│   ├── 5_🔀_Compare_Videos.py
│   ├── 6_🕸️_Knowledge_Graph.py
│   └── 7_📊_Learner_Progress.py
├── gemini_helper.py           # Multi-provider LLM/transcription wrapper
├── learner_profile.py         # Quiz tracking and adaptive difficulty
├── theme.py                   # Centralized styling
├── 5_migrate_to_chromadb.py   # One-time DB migration from JSON transcripts
├── 6_evaluate.py              # RAG pipeline benchmarking tool
├── jsons/                     # Raw video transcripts (auto-generated)
├── chroma_db/                 # Vector database (auto-generated, gitignored)
├── requirements.txt
└── .env                       # API keys (gitignored)
```

---

## Evaluation

To benchmark the RAG pipeline against test questions:

```bash
python 6_evaluate.py
```

Measures hit rate, MRR (Mean Reciprocal Rank), and topic-wise accuracy across different `top_k` values. Results are saved to `evaluation_results.json`.
