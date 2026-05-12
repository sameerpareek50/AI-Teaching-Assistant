# VIDEX — AI Teaching Assistant
## Phase 2 Presentation Content
### RAG-based Video Lecture Q&A System for the Sigma Web Development Course

---

## SLIDE 1: TITLE SLIDE

**Title:** VIDEX — AI Teaching Assistant
**Subtitle:** A RAG-based Intelligent Q&A System for Video Lecture Content
**Course:** Sigma Web Development Course (40 Videos)
**Phase:** Phase 2 — Web UI, Advanced Features & Evaluation
**Presented by:** [Your Name]
**Date:** [Presentation Date]

---

## SLIDE 2: PROBLEM STATEMENT

**The Problem:**
- Students spend hours watching long video lectures (30-60 min each) to find specific topics
- No way to search inside video content — only video titles are searchable
- Students forget which video covered what topic and at what timestamp
- Rewatching entire videos just to find a 2-minute explanation is extremely inefficient

**Real-world scenario:**
- A course has 40+ videos totaling 20+ hours of content
- A student wants to know "How does Flexbox work?" — they'd have to manually scan through multiple CSS videos
- There's no "Ctrl+F" for video lectures

**Our Solution:**
- VIDEX extracts, indexes, and makes video lecture content searchable using AI
- Students ask natural language questions and get precise answers with exact video timestamps

---

## SLIDE 3: WHAT IS RAG?

**Retrieval-Augmented Generation (RAG):**
- A technique that combines information retrieval with AI text generation
- Instead of relying solely on the LLM's training data, RAG retrieves relevant documents first
- The retrieved documents are injected into the LLM prompt as context
- This grounds the AI's answer in actual data, reducing hallucination

**Why RAG for this project:**
- LLMs don't know the content of our specific video lectures
- RAG lets us feed the actual transcript content to the LLM at query time
- The LLM can then generate accurate answers citing specific videos and timestamps
- No fine-tuning needed — works with any new video content immediately

**RAG Pipeline:**
1. User Question → Embedding → Vector Search → Top-K Relevant Chunks Retrieved
2. Retrieved Chunks + User Question → LLM Prompt → Grounded Answer

---

## SLIDE 4: SYSTEM ARCHITECTURE — INDEXING PIPELINE (OFFLINE)

**Step 1: Video to Audio Conversion**
- Tool: FFmpeg
- Input: Raw video lectures (.mp4)
- Output: Audio files (.mp3)
- Result: ~90% file size reduction while preserving all speech content

**Step 2: Speech-to-Text Transcription**
- Model: OpenAI Whisper large-v2
- Language: Hindi audio → English text translation
- Output: Time-stamped transcript segments (2-5 seconds each)
- Total: 7,718 raw segments across 40 videos

**Step 3: Semantic Chunk Merging**
- Strategy: Merge every 5 consecutive Whisper segments into one semantic chunk
- Why: Raw 2-5 second segments are too small for meaningful embedding
- Result: 7,718 segments → 1,560 chunks (5x compression ratio)
- Each chunk retains start and end timestamps

**Step 4: Vector Embedding & Storage**
- Embedding Model: all-MiniLM-L6-v2 (384-dimensional vectors, runs locally)
- Vector Database: ChromaDB (persistent storage, cosine similarity search)
- Total indexed: 1,560 semantic chunks with metadata (video title, number, timestamps)

---

## SLIDE 5: SYSTEM ARCHITECTURE — QUERY PIPELINE (REAL-TIME)

**Step 5: Semantic Search & Retrieval**
- User's natural language question is converted to a 384-dim vector embedding
- Cosine similarity search against all 1,560 stored chunk embeddings
- Top-K most relevant chunks retrieved with metadata
- Average search latency: ~68ms

**Step 6: LLM Answer Generation**
- LLM: Google Gemini 2.5 Flash (API-based)
- Retrieved chunks injected into a carefully crafted prompt
- LLM generates natural language answer referencing specific videos and timestamps
- Conversation memory: last 3 exchanges included for follow-up question support
- This is the RAG pattern — the LLM is grounded in actual course content

---

## SLIDE 6: TECHNOLOGY STACK

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Speech-to-Text | OpenAI Whisper large-v2 | Hindi → English audio transcription with timestamps |
| Embedding Model | all-MiniLM-L6-v2 | 384-dim sentence embeddings (runs locally, no API) |
| Vector Database | ChromaDB | Persistent vector storage with cosine similarity search |
| LLM | Google Gemini 2.5 Flash | RAG answer generation via API |
| Frontend | Streamlit | Multi-page web application with dark theme |
| Visualization | Plotly | Interactive evaluation charts and knowledge graph |
| Video Processing | FFmpeg + yt-dlp | Audio extraction, YouTube video download |
| Transcript API | youtube-transcript-api | YouTube caption extraction with auto-translation |

---

## SLIDE 7: KEY FEATURES — CHAT INTERFACE

**AI Chat with Source Citations:**
- Natural language Q&A about any topic in the course
- Each answer includes source video references with timestamps
- Performance metrics shown: search time, LLM response time, chunks retrieved
- Suggested questions for quick start
- Thumbs up/down feedback on answers
- Conversation memory — supports follow-up questions like "Tell me more about that"
- Chat export — download full conversation as text file
- Dark/Light theme toggle

**Example:**
- User: "How does Flexbox work?"
- VIDEX: "Flexbox is covered in Video 15: CSS Flexbox. Starting at 2m 30s, the video explains the flex container and flex items concept. The key properties like justify-content and align-items are demonstrated from 5m 15s..."

---

## SLIDE 8: KEY FEATURES — PROCESS ANY VIDEO

**YouTube URL Processing:**
- Paste any YouTube URL
- 3-level smart transcript fallback:
  1. English captions (instant, free)
  2. Any language → auto-translate to English (e.g., Hindi captions → English)
  3. Download audio → Gemini AI transcription (for videos with no captions at all)
- AI-generated summary of the video
- Content automatically indexed in ChromaDB for search

**File Upload Processing:**
- Upload any audio/video file (.mp3, .mp4, .wav, .m4a, .webm)
- Gemini AI transcribes the content to English
- Same pipeline: chunk merging → embedding → indexing

**Inline Q&A:**
- After processing, ask questions about the video directly on the same page
- No need to navigate to the Chat page

---

## SLIDE 9: KEY FEATURES — QUIZ GENERATOR

**AI-Generated Quizzes:**
- Select any topic (HTML, CSS, JavaScript) or a specific video
- Choose difficulty level: Easy, Medium, Hard
- Choose number of questions: 5, 10, or 15
- AI generates multiple-choice questions from actual course content
- Submit answers and get instant scoring with explanations
- Score card with percentage and grade

**How it works:**
1. Retrieves relevant chunks from ChromaDB based on selected topic
2. Sends course content to Gemini with quiz generation prompt
3. LLM generates structured MCQs with correct answers and explanations
4. Works for newly uploaded YouTube videos too

---

## SLIDE 10: KEY FEATURES — SMART NOTES GENERATOR

**AI-Generated Study Notes:**
- Select any video from the course (including uploaded YouTube videos)
- Choose note style:
  - Comprehensive Study Notes (headings, subheadings, bullet points)
  - Quick Revision (concise bullet points only)
  - Key Concepts & Definitions (glossary format)
  - Code Examples & Syntax (practical code snippets)
  - Exam Prep (Q&A format for exam preparation)
- Download notes as Markdown file
- Notes generated from actual video transcript content

---

## SLIDE 11: KEY FEATURES — VIDEO COMPARISON

**Compare Two Videos:**
- Select any two videos from the course
- AI analyzes both transcripts and identifies:
  - Similarity score (0-100%)
  - Overlapping topics (taught in both videos)
  - Unique topics to Video A
  - Unique topics to Video B
  - Recommendation: which to watch first and why
- Helps students plan their learning path efficiently

---

## SLIDE 12: KEY FEATURES — KNOWLEDGE GRAPH

**Visual Topic Relationship Map:**
- Interactive graph visualization built with Plotly
- 30 topic nodes covering HTML, CSS, JavaScript, Setup, and Projects
- Color-coded by category (HTML=red, CSS=blue, JS=yellow, Projects=purple, Setup=green)
- Edge thickness represents connection strength between topics
- Node size represents topic importance in the course
- Force-directed layout for natural visual clustering
- Shows how topics build on each other (e.g., DOM → Events → Todo App)

---

## SLIDE 13: KEY FEATURES — EVALUATION DASHBOARD

**Automated Performance Benchmarks:**
- 34 carefully crafted test questions across 4 topics
- Metrics evaluated:
  - Hit Rate at different Top-K values (K=1,3,5,7,10)
  - Mean Reciprocal Rank (MRR)
  - Topic-wise accuracy breakdown
  - Retrieval latency
  - End-to-end LLM accuracy (does the answer reference the correct video?)

**Topic Similarity Heatmap:**
- 8x8 matrix showing semantic similarity between topic areas
- Computed from vector embedding space proximity
- Reveals which topics share conceptual overlap

---

## SLIDE 14: EVALUATION RESULTS

**Retrieval Performance:**
| Top-K | Hit Rate | MRR | Avg Latency |
|-------|----------|-----|-------------|
| K=1 | 67.6% | 0.676 | 54ms |
| K=3 | 91.2% | 0.811 | 62ms |
| K=5 | 97.1% | 0.858 | 68ms |
| K=7 | 97.1% | 0.858 | 75ms |
| K=10 | 97.1% | 0.858 | 85ms |

**Best Configuration:** Top-K=5 (97.1% hit rate, 0.858 MRR, 68ms latency)

**Topic-wise Accuracy (Top-5):**
| Topic | Hit Rate | Questions |
|-------|----------|-----------|
| CSS | 100% | 12 |
| JavaScript | 100% | 10 |
| HTML | 91.7% | 12 |

**Key Insight:** K=5 is the optimal retrieval depth — it captures nearly all relevant results while keeping latency low. Beyond K=5, accuracy plateaus but latency increases.

---

## SLIDE 15: PHASE 1 vs PHASE 2 COMPARISON

| Aspect | Phase 1 | Phase 2 |
|--------|---------|---------|
| Interface | Command-line (CLI) | Web UI (Streamlit) |
| Vector Storage | Joblib pickle file | ChromaDB vector database |
| Embedding Model | BGE-M3 (Ollama, local) | all-MiniLM-L6-v2 (local) |
| LLM | Llama 3.2 (Ollama, local) | Gemini 2.5 Flash (API) |
| Chunking Strategy | Raw Whisper segments (7,252) | Merged semantic chunks (1,560) |
| Content Coverage | 18 videos (HTML + CSS) | 40 videos (HTML + CSS + JS) |
| Evaluation | Manual testing only | Automated benchmarks (34 questions) |
| Scalability | In-memory, single session | Persistent vector DB |
| Features | Single query, no history | Chat, Quiz, Notes, Comparison, Knowledge Graph, Reports |
| Video Input | Pre-processed only | Any YouTube URL or file upload in real-time |
| Language Support | English only | Hindi/multilingual (auto-translate) |
| User Experience | Single query, no history | Chat with memory, source cards, feedback, themes |

---

## SLIDE 16: CHUNKING STRATEGY OPTIMIZATION

**The Problem with Raw Whisper Segments:**
- Whisper outputs very small segments (2-5 seconds of speech each)
- Each segment contains only a sentence fragment
- Too small for meaningful semantic embedding — context is lost
- 7,718 raw segments = too many vectors, slower search

**Our Solution — Semantic Chunk Merging:**
- Merge every 5 consecutive segments into one larger chunk
- Each merged chunk contains ~15-25 seconds of continuous speech
- Preserves start/end timestamps from first and last segment
- Result: 7,718 → 1,560 chunks (4.95x compression)

**Impact:**
- Better embedding quality (more context per vector)
- 5x fewer vectors = faster search
- Improved retrieval accuracy (97.1% vs lower with raw segments)
- Reduced storage requirements

---

## SLIDE 17: MULTI-LANGUAGE TRANSCRIPT HANDLING

**3-Level Fallback Strategy:**

**Level 1: English Transcript (Instant)**
- Check YouTube for English captions (en, en-US, en-GB)
- Fastest method — no processing needed
- Works for most English-language videos

**Level 2: Auto-Translation (Fast)**
- If no English captions, check for any language captions
- Auto-translate to English using YouTube's translation API
- Works for Hindi, Spanish, French, etc. — any language with captions
- Free, no additional API key needed

**Level 3: AI Audio Transcription (Fallback)**
- If no captions at all, download the audio
- Upload to Gemini AI for multimodal transcription
- AI listens to the audio and generates English transcript with timestamps
- Works for any language, any video

---

## SLIDE 18: PROJECT REPORT GENERATOR

**One-Click Report Export:**
- Generates a complete HTML report with:
  - Project overview and metrics
  - Full system architecture documentation
  - Technology stack details
  - Evaluation results with all metrics
  - Feature list
  - Phase 1 vs Phase 2 comparison
  - Future work roadmap
- Download as HTML file
- Print as PDF from browser (Ctrl+P)
- Professionally formatted for submission

---

## SLIDE 19: LIVE DEMO MODE

**Guided Walkthrough:**
- Built-in demo mode on the main chat page
- 3 pre-configured demo steps:
  1. Basic course navigation question ("What is HTML?")
  2. Specific topic deep-dive ("How does Flexbox work?")
  3. Follow-up question demonstrating conversation memory
- One-click execution of each demo step
- Perfect for presentations — no typing needed
- Shows the full RAG pipeline in action

---

## SLIDE 20: TECHNICAL DEEP DIVE — HOW CHROMADB WORKS

**Vector Database:**
- ChromaDB stores documents as high-dimensional vectors (384 dimensions)
- When a document is added, it's automatically embedded by the built-in model
- Documents stored with metadata (video title, number, timestamps)
- Persistent storage — data survives server restarts

**Similarity Search:**
- Query text is embedded using the same model
- Cosine similarity computed between query vector and all stored vectors
- Top-K most similar vectors returned with their documents and metadata
- Sub-100ms latency even with 1,500+ vectors

**Why ChromaDB over alternatives:**
- Runs locally, no cloud dependency
- Built-in embedding model (no separate embedding API needed)
- Python-native, easy Streamlit integration
- Persistent storage with zero configuration

---

## SLIDE 21: NUMBERS & SCALE

| Metric | Value |
|--------|-------|
| Total videos indexed | 40 |
| Raw transcript segments | 7,718 |
| Semantic chunks (after merging) | 1,560 |
| Embedding dimensions | 384 |
| Vector database size | ~15 MB |
| Average search latency | 68ms |
| Best retrieval accuracy | 97.1% |
| MRR (Mean Reciprocal Rank) | 0.858 |
| Test questions evaluated | 34 |
| Topics covered | HTML, CSS, JavaScript, Setup |
| Supported languages | Any (via auto-translate + AI transcription) |
| Total app pages | 10 |
| Total features | 15+ |

---

## SLIDE 22: FUTURE WORK (PHASE 3)

1. **Research Paper** — Detailed experimental analysis with ablation studies
2. **Multi-modal Support** — Extract and index images/slides from video frames
3. **User Authentication** — Login system with personalized learning paths
4. **Fine-tuned Embeddings** — Train domain-specific embedding model for educational content
5. **Cloud Deployment** — Deploy on GCP/AWS for public access
6. **Collaborative Features** — Students can share notes, quiz scores, and discussions
7. **Progress Tracking** — Track which topics a student has studied and suggest next steps
8. **Advanced Evaluation** — RAGAS framework, human evaluation, A/B testing different models

---

## SLIDE 23: CHALLENGES & SOLUTIONS

| Challenge | Solution |
|-----------|----------|
| Whisper segments too small for embedding | Semantic chunk merging (5-to-1 compression) |
| Gemini embedding API rate limits (100 req/min) | Switched to ChromaDB's built-in local embedding model |
| YouTube blocking audio downloads (403 errors) | Added file upload alternative + Gemini audio transcription |
| Hindi video lectures | 3-level transcript fallback with auto-translation |
| LLM hallucination | RAG pattern grounds answers in actual transcript content |
| Large context for LLM | Retrieve only top-K relevant chunks, not entire transcripts |
| Slow initial indexing | One-time offline process; queries are real-time (~68ms) |

---

## SLIDE 24: THANK YOU

**VIDEX — AI Teaching Assistant**

Key Achievements:
- 97.1% retrieval accuracy on 34 benchmark questions
- 68ms average search latency
- 40 videos indexed, 1,560 semantic chunks
- 10-page web application with 15+ features
- Supports any YouTube video or uploaded file
- Multi-language support with automatic translation

**GitHub:** [Your GitHub URL]
**Live Demo:** localhost:8501

Questions?
