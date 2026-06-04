# VIDEX: An AI-Powered Video-Based Intelligent Teaching Assistant Using Retrieval-Augmented Generation

**Submitted Towards the Partial Fulfilment of the Requirements for the Award of the Degree of**
**Bachelor of Technology**

by

**Sameer Pareek** — Roll No. 22124087

**Under the Mentorship of**

Dr. Simranjit Singh
Assistant Professor

**Department of Information Technology**
Dr B R Ambedkar National Institute of Technology
Jalandhar-144008, Punjab (INDIA)

---

## Undertaking

We declare that the project work presented in this report entitled **VIDEX: An AI-Powered Video-Based Intelligent Teaching Assistant Using Retrieval-Augmented Generation**, submitted to the Department of Information Technology, Dr B R Ambedkar National Institute of Technology, Jalandhar, for the award of the Bachelor of Technology degree in Information Technology, is our original work. We have not plagiarized or submitted the same work for the award of any other degree. In case this undertaking is found incorrect, we accept that our degree may be unconditionally withdrawn.

Sameer Pareek (Roll No. 22124087)

Department of Information Technology
Dr B R Ambedkar National Institute of Technology
Jalandhar, Punjab, India
June, 2026

---

## Certificate

This is to certify that the project report entitled **VIDEX: An AI-Powered Video-Based Intelligent Teaching Assistant Using Retrieval-Augmented Generation** submitted by **Sameer Pareek (Roll No. 22124087)** to Dr B R Ambedkar National Institute of Technology Jalandhar, in partial fulfillment for the award of the degree of B. Tech in Information Technology has been carried out under our supervision and that this work has not been submitted elsewhere for a degree.

**Dr. Simranjit Singh**
Assistant Professor
Department of Information Technology
Dr B R Ambedkar National Institute of Technology
Jalandhar, Punjab, India
June, 2026

---

## Acknowledgement

We would like to express our sincere gratitude to all those who have contributed to the successful completion of this project.

First and foremost, we extend our heartfelt thanks to our project supervisor, **Dr. Simranjit Singh**, Assistant Professor, for their invaluable guidance, continuous support, and encouragement throughout the duration of this project. Their expertise and constructive feedback have been instrumental in shaping this work.

We are deeply grateful to **Dr. Vijay Kumar**, Head of the Department of Information Technology, for providing us with the necessary facilities and resources to carry out this project.

We would also like to thank all the faculty members of the Department of Information Technology for their support and valuable suggestions during various stages of this project.

Our sincere thanks go to our family and friends for their constant encouragement and support throughout our academic journey.

Finally, we acknowledge the contributions of all those who directly or indirectly helped us in completing this project successfully.

Sameer Pareek

---

## Abstract

This project presents VIDEX, an AI-powered, video-based intelligent teaching assistant built using Retrieval-Augmented Generation (RAG). With the rapid growth of online video-based education, learners face a significant challenge: video content is inherently non-searchable, non-queryable, and does not adapt to individual learning needs. Traditional e-learning platforms allow learners to only passively watch videos, with no mechanism to ask questions, test understanding, or receive personalised guidance.

VIDEX addresses this by building a complete end-to-end pipeline that ingests educational videos from multiple sources (local files, YouTube URLs, and direct uploads), transcribes them using a multi-provider fallback chain (Groq Whisper, Google Gemini, OpenAI Whisper), indexes the transcripts as semantic vector embeddings in ChromaDB, and enables natural language question-answering grounded in the video content via a RAG architecture. Key features include an adaptive quiz generator that adjusts difficulty based on learner performance, AI-generated personalised study notes, a knowledge graph visualiser, multi-video comparison, and a learner progress dashboard. The system employs a multi-provider LLM orchestration layer (Groq Llama-3.3-70B, Google Gemini 2.5 Flash, Ollama) with automatic fallback to ensure high availability. Evaluation on 34 test questions across HTML, CSS, and JavaScript content demonstrates a retrieval hit rate of 97.06% and a Mean Reciprocal Rank (MRR) of 0.858 at top-k=5, with an average retrieval latency of just 69ms. This work contributes a robust, fully offline-capable, adaptive AI teaching assistant that transforms passive video content into an interactive and personalised learning experience.

**Keywords:** Retrieval-Augmented Generation, Large Language Models, Vector Database, Speech Recognition, Adaptive Learning, Educational Technology, ChromaDB, Whisper

---

## Contents

- Acknowledgement
- Abstract
- List of Figures
- List of Tables
- Chapter 1: Introduction
- Chapter 2: Literature Review
- Chapter 3: Background and Preliminaries
- Chapter 4: Proposed Methodology
- Chapter 5: Experimental Results and Analysis
- Chapter 6: Conclusion and Future Work
- References
- Appendix A: Source Code Snippets
- Appendix B: User Manual
- Appendix C: Additional Results

---

## List of Figures

- Figure 4.1: Overall System Architecture of VIDEX
- Figure 4.2: Video Ingestion and Transcription Pipeline
- Figure 4.3: RAG Query Pipeline
- Figure 4.4: Multi-Provider LLM Fallback Chain
- Figure 4.5: Adaptive Learning Flow
- Figure 4.6: ChromaDB Data Schema
- Figure 4.7: Use Case Diagram
- Figure 4.8: Sequence Diagram for Chat Q&A

---

## List of Tables

- Table 2.1: Comparison of Existing Approaches
- Table 3.1: Technology Stack Summary
- Table 5.1: Retrieval Performance Metrics at Different Top-K Values
- Table 5.2: Topic-wise Retrieval Accuracy
- Table 5.3: End-to-End System Latency

---

# Chapter 1: Introduction

## 1.1 Overview

The proliferation of online video-based education has fundamentally changed how students acquire knowledge. Platforms such as YouTube, Coursera, and NPTEL host millions of hours of educational video content covering every conceivable subject. However, despite the abundance of this content, a critical gap exists: video is a passive medium. A learner watching a tutorial cannot pause and ask the video a question, cannot automatically generate a quiz to test their understanding, and cannot receive guidance tailored to their specific weaknesses.

This project addresses this gap by building VIDEX — a Video-based Intelligent Educational eXperience system — an AI-powered teaching assistant that transforms video content into a fully interactive, queryable, and adaptive learning environment. VIDEX ingests educational videos, extracts and indexes their content using state-of-the-art speech recognition and vector embedding models, and provides a suite of intelligent features including natural language Q&A, adaptive quizzes, personalised study notes, knowledge graphs, and video comparison — all powered by a Retrieval-Augmented Generation (RAG) architecture backed by multiple large language model providers.

The domain of Intelligent Tutoring Systems (ITS) has long sought to replicate the personalised attention of a human tutor. VIDEX represents a modern, LLM-powered instantiation of this vision, specifically designed for video-based learning environments.

## 1.2 Problem Statement

Despite the wide availability of educational video content, the following critical limitations remain unsolved:

1. **Non-searchability of video content**: Unlike text documents, video content cannot be semantically searched. A learner cannot ask "explain flexbox alignment" and be directed to the precise moment in a video where it is discussed.
2. **Passive consumption**: Learners passively watch videos without any mechanism to verify their understanding in real time.
3. **Lack of personalisation**: Existing platforms treat all learners the same — there is no system that identifies individual weak areas and adapts its content delivery accordingly.
4. **Language barriers**: Many high-quality educational videos are in languages other than English, making them inaccessible to a large audience.
5. **No cross-video intelligence**: Learners have no tool to understand how topics relate across multiple videos or to compare the coverage of different lectures.

## 1.3 Motivation

Key motivations behind this project include:

- The exponential growth of video-based content on platforms like YouTube, making intelligent video retrieval a critical need.
- The demonstrated effectiveness of RAG architectures in grounding LLM responses in factual, source-specific content, making them ideal for educational Q&A.
- The availability of powerful, free-tier AI APIs (Groq, Gemini) that make cloud-grade AI accessible without significant infrastructure cost.
- The need for a system that works offline and without institutional infrastructure, enabling deployment in resource-constrained environments.
- The personal observation that learners forget what was covered in which lecture, and spend significant time re-watching videos to find specific content.

## 1.4 Objectives

The primary objectives of this project are:

1. To build a complete pipeline for ingesting educational videos from multiple sources (local files, YouTube, uploads) and indexing their content as semantic vector embeddings.
2. To implement a RAG-based natural language Q&A system that can answer questions about video content with source citations and timestamps.
3. To develop an adaptive quiz generation system that adjusts question difficulty based on the learner's historical performance.
4. To create personalised study note generation in multiple formats (comprehensive notes, quick revision, key concepts, code examples, exam prep).
5. To build supporting analytical tools including a knowledge graph visualiser, multi-video comparison, and a learner progress dashboard.
6. To ensure system robustness through a multi-provider LLM fallback chain and offline capability via local models.

## 1.5 Scope of the Project

**In scope:**
- Video ingestion from YouTube URLs, local MP4 files, and direct file uploads (MP4, WebM, MP3, WAV, M4A)
- Speech transcription using Groq Whisper, Google Gemini, and local OpenAI Whisper
- Visual analysis for silent/no-audio videos using AI vision models
- Semantic indexing in ChromaDB using the all-MiniLM-L6-v2 embedding model
- RAG-based chat Q&A with source attribution and timestamps
- Adaptive quiz generation with difficulty auto-adjustment
- Personalised study notes in 6 styles
- Knowledge graph generation, video comparison, and learner progress tracking
- Subtitle generation and burning for uploaded videos
- Multi-provider LLM orchestration with automatic fallback

**Out of scope:**
- Real-time video streaming or live lecture capture
- Multi-user/multi-tenant deployment with authentication
- Mobile application development
- Integration with existing LMS platforms (Moodle, Canvas)
- Support for scanned PDFs or image-based content

## 1.6 Organization of Report

This report is organized as follows:

- **Chapter 2** presents a comprehensive literature review of existing work in intelligent tutoring systems, video retrieval, and RAG-based educational tools.
- **Chapter 3** describes the background concepts and technologies used, including RAG, vector databases, speech recognition, and LLMs.
- **Chapter 4** details the proposed methodology and system architecture, including all pipeline stages, modules, and design decisions.
- **Chapter 5** presents the experimental results and analysis, including retrieval benchmarks, latency measurements, and feature-level testing.
- **Chapter 6** concludes the report with key findings and future directions.

---

# Chapter 2: Literature Review

## 2.1 Introduction

This chapter surveys the existing literature and systems relevant to VIDEX across three domains: intelligent tutoring systems and educational AI, video content retrieval and indexing, and Retrieval-Augmented Generation architectures. The review identifies the strengths and limitations of existing approaches and establishes the research gaps that VIDEX addresses.

## 2.2 Existing Systems/Approaches

### 2.2.1 Traditional Approaches

Traditional e-learning platforms such as Moodle, Blackboard, and early versions of Coursera treated video as a standalone media file. Learner interaction was limited to play/pause controls, and search functionality was restricted to video titles and manually added tags. Closed captions, when available, were generated manually or via basic forced-alignment tools, and were not used for semantic search. Quizzes were static, authored by instructors, and did not adapt to individual learner performance. The primary limitation of these approaches is that they treat the rich semantic content within a video as opaque.

Keyword-based search systems used inverted indices over manually transcribed text. These systems suffered from vocabulary mismatch problems — a learner searching for "centering elements horizontally" would not find a video explaining `justify-content: center` unless those exact keywords appeared in the manual transcript. Furthermore, such systems provided no mechanism for follow-up questions or contextual conversation.

### 2.2.2 Modern Solutions

Modern platforms have begun leveraging AI to improve video-based learning. YouTube's automatic captions use Google's speech recognition to generate transcripts, enabling keyword search within videos. Khan Academy's Khanmigo uses GPT-4 to answer student questions, though it is not grounded in specific video content. Coursera has introduced AI-generated summaries for some courses.

However, none of these systems provide the complete pipeline that VIDEX offers: multi-source video ingestion, semantic vector search, adaptive quizzes based on personal performance history, multi-video knowledge graphs, and a fully local fallback chain. Most existing solutions are cloud-only, proprietary, and not extensible to custom video libraries.

## 2.3 Technology Review

### 2.3.1 Retrieval-Augmented Generation (RAG)

RAG, introduced by Lewis et al. (2020), combines a dense retrieval system with a generative language model. Given a user query, the retrieval system fetches the most semantically relevant documents from a corpus, which are then provided as context to the LLM to generate a grounded response. This approach significantly reduces hallucination, as the LLM is constrained to answer from retrieved evidence rather than parametric memory alone. VIDEX implements RAG using ChromaDB as the retrieval backend and Groq/Gemini/Ollama as the generative component.

### 2.3.2 Automatic Speech Recognition (ASR) and Whisper

OpenAI's Whisper (Radford et al., 2022) is a transformer-based ASR model trained on 680,000 hours of multilingual audio. It supports transcription and translation in over 99 languages. Whisper's `large-v2` variant achieves near-human accuracy on diverse audio conditions. VIDEX uses Whisper via both the Groq cloud API (whisper-large-v3, 8 hours/day free tier) and as a local model (base variant) for offline fallback.

### 2.3.3 Vector Databases and Semantic Embeddings

Vector databases store dense vector representations of text, enabling semantic similarity search via nearest-neighbour algorithms. ChromaDB is an open-source, embeddable vector database that provides persistent storage, automatic embedding generation, and efficient approximate nearest-neighbour search. VIDEX uses ChromaDB with the `all-MiniLM-L6-v2` sentence embedding model (384-dimensional vectors) from the `sentence-transformers` library.

### 2.3.4 Large Language Models

VIDEX integrates three LLM providers in a fallback hierarchy: Groq (Llama-3.3-70B, 100K tokens/day free), Google Gemini (gemini-2.5-flash, gemini-2.0-flash), and Ollama (gemma3:4b, local). This hierarchy ensures the system remains operational even when cloud API rate limits are exhausted, with complete offline capability via Ollama.

## 2.4 Comparative Analysis

| Approach | Advantages | Limitations |
|---|---|---|
| YouTube Auto-Captions | Free, automated, widely available | Keyword search only, no Q&A, no personalisation |
| Khan Academy Khanmigo | GPT-4 powered, conversational | Not grounded in specific videos, cloud-only |
| Coursera AI Summaries | Automated course summaries | No Q&A, no adaptive quizzes, proprietary |
| Traditional ITS (Moodle + Quizzes) | Structured learning paths | Static quizzes, no video grounding, manual content |
| **VIDEX (Proposed)** | **Semantic search, adaptive quizzes, multi-source ingestion, offline fallback, personalised notes** | **Requires initial video processing time** |

## 2.5 Research Gaps

The literature review reveals the following gaps that VIDEX addresses:

1. **No existing system** combines multi-source video ingestion, semantic RAG-based Q&A, and adaptive learning in a single unified platform.
2. **Hallucination in educational LLMs** is a known problem; RAG grounding with source citations and timestamps is not widely deployed in video-based tutoring systems.
3. **Offline capability** is absent from most modern AI tutoring tools, limiting their use in low-connectivity environments.
4. **Visual analysis fallback** for silent or non-English videos is not addressed by existing systems.
5. **Cross-video intelligence** — knowledge graphs and comparative analysis — has not been applied to personalised video-based learning.

## 2.6 Summary

Existing approaches either treat video as a passive medium without semantic search, or use LLMs without grounding them in specific video content. VIDEX fills this gap by combining robust video transcription, semantic vector indexing, RAG-based Q&A, and an adaptive learner model — creating a system that is simultaneously more capable, more personalised, and more robust than any single existing solution.

---

# Chapter 3: Background and Preliminaries

## 3.1 Introduction

This chapter introduces the core concepts and technologies that form the foundation of VIDEX. Understanding these concepts is essential to appreciate the design decisions made in the system architecture.

## 3.2 Fundamental Concepts

### 3.2.1 Retrieval-Augmented Generation (RAG)

RAG is a paradigm that augments a language model's parametric knowledge with non-parametric retrieval from an external knowledge base. The RAG pipeline consists of two phases:

**Indexing Phase:** Documents (in our case, video transcript chunks) are split into segments, converted to dense vector embeddings using an encoder model, and stored in a vector database.

**Inference Phase:** Given a user query, the query is also embedded into the same vector space. A similarity search retrieves the top-K most semantically relevant chunks. These chunks are concatenated with the user query and passed as context to the LLM, which generates a response grounded in the retrieved content.

The key advantage is that the LLM cannot "hallucinate" content that is not in the retrieved chunks, and the system can always point the user to the exact source (video number and timestamp) of the information.

### 3.2.2 Semantic Embeddings and Cosine Similarity

A sentence embedding model maps a natural language sentence to a fixed-dimensional vector such that semantically similar sentences are close in vector space. The `all-MiniLM-L6-v2` model produces 384-dimensional vectors. ChromaDB uses L2 distance (Euclidean) for similarity search, where a lower distance indicates higher similarity. A relevance score is computed as `1 - distance` for display in the UI.

### 3.2.3 Adaptive Learning

Adaptive learning systems dynamically adjust content difficulty based on learner performance. VIDEX implements a simple but effective adaptive model: after each quiz, the system computes the average score across the last 3 quizzes and adjusts the difficulty level (Easy / Medium / Hard) accordingly. Difficulty ≥80% accuracy triggers promotion to Hard; <50% triggers demotion to Easy. Weak topics (accuracy <70%) are surfaced in quiz generation prompts to encourage targeted practice.

### 3.2.4 Chunk Merging Strategy

Raw Whisper transcription produces segments of 2–5 seconds each, which are too granular for meaningful semantic search. VIDEX merges every 5 consecutive segments into a single chunk (producing 10–25 second chunks) before indexing. This 5x compression reduces the vector database size while preserving sufficient context for accurate retrieval.

## 3.3 Technologies and Tools

### 3.3.1 Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | Streamlit 1.x | Web UI framework, multi-page app |
| **Vector Database** | ChromaDB | Persistent semantic search index |
| **Embedding Model** | all-MiniLM-L6-v2 | Text-to-vector encoding |
| **Primary LLM** | Groq Llama-3.3-70B | Text generation (cloud, fast) |
| **Secondary LLM** | Google Gemini 2.5 Flash | Text generation (cloud, fallback) |
| **Tertiary LLM** | Ollama gemma3:4b | Text generation (local, offline) |
| **ASR (Cloud)** | Groq Whisper-large-v3 | Audio transcription |
| **ASR (Cloud 2)** | Google Gemini Audio | Audio transcription fallback |
| **ASR (Local)** | OpenAI Whisper base | Offline audio transcription |
| **Vision** | Groq Llama-4-Scout-17B | Frame analysis for silent videos |
| **Video Processing** | FFmpeg, yt-dlp | Audio extraction, subtitle burning |
| **Subtitle Rendering** | imageio_ffmpeg (libass) | Hardcoded subtitle generation |
| **Visualisation** | Plotly | Interactive knowledge graph |
| **Language** | Python 3.11 | Core implementation |

### 3.3.2 Development Environment

- **Operating System:** macOS (Apple Silicon)
- **IDE:** VS Code with Python extension
- **Version Control:** Git / GitHub (`sameerpareek50/AI-Teaching-Assistant`)
- **Package Management:** pip with virtual environment (venv)
- **Local LLM Server:** Ollama (running at `localhost:11434`)

## 3.4 Theoretical Framework

**RAG Retrieval Formulation:** Given a query q and a corpus of chunks C = {c₁, c₂, ..., cₙ}, the retrieval function R selects the top-K chunks:

```
R(q, C, K) = argtopK_{cᵢ ∈ C} similarity(embed(q), embed(cᵢ))
```

**Adaptive Difficulty Update Rule:**

```
avg_score = mean(scores of last 3 quizzes)
difficulty = Hard   if avg_score ≥ 0.80
           = Medium if 0.50 ≤ avg_score < 0.80
           = Easy   if avg_score < 0.50
```

**Chunk Merging:** For a video with segments S = {s₁, s₂, ..., sₘ}, merged chunks are formed as:

```
chunk_i = concat(s_{5i}, s_{5i+1}, s_{5i+2}, s_{5i+3}, s_{5i+4})
        for i = 0, 1, ..., floor(m/5)
```

## 3.5 Summary

The core technologies underlying VIDEX — RAG, vector databases, multi-provider LLMs, and ASR — are individually well-established. VIDEX's contribution lies in their novel integration into a unified, adaptive, video-focused educational platform with robust fallback mechanisms and offline capability.

---

# Chapter 4: Proposed Methodology

## 4.1 Introduction

VIDEX is designed as a two-phase system: an offline ingestion pipeline that processes video content into a searchable vector database, and an online inference layer that serves learner requests via a Streamlit web interface. This chapter describes the complete methodology, architecture, and module design.

## 4.2 System Overview

VIDEX takes educational video content as input — from local MP4 files, YouTube URLs, or direct browser uploads — and produces an interactive learning environment as output. The system:

1. Extracts and transcribes audio (or analyzes video frames if no speech is present)
2. Segments and indexes the transcript into a persistent ChromaDB vector database
3. Serves a 7-page Streamlit web application that uses RAG to answer questions, generate quizzes, create notes, visualise knowledge, and track learner progress

All processing is handled locally (using Ollama and local Whisper as fallback), making the system operable without any external API keys.

## 4.3 System Architecture

### 4.3.1 Architecture Design

```
┌──────────────────────────────────────────────────────┐
│                    INPUT SOURCES                      │
│   Local MP4  │  YouTube URL  │  Uploaded File        │
└──────────────┴───────────────┴───────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│            TRANSCRIPTION PIPELINE                     │
│  1. YouTube Captions → 2. Groq Whisper →             │
│  3. Gemini Audio → 4. Local Whisper →                │
│  5. Vision Frame Analysis                             │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│         CHUNK PROCESSING & INDEXING                  │
│  Segment Merging (5×) → ChromaDB Embedding           │
│  Collection: "video_chunks"                          │
│  Embedding: all-MiniLM-L6-v2 (384-dim)              │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│              CHROMADB VECTOR STORE                    │
│         (Persistent, ./chroma_db)                    │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│           STREAMLIT WEB APPLICATION                  │
│  Chat Q&A │ Quiz │ Notes │ Graph │ Compare │ Progress │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│         MULTI-PROVIDER LLM LAYER                     │
│   Groq (Primary) → Gemini (Secondary) →              │
│   Ollama (Tertiary / Offline)                        │
└──────────────────────────────────────────────────────┘
```

### 4.3.2 Component Description

**Transcription Engine:** Implements a 5-level fallback chain for maximum reliability. First attempts YouTube caption extraction (instant, no API cost). Falls back to Groq Whisper cloud transcription (fast, handles Hindi→English translation). Further falls back to Gemini audio transcription, local Whisper, and finally AI vision frame analysis for silent videos.

**Chunk Processor:** Merges raw 2–5 second Whisper segments into 10–25 second semantic chunks (merge size = 5). Assigns metadata: video number, title, start time, end time, source type.

**Vector Database (ChromaDB):** Persistent vector store using the `all-MiniLM-L6-v2` embedding model. Stores merged chunks with full metadata. Supports semantic similarity queries with optional metadata filtering.

**LLM Orchestration Layer (`gemini_helper.py`):** Routes generation requests through Groq → Gemini → Ollama with automatic key rotation on rate limits (3 Groq keys, 2 Gemini keys). Also handles audio transcription and vision-based frame description.

**Learner Profile Engine (`learner_profile.py`):** Persists quiz history, topic/video accuracy scores, and wrong questions to `learner_data.json`. Computes adaptive difficulty from last 3 quiz results. Generates context summaries injected into RAG prompts.

## 4.4 Module Description

### 4.4.1 Module 1: Video Ingestion Pipeline (`pages/2_Process_Video.py`)

**Purpose:** Ingest new videos into the knowledge base from any source.

**Functionality:**
- Extract YouTube video titles using `yt-dlp`
- Attempt 5-level transcript extraction with intelligent fallback
- Detect Whisper hallucinations via `no_speech_prob > 0.5` and `avg_logprob < -0.7`
- Merge segments and add to ChromaDB with `source` metadata tag
- Generate AI summary using the LLM orchestration layer
- Optionally burn SRT subtitles into the video using `imageio_ffmpeg`

**Implementation Details:** The module uses Python's `tempfile` for all intermediate processing, ensuring no permanent disk artifacts. The `imageio_ffmpeg` bundled binary (with `libass` support) is used for subtitle rendering, bypassing the system FFmpeg's missing `libass` dependency.

### 4.4.2 Module 2: RAG Chat Interface (`app.py`)

**Purpose:** Answer natural language questions about indexed video content.

**Functionality:**
- Load ChromaDB collection at startup (cached with `@st.cache_resource`)
- Accept user queries and retrieve top-K semantic chunks
- Build RAG prompt with retrieved chunks, chat history (last 3 turns), and learner context
- Stream LLM response via multi-provider fallback
- Display source citations with video title, timestamp, and relevance score

**Implementation Details:** Chat history is maintained in `st.session_state` for within-session continuity. Learner weak topic context is injected into the RAG prompt when quiz history is available, personalising responses to highlight areas of difficulty.

### 4.4.3 Module 3: Adaptive Quiz Generator (`pages/3_Quiz_Generator.py`)

**Purpose:** Generate and evaluate adaptive multiple-choice quizzes.

**Functionality:**
- Select topic (all videos or specific video)
- Auto-set difficulty based on learner's computed adaptive level
- Retrieve relevant chunks from ChromaDB
- Inject recently wrong questions into prompt for targeted practice
- Parse LLM-generated JSON quiz and display as interactive form
- Record results in learner profile and recompute difficulty

**Implementation Details:** The LLM is prompted to output strict JSON arrays with `question`, `options`, `correct`, `explanation`, and `concept` fields. A regex-based cleanup step strips markdown code fences before JSON parsing.

### 4.4.4 Module 4: Smart Notes Generator (`pages/4_Smart_Notes.py`)

**Purpose:** Generate personalised study notes in 6 styles.

**Functionality:**
- 5 standard styles: Comprehensive, Quick Revision, Key Concepts, Code Examples, Exam Prep Q&A
- 1 adaptive style: Focus on Weak Areas (uses learner's weak topics and recent wrong questions)
- Fetches up to 8000 characters of transcript as context
- Generates markdown-formatted notes with download option

### 4.4.5 Module 5: Knowledge Graph (`pages/6_Knowledge_Graph.py`)

**Purpose:** Visualise the topic network across all indexed videos.

**Functionality:**
- Load all videos from ChromaDB, extract representative content (1500 chars per video)
- Prompt LLM to generate JSON graph with nodes (topics) and edges (relationships)
- Apply force-directed layout with group-based clustering
- Render interactive Plotly graph with hover tooltips

### 4.4.6 Module 6: Video Comparison (`pages/5_Compare_Videos.py`)

**Purpose:** Analyse topic overlap and differences between two videos.

**Functionality:**
- Extract transcript content from ChromaDB for both selected videos
- Prompt LLM to output structured JSON: summaries, overlapping topics, unique topics, recommendation, similarity score
- Display side-by-side comparison with colour-coded badges

### 4.4.7 Module 7: Learner Progress Dashboard (`pages/7_Learner_Progress.py`)

**Purpose:** Track and visualise all quiz performance over time.

**Functionality:**
- Overview stats: total quizzes, overall accuracy, adaptive difficulty
- Line chart of quiz score trend
- Horizontal bar charts for topic and video accuracy (colour-coded by threshold)
- Personalised recommendations based on weak areas
- Recently missed questions with explanations

## 4.5 Algorithms and Techniques

### 4.5.1 RAG Retrieval Algorithm

```
Algorithm: RAG Query
Input: user_query q, top_k K, learner_profile P
Output: LLM response with citations

1. embed_q ← embed(q)  // using all-MiniLM-L6-v2 via ChromaDB
2. chunks ← ChromaDB.query(embed_q, K)
3. history ← last 3 messages from st.session_state
4. learner_ctx ← get_learner_summary(P) if P has quiz data
5. prompt ← build_prompt(q, chunks, history, learner_ctx)
6. response ← LLMOrchestrator.generate(prompt)
7. return response, chunks (as citations)
```

### 4.5.2 Transcription Fallback Algorithm

```
Algorithm: Smart Transcription
Input: video_source (URL or file)
Output: segments [{text, start, end}], method_used

1. Try YouTube captions (English) → return if successful
2. Try YouTube translated captions → return if successful
3. Download audio as MP3
4. Try Groq Whisper (if file ≤ 25MB)
   a. If is_transcription_useful(segments): return
   b. Else: set no_speech_detected = True
5. If not no_speech_detected:
   a. Try Gemini audio transcription → return if useful
   b. Try local Whisper → return if useful
6. If file is video: run vision frame analysis → return
7. Raise exception with all error details
```

### 4.5.3 SRT Subtitle Generation

```
Algorithm: generate_srt(segments)
Input: segments [{text, start, end}]
Output: SRT formatted string

For each segment:
  1. Split text by sentence boundaries (. ! ? \n)
  2. Distribute duration evenly across sentences
  3. For each sentence: wrap to max 50 chars/line, max 2 lines
  4. Format as SRT block with HH:MM:SS,mmm timestamps
Return concatenated SRT blocks
```

## 4.6 Design Diagrams

### 4.6.1 Data Flow Diagram

**Level 0 (Context):** User → VIDEX System → [ChromaDB, LLM APIs, Learner Profile] → User

**Level 1:**
- User submits video → Ingestion Pipeline → ChromaDB
- User submits query → RAG Engine → [ChromaDB, LLM] → Response + Citations
- User submits quiz answers → Quiz Engine → [LLM, Learner Profile] → Score + Feedback
- Learner Profile → Adaptive Engine → Difficulty Level

### 4.6.2 Entity-Relationship Diagram

**Entities:**
- **Video** (video_id, title, source, duration)
- **Chunk** (chunk_id, video_id, start, end, text, embedding)
- **Learner** (profile with quiz_history, topic_scores, weak_questions)
- **Quiz** (quiz_id, topic, difficulty, questions, score, timestamp)
- **Question** (question_id, text, options, correct_index, explanation, concept)

**Relationships:** Video has-many Chunks; Learner takes-many Quizzes; Quiz contains-many Questions; Learner has weak Questions

### 4.6.3 Use Case Diagram

**Actors:** Learner

**Use Cases:**
- Ask question about video content
- Process new YouTube video
- Upload local video file
- Generate adaptive quiz
- View quiz results and feedback
- Generate personalised study notes
- View knowledge graph
- Compare two videos
- View learning progress dashboard
- Download subtitled video

### 4.6.4 Sequence Diagram (Chat Q&A)

```
Learner → UI: Submit question q
UI → ChromaDB: query(embed(q), top_k=5)
ChromaDB → UI: Return top-5 chunks with metadata
UI → LLM Orchestrator: generate(RAG prompt)
LLM Orchestrator → Groq API: request
Groq API → LLM Orchestrator: response
LLM Orchestrator → UI: response text
UI → Learner: Display answer + source citations + timestamps
```

## 4.7 Database Design

VIDEX uses ChromaDB as its sole persistent data store for video content. ChromaDB is a document-oriented vector database with the following schema:

**Collection: `video_chunks`**

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique chunk identifier (e.g., `yt_abc123_0`) |
| `document` | string | Merged transcript text (10–25 seconds of speech) |
| `embedding` | float[384] | all-MiniLM-L6-v2 vector embedding |
| `metadata.number` | string | Video number or YouTube video ID prefix |
| `metadata.title` | string | Video title |
| `metadata.start` | float | Start time in seconds |
| `metadata.end` | float | End time in seconds |
| `metadata.chunk_id` | int | Sequential chunk ID |
| `metadata.source` | string | `"local"`, `"youtube"`, or `"upload"` |

**Learner Profile: `learner_data.json`**

A flat JSON file storing quiz history, topic/video accuracy scores, recent wrong questions, current difficulty level, and aggregate statistics. Persists across sessions.

## 4.8 User Interface Design

VIDEX uses a dark-themed Streamlit multi-page application with a consistent visual design system defined in `theme.py`. The colour palette uses deep navy backgrounds (#0f0c29), purple/magenta accents (#667eea, #764ba2), and semantic colours for feedback (green for success, red for error, yellow for warning).

The UI is structured as 7 pages accessible via the Streamlit sidebar, with each page following a consistent layout: page header, controls/inputs, processing status indicators, and results display. The chat interface uses Streamlit's native `st.chat_message` API. Performance metrics (retrieval latency, LLM latency, chunk count) are displayed below each chat response.

## 4.9 Security and Privacy

- **API keys** are stored in a `.env` file (excluded from version control via `.gitignore`) and loaded via `python-dotenv`.
- **Uploaded videos** are written to OS temporary directories (`tempfile.mkdtemp()`) and deleted immediately after processing.
- **Learner data** is stored locally in `learner_data.json` — no data is transmitted to any external service.
- **No authentication layer** is currently implemented, as the system is designed for single-user local deployment.

## 4.10 Implementation Strategy

Development followed an iterative, module-by-module approach:

1. **Phase 1 (Foundation):** Video-to-MP3 extraction, Whisper transcription, basic JSON storage.
2. **Phase 2 (RAG Core):** Embedding generation, ChromaDB migration, cosine similarity retrieval, Ollama-based Q&A.
3. **Phase 3 (Streamlit UI):** Main chat interface, multi-provider LLM fallback, source citations.
4. **Phase 4 (Feature Expansion):** Quiz generator, smart notes, knowledge graph, video comparison, learner progress.
5. **Phase 5 (Robustness):** Multi-source ingestion (YouTube, uploads), 5-level transcription fallback, vision analysis.
6. **Phase 6 (Enhancement):** Subtitle generation, adaptive source tagging, evaluation benchmarking.

## 4.11 Summary

VIDEX's methodology is built on three pillars: robustness (multi-level fallbacks for transcription and LLM generation), personalisation (adaptive difficulty, learner context injection into prompts), and completeness (a full suite of learning tools built on a single unified knowledge base). The ChromaDB vector store acts as the single source of truth, serving all seven application modules.

---

# Chapter 5: Experimental Results and Analysis

## 5.1 Introduction

This chapter presents the evaluation of VIDEX's performance across two dimensions: retrieval accuracy (how well the system finds the correct video content for a given query) and end-to-end system performance (latency and functional correctness). Evaluation was conducted on a dataset of 34 test questions covering HTML, CSS, JavaScript, and setup-related content from a web development course.

## 5.2 Experimental Setup

### 5.2.1 Hardware Configuration

- **Processor:** Apple M4 (ARM64, 10-core)
- **RAM:** 16 GB unified memory
- **Storage:** 512 GB SSD
- **GPU:** Apple M4 integrated GPU (used by Ollama for local inference)
- **Network:** Standard broadband (for cloud API calls)

### 5.2.2 Software Configuration

- **Operating System:** macOS (Apple Silicon)
- **Python Version:** 3.11
- **Key Libraries:** ChromaDB, Streamlit, Groq SDK, Google GenAI SDK, OpenAI Whisper, yt-dlp, FFmpeg 8.1, imageio_ffmpeg
- **Embedding Model:** all-MiniLM-L6-v2 (via ChromaDB default)
- **LLM (Evaluation):** Google Gemini 2.5 Flash
- **Test Dataset:** 34 curated questions with ground-truth video number labels
- **Knowledge Base:** 38 web development video transcripts (HTML, CSS, JavaScript)

## 5.3 Implementation Details

### 5.3.1 Feature Implementation

The system was implemented in approximately 3,500 lines of Python across 12 source files. The ChromaDB collection was populated by running `5_migrate_to_chromadb.py`, which processed 38 JSON transcript files, applied 5-segment chunk merging, and indexed the result. Evaluation was conducted by `6_evaluate.py`, which ran structured retrieval tests at multiple top-K values and an end-to-end Q&A evaluation with Gemini as the generative model.

## 5.4 Testing and Validation

### 5.4.1 Unit Testing

Individual components were tested in isolation:
- **`generate_srt()`**: Validated against manually crafted segment inputs, verifying correct SRT timestamp formatting and sentence-level splitting.
- **`burn_subtitles()`**: Tested against a synthetic MP4 video, confirming correct subtitle rendering and output file integrity.
- **`is_transcription_useful()`**: Verified against known silence audio (low speech probability) and speech audio (high speech probability).
- **`groq_transcribe()`**: Validated against a 10-second test audio clip, confirming segment structure with `text`, `start`, `end`, `no_speech_prob` fields.

### 5.4.2 Integration Testing

The full RAG pipeline was tested end-to-end: a user question was submitted, ChromaDB retrieved relevant chunks, the LLM generated a grounded response, and source citations were correctly attributed to the expected video and timestamp range.

### 5.4.3 System Testing

The 7-page Streamlit application was tested for UI correctness, including: proper display of source chunks, correct score calculation in the quiz module, accurate topic accuracy tracking in the learner profile, and correct subtitle embedding in the video download feature.

### 5.4.4 User Acceptance Testing

The system was evaluated against the following acceptance criteria:
- A question about a specific CSS property returns results from the correct CSS video: **PASS**
- Adaptive difficulty increases after 3 consecutive quizzes with >80% score: **PASS**
- YouTube video is successfully ingested and immediately queryable: **PASS**
- A silent video (no audio) is processed via vision analysis and returned as segments: **PASS**
- Downloaded subtitled video displays correct, time-synchronised subtitles: **PASS**

## 5.5 Results

### 5.5.1 Performance Metrics

| Top-K | Hit Rate | MRR | Avg. Latency (ms) | Hits / Total |
|---|---|---|---|---|
| 3 | 91.18% | 0.843 | 71.1 ms | 31 / 34 |
| **5** | **97.06%** | **0.858** | **69.5 ms** | **33 / 34** |
| 7 | 97.06% | 0.858 | 69.0 ms | 33 / 34 |
| 10 | 97.06% | 0.858 | 71.8 ms | 33 / 34 |

The system achieves peak performance at top-K=5, with a 97.06% hit rate and MRR of 0.858. Beyond K=5, no additional improvement is observed, confirming that 5 is the optimal retrieval count for this knowledge base.

### 5.5.2 Topic-wise Retrieval Accuracy

| Topic | Hits | Total | Accuracy |
|---|---|---|---|
| Setup (VS Code, Installation) | 2 | 2 | 100.0% |
| HTML | 11 | 12 | 91.67% |
| CSS | 10 | 10 | 100.0% |
| JavaScript | 10 | 10 | 100.0% |
| **Overall** | **33** | **34** | **97.06%** |

CSS and JavaScript topics achieve perfect retrieval accuracy. The single HTML miss is attributed to a question about a concept discussed across multiple videos, causing the retrieval to return a closely related but not exactly matching video.

### 5.5.3 End-to-End System Latency

| Component | Average Latency |
|---|---|
| ChromaDB Semantic Retrieval | ~69 ms |
| LLM Response (Gemini 2.5 Flash) | ~4.5 seconds |
| Video Transcription (Groq Whisper) | ~8 seconds (per 10-min video) |
| Subtitle Generation (FFmpeg) | ~15 seconds (per 10-min video) |

## 5.6 Analysis and Discussion

### 5.6.1 Key Findings

1. **97.06% retrieval accuracy at top-K=5** demonstrates that the combination of all-MiniLM-L6-v2 embeddings and the 5-segment chunk merging strategy is highly effective for educational content retrieval.
2. **Sub-100ms retrieval latency** confirms that ChromaDB's local persistent index is suitable for real-time interactive use.
3. **The 5-level transcription fallback** successfully handles videos with no captions, non-English audio, and even completely silent videos — a robustness characteristic not present in any existing comparable system.
4. **MRR of 0.858** indicates that the correct video appears as the first result in approximately 86% of queries, meaning learners are directed to the right content on the first try in the vast majority of cases.

### 5.6.2 Advantages

- **High accuracy**: 97.06% retrieval hit rate significantly exceeds keyword-based baselines.
- **Low latency**: 69ms retrieval makes the system feel instantaneous to users.
- **Robustness**: The fallback chain ensures functionality even with rate-limited APIs or missing audio.
- **Personalisation**: Adaptive difficulty and weak-topic injection are unique differentiators.
- **Offline capability**: Complete functionality is achievable without any internet access via Ollama and local Whisper.

### 5.6.3 Limitations

- **Single-user design**: The system stores learner data locally and is not designed for multi-user concurrent access.
- **LLM hallucination risk**: While RAG significantly reduces hallucination, the LLM can still occasionally misinterpret retrieved chunks.
- **Vision analysis quality**: For silent videos, frame-based descriptions are less precise than speech transcription.
- **Large model size**: The local Whisper `large-v2` model requires ~3GB of storage, making it impractical on low-storage devices.
- **Processing time**: Initial video ingestion (especially for long videos) can take several minutes.

## 5.7 Summary

VIDEX achieves strong empirical results with a 97.06% retrieval hit rate, 0.858 MRR, and sub-70ms retrieval latency on a 38-video knowledge base. The system successfully handles all tested input types and demonstrates robust adaptive behaviour in quiz generation and learner tracking.

---

# Chapter 6: Conclusion and Future Work

## 6.1 Conclusion

### 6.1.1 Summary of Work

This project designed, implemented, and evaluated VIDEX — an AI-powered video-based intelligent teaching assistant. Starting from raw educational video files, the system builds a complete end-to-end pipeline: audio extraction, multi-provider speech transcription, semantic chunk indexing in ChromaDB, and a 7-feature Streamlit web application. The project involved approximately 3,500 lines of Python code across 12 modules, integrating 8 distinct AI services and 3 LLM providers.

### 6.1.2 Key Contributions

1. **Multi-source video ingestion** with a 5-level transcription fallback chain, including AI vision analysis for silent videos — a novel approach not present in existing educational AI tools.
2. **RAG-based video Q&A** with source attribution to specific video timestamps, enabling verifiable, grounded answers.
3. **Adaptive quiz system** with automatic difficulty adjustment based on learner performance history and targeted weak-area reinforcement.
4. **Subtitle generation pipeline** that burns SRT subtitles into uploaded videos using the `imageio_ffmpeg` bundled binary, bypassing system-level FFmpeg limitations.
5. **Multi-provider LLM orchestration** with API key rotation and automatic fallback, ensuring near-100% uptime regardless of individual provider rate limits.

### 6.1.3 Achievement of Objectives

All primary objectives stated in Section 1.4 have been achieved:
- Multi-source video ingestion pipeline: **Achieved**
- RAG-based natural language Q&A: **Achieved** (97.06% hit rate)
- Adaptive quiz generation: **Achieved**
- Personalised study note generation: **Achieved** (6 styles)
- Supporting analytical tools: **Achieved** (knowledge graph, comparison, progress dashboard)
- Multi-provider fallback and offline capability: **Achieved**

### 6.1.4 Impact and Significance

VIDEX demonstrates that a single developer, using free-tier AI APIs and open-source libraries, can build an intelligent tutoring system that rivals the capabilities of funded commercial platforms. The system makes video-based educational content searchable, queryable, and adaptive — transforming passive video consumption into an active, personalised learning experience.

## 6.2 Future Scope

### 6.2.1 Short-term Enhancements

- **Multi-user support**: Add authentication (OAuth/JWT) and per-user learner profiles stored in a database.
- **Real-time lecture capture**: Integrate microphone input for live transcription and indexing of in-person lectures.
- **Mobile-responsive UI**: Adapt the Streamlit interface or rebuild in React for better mobile experience.
- **PDF and slide support**: Extend ingestion to handle lecture slides and PDF documents alongside videos.

### 6.2.2 Long-term Vision

- **LMS Integration**: Build connectors for Moodle, Canvas, and Google Classroom to deploy VIDEX within existing institutional infrastructure.
- **Collaborative learning**: Enable multiple learners to share a knowledge base, compare progress, and engage in collaborative Q&A.
- **Spaced repetition**: Integrate a spaced repetition algorithm (e.g., SM-2) to schedule quiz reviews at optimal retention intervals.
- **Voice interface**: Allow learners to ask questions via speech input, making the system hands-free.

### 6.2.3 Research Directions

- **Evaluation of adaptive learning efficacy**: Conduct longitudinal studies to measure whether the adaptive difficulty mechanism improves learner outcomes compared to static difficulty.
- **Multi-modal RAG**: Extend retrieval to include video frames alongside transcript text, enabling queries grounded in visual content.
- **Fine-tuned embedding models**: Train domain-specific embedding models on educational content to improve retrieval precision beyond the general-purpose all-MiniLM-L6-v2 baseline.
- **Hallucination detection**: Integrate automated hallucination detection to flag low-confidence LLM responses for human review.

## 6.3 Final Remarks

VIDEX represents a significant step toward making video-based education intelligent, interactive, and personalised. The project validates the practical applicability of RAG architectures for domain-specific educational Q&A and demonstrates the potential of combining multiple open-source and free-tier AI tools into a coherent, robust system. With further development, VIDEX has the potential to serve as a foundational platform for next-generation intelligent tutoring systems in universities and self-learning environments alike.

---

## References

[1] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *Advances in Neural Information Processing Systems*, 33, 9459–9474.

[2] Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022). Robust Speech Recognition via Large-Scale Weak Supervision. *OpenAI Technical Report*.

[3] Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *Proceedings of EMNLP 2019*.

[4] Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*, 7(3), 535–547.

[5] Anderson, J. R., Corbett, A. T., Koedinger, K. R., & Pelletier, R. (1995). Cognitive tutors: Lessons learned. *Journal of the Learning Sciences*, 4(2), 167–207.

[6] Bloom, B. S. (1984). The 2 Sigma Problem: The Search for Methods of Group Instruction as Effective as One-to-One Tutoring. *Educational Researcher*, 13(6), 4–16.

[7] Chroma DB Documentation. (2024). ChromaDB: The AI-native open-source embedding database. Retrieved from https://docs.trychroma.com/

[8] Groq Inc. (2024). Groq API Documentation: Llama-3.3-70B and Whisper-large-v3.

[9] Google DeepMind. (2024). Gemini 2.5 Flash Technical Report.

[10] Meta AI. (2024). Llama 3.3: Open Foundation Language Models.

---

## Appendix A: Source Code Snippets

### A.1 RAG Query and Prompt Construction (`app.py`)

```python
def query_chromadb(collection, query_text, top_k=5):
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    return results

def build_prompt(query, results, chat_history, learner_context=""):
    chunks_info = []
    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]
        chunks_info.append({
            "title": meta['title'], "number": meta['number'],
            "start": meta['start'], "end": meta['end'],
            "text": results['documents'][0][i]
        })
    history_context = ""
    if chat_history:
        recent = chat_history[-6:]
        history_context = "Recent conversation:\n" + "\n".join(
            [f"{m['role'].upper()}: {m['content'][:200]}" for m in recent]
        )
    return f"""You are an AI video assistant. Here are video transcript chunks:
{json.dumps(chunks_info)}
{history_context}
{learner_context}
"{query}"
Answer in a helpful, human way including video number and timestamp."""
```

### A.2 Multi-Provider LLM Fallback (`gemini_helper.py`)

```python
def generate(prompt, status_callback=None):
    errors = []
    # Priority 1: Groq
    try:
        return _groq_generate(prompt)
    except Exception as e:
        errors.append(f"Groq: {e}")
    # Priority 2: Gemini
    try:
        return _gemini_generate(prompt)
    except Exception as e:
        errors.append(f"Gemini: {e}")
    # Priority 3: Ollama (local fallback)
    try:
        return _ollama_generate(prompt)
    except Exception as e:
        errors.append(f"Ollama: {e}")
    raise Exception(f"All providers failed:\n" + "\n".join(errors))
```

### A.3 Adaptive Difficulty Computation (`learner_profile.py`)

```python
def compute_adaptive_difficulty(profile):
    history = profile["quiz_history"]
    if len(history) < 1:
        return "Easy"
    recent = history[-3:]
    scores = [h["score"] / h["total"] for h in recent if h["total"] > 0]
    if not scores:
        return profile["current_difficulty"]
    avg = sum(scores) / len(scores)
    if avg >= 0.80:
        return "Hard"
    elif avg >= 0.50:
        return "Medium"
    else:
        return "Easy"
```

---

## Appendix B: User Manual

### B.1 Installation

```bash
# Clone the repository
git clone https://github.com/sameerpareek50/AI-Teaching-Assistant.git
cd AI-Teaching-Assistant

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Set up API keys
cp .env.example .env
# Edit .env and add GEMINI_API_KEY_1, GROQ_API_KEY_1, etc.

# Index existing videos
python 5_migrate_to_chromadb.py
```

### B.2 Starting the Application

```bash
streamlit run app.py
```
Navigate to `http://localhost:8501` in your browser.

### B.3 Adding a New Video

1. Go to **Process Video** page in the sidebar
2. Paste a YouTube URL or switch to **Upload Audio/Video File** tab
3. Optionally check **Generate subtitled video** for video uploads
4. Click **Process** and wait for all 4 (or 5) steps to complete
5. The video is now searchable in the **Chat** page

### B.4 Taking a Quiz

1. Go to **Adaptive Quiz** page
2. Select a topic (or All Topics) and number of questions
3. The difficulty is auto-set based on your quiz history — you can override it
4. Submit answers and review explanations
5. Your performance is tracked and will influence future quiz difficulty

---

## Appendix C: Additional Results

### C.1 End-to-End Q&A Evaluation Sample

| Question | Expected Video | Correct | Latency |
|---|---|---|---|
| How to install VS Code? | Video 1 | Yes | 5.1s |
| What is HTML? | Video 2/3 | No | 4.5s |
| How to create headings in HTML? | Video 3 | Yes | 3.8s |
| What is CSS? | Video 8 | Yes | 4.2s |
| How does flexbox work? | Video 19 | Yes | 5.3s |
| What are JavaScript arrays? | Video 28 | Yes | 4.9s |

### C.2 Chunk Statistics

| Metric | Value |
|---|---|
| Total videos indexed | 38 |
| Raw Whisper segments | ~4,200 |
| Merged chunks (5× compression) | ~840 |
| Average chunk duration | ~18 seconds |
| ChromaDB collection size | ~12 MB |
| Embedding dimensions | 384 |
