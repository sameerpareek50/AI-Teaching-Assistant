import streamlit as st
import chromadb
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from learner_profile import load_profile, get_weak_topics, get_learner_summary_for_prompt
from gemini_helper import generate as gemini_generate
from theme import get_theme, get_common_css

# --- Configuration ---
CHROMA_PATH = "./chroma_db"
TOP_K = 5

# --- Page Config ---
st.set_page_config(
    page_title="AI Teaching Assistant | Videx",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={}
)

# --- Theme ---
t = get_theme()

# --- Global CSS ---
st.markdown(get_common_css(t) + f"""
<style>
    [data-testid="stToolbarActions"] {{ display: none !important; }}

    .hero-banner {{
        position: relative;
        background: linear-gradient(135deg, #eef2ff 0%, #f0f5ff 50%, #e8eefe 100%);
        border: 1px solid rgba(29,78,216,0.1);
        border-radius: 24px;
        overflow: hidden;
        margin-bottom: 1.8rem;
        animation: fadeIn 0.7s ease;
    }}
    .hero-banner::before {{
        content: '';
        position: absolute;
        width: 420px; height: 420px;
        background: radial-gradient(circle, rgba(59,130,246,0.18) 0%, transparent 65%);
        top: -140px; left: -120px;
        pointer-events: none;
    }}
    .hero-banner::after {{
        content: '';
        position: absolute;
        width: 320px; height: 320px;
        background: radial-gradient(circle, rgba(29,78,216,0.14) 0%, transparent 65%);
        bottom: -100px; right: -90px;
        pointer-events: none;
    }}
    .hero-dots {{
        position: absolute;
        inset: 0;
        background-image: radial-gradient(rgba(29,78,216,0.07) 1.5px, transparent 1.5px);
        background-size: 28px 28px;
        pointer-events: none;
    }}
    .welcome-container {{
        text-align: center;
        padding: 4rem 1rem 3rem;
        position: relative;
        z-index: 1;
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(-12px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(16px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    .welcome-icon {{ font-size: 3.5rem; display: block; margin-bottom: 1.2rem; }}
    .welcome-title {{
        font-size: 3.4rem; font-weight: 800; letter-spacing: 3px;
        background: linear-gradient(135deg, {t['accent']} 0%, #2563eb 60%, #3b82f6 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.6rem;
    }}
    .welcome-tagline {{
        font-size: 1rem; color: {t['text_secondary']}; font-weight: 400;
        letter-spacing: 0.5px; margin-bottom: 0;
    }}

    .feature-cards {{
        display: flex; gap: 1.2rem; justify-content: center;
        margin: 2.5rem 0 2rem;
        animation: fadeInUp 0.7s ease 0.2s both;
    }}
    .feature-card {{
        background: #ffffff;
        border: 1px solid rgba(29,78,216,0.1);
        border-radius: 18px; padding: 1.6rem 1.4rem;
        text-align: center; flex: 1; max-width: 220px;
        box-shadow: 0 2px 12px rgba(29,78,216,0.06);
        transition: all 0.25s ease;
        cursor: default;
    }}
    .feature-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 10px 28px rgba(29,78,216,0.12);
        border-color: rgba(29,78,216,0.22);
    }}
    .feature-card-icon {{ font-size: 2rem; margin-bottom: 0.8rem; display: block; }}
    .feature-card-title {{
        font-size: 0.9rem; font-weight: 700;
        color: {t['text_heading']}; margin-bottom: 0.45rem;
    }}
    .feature-card-desc {{
        font-size: 0.78rem; color: {t['text_secondary']}; line-height: 1.55;
    }}
    .source-card {{
        background: {t['bg_card']};
        border: 1px solid {t['border_card']}; padding: 1rem 1.3rem;
        border-radius: 14px; margin: 0.6rem 0; transition: all 0.3s ease;
        box-shadow: 0 1px 4px rgba(29,78,216,0.06);
    }}
    .source-card:hover {{
        border-color: rgba(29,78,216,0.3); transform: translateX(4px);
        box-shadow: 0 4px 16px rgba(29,78,216,0.1);
    }}
    .source-card h4 {{ color: {t['text_heading']} !important; margin: 0 0 0.4rem 0; font-size: 0.95rem; font-weight: 600; }}
    .source-card p {{ color: {t['text_explanation']} !important; margin: 0.2rem 0; font-size: 0.85rem; line-height: 1.5; }}

    .badge {{
        background: {t['code_bg']}; border: 1px solid rgba(29,78,216,0.2);
        padding: 3px 12px; border-radius: 20px; font-size: 0.78rem;
        color: {t['accent']}; display: inline-block; margin: 0.4rem 0.3rem 0 0;
    }}
    .badge-green {{
        background: {t['success_bg']}; border-color: {t['success_border']}; color: {t['success']};
    }}

    .stChatMessage {{ border-radius: 14px; }}
    [data-testid="stChatInput"] textarea {{ border-radius: 14px !important; }}

    .perf-bar {{
        background: {t['bg_card']}; border: 1px solid {t['border_card']};
        border-radius: 10px; padding: 0.5rem 1rem;
        display: flex; gap: 1.5rem; justify-content: center;
        margin-top: 0.5rem; font-size: 0.78rem; color: {t['text_muted']};
        box-shadow: 0 1px 4px rgba(29,78,216,0.05);
    }}
    .perf-item {{ display: flex; align-items: center; gap: 0.4rem; }}
    .perf-dot {{ width: 6px; height: 6px; border-radius: 50%; display: inline-block; }}
    .dot-green {{ background: {t['success']}; }}
    .dot-blue {{ background: {t['accent']}; }}
    .dot-teal {{ background: #0ea5e9; }}

    @keyframes pulse {{
        0% {{ opacity: 1; }} 50% {{ opacity: 0.4; }} 100% {{ opacity: 1; }}
    }}
    .live-dot {{
        width: 8px; height: 8px; background: {t['success']}; border-radius: 50%;
        display: inline-block; animation: pulse 2s infinite; margin-right: 6px;
    }}
</style>
""", unsafe_allow_html=True)


# --- Resource loading ---
@st.cache_resource
def load_chromadb():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection("video_chunks")


def query_chromadb(collection, query_text, top_k=TOP_K):
    return collection.query(
        query_texts=[query_text],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )


def inference(prompt):
    return gemini_generate(prompt)


def format_timestamp(seconds):
    mins = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{mins}m {secs}s"


def build_prompt(query, results, chat_history=None):
    chunks_info = []
    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]
        chunks_info.append({
            "title": meta['title'], "number": meta['number'],
            "start": meta['start'], "end": meta['end'],
            "text": results['documents'][0][i]
        })

    # Conversation memory: include recent chat history for follow-up support
    history_context = ""
    if chat_history:
        recent = chat_history[-6:]  # Last 3 exchanges
        history_lines = []
        for msg in recent:
            role = "Student" if msg["role"] == "user" else "Assistant"
            history_lines.append(f"{role}: {msg['content'][:300]}")
        history_context = f"\n\nPrevious conversation:\n" + "\n".join(history_lines) + "\n\n"

    # Inject learner context if available
    learner_context = ""
    profile = load_profile()
    learner_summary = get_learner_summary_for_prompt(profile)
    if learner_summary:
        learner_context = f"\n\n{learner_summary}\nAdapt your explanations to focus more on concepts the learner struggles with. If relevant to their question, provide extra detail on their weak areas.\n"

    return f'''You are an AI video assistant. Here are video transcript chunks containing video title, video number, start time in seconds, end time in seconds, and the text at that time:

{json.dumps(chunks_info)}
---------------------------------{history_context}{learner_context}
"{query}"
User asked this question related to the video content. Answer in a helpful, human way — explain where and how much content is taught in which video (including video number and timestamp) and guide the user to the right video. If the question is unrelated to the indexed video content, let them know politely. If the user is asking a follow-up question, use the conversation history to understand context.'''


# --- Sidebar ---
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 1rem 0;">
        <div style="font-size:2rem;">🎓</div>
        <div style="font-size:1.1rem; font-weight:700; color:#ffffff; letter-spacing:1px;">VIDEX</div>
        <div style="font-size:0.7rem; color:#7aa8cc; letter-spacing:2px;">AI TEACHING ASSISTANT</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="color:#e2eeff; font-size:1rem; font-weight:700; margin-bottom:0.6rem;">How it works</div>
    <div style="color:#b8d4f0; font-size:0.85rem; line-height:1.9;">
        <div style="color:#b8d4f0; margin-bottom:0.3rem;">1. Your question → vector embedding</div>
        <div style="color:#b8d4f0; margin-bottom:0.3rem;">2. Semantic search finds relevant chunks</div>
        <div style="color:#b8d4f0;">3. LLM generates answer with references</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Show learner-aware suggestions
    profile = load_profile()
    if profile["total_quizzes"] > 0:
        weak = get_weak_topics(profile, top_n=3)
        if weak:
            st.markdown('<div style="color:#e2eeff; font-size:1rem; font-weight:700; margin-bottom:0.5rem;">Suggested for You</div>', unsafe_allow_html=True)
            st.markdown('<div style="color:#7aa8cc; font-size:0.8rem; margin-bottom:0.4rem;"><em>Based on your quiz performance:</em></div>', unsafe_allow_html=True)
            for w in weak:
                st.markdown(f'<div style="color:#b8d4f0; font-size:0.85rem; margin:0.2rem 0;">— Ask about <strong style="color:#e2eeff;">{w["topic"]}</strong> ({w["accuracy"]:.0%})</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div style="color:#e2eeff; font-size:1rem; font-weight:700; margin-bottom:0.5rem;">Settings</div>', unsafe_allow_html=True)
    top_k = st.slider("Chunks to retrieve", 1, 10, TOP_K)
    show_sources = st.checkbox("Show source chunks", value=True)

    # Chat export
    if st.session_state.get("messages") and len(st.session_state.messages) > 0:
        export_text = ""
        for msg in st.session_state.messages:
            role = "You" if msg["role"] == "user" else "Videx AI"
            export_text += f"[{role}]\n{msg['content']}\n\n"
        st.download_button(
            "Export Chat History",
            export_text,
            file_name="videx_chat_export.txt",
            mime="text/plain",
            use_container_width=True
        )

    st.markdown(f"""
    <div style="text-align:center; padding:0.5rem 0;">
        <span class="live-dot"></span>
        <span style="color:{t['success']}; font-size:0.8rem;">System Online</span>
    </div>
    """, unsafe_allow_html=True)


# --- Main Content ---

# Load ChromaDB
try:
    collection = load_chromadb()
except Exception as e:
    st.error(f"Could not load ChromaDB. Run `python 5_migrate_to_chromadb.py` first.\n\nError: {e}")
    st.stop()

suggestion_clicked = None

# Welcome header — always visible, sits above chat history
st.markdown("""
<div class="hero-banner">
    <div class="hero-dots"></div>
    <div class="welcome-container">
        <span class="welcome-icon">🎓</span>
        <div class="welcome-title">VIDEX</div>
        <div class="welcome-tagline">Ask anything about your course videos</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Feature cards — only shown on empty chat
if not st.session_state.get("messages"):
    st.markdown(f"""
    <div class="feature-cards">
        <div class="feature-card">
            <span class="feature-card-icon">🔍</span>
            <div class="feature-card-title">Search & Ask</div>
            <div class="feature-card-desc">Ask questions across all indexed videos and get timestamped answers</div>
        </div>
        <div class="feature-card">
            <span class="feature-card-icon">🧠</span>
            <div class="feature-card-title">Adaptive Quizzes</div>
            <div class="feature-card-desc">Test your knowledge with quizzes that adjust to your performance</div>
        </div>
        <div class="feature-card">
            <span class="feature-card-icon">📝</span>
            <div class="feature-card-title">Smart Notes</div>
            <div class="feature-card-desc">Auto-generate structured study notes from any course video</div>
        </div>
        <div class="feature-card">
            <span class="feature-card-icon">🕸️</span>
            <div class="feature-card-title">Knowledge Graph</div>
            <div class="feature-card-desc">Visualise how topics connect across all your videos</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback" not in st.session_state:
    st.session_state.feedback = {}

# Display chat history
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if "sources" in message and show_sources:
                with st.expander("View Source Chunks", expanded=False):
                    for src in message["sources"]:
                        similarity = max(0, 1 - src.get('distance', 0))
                        st.markdown(f"""<div class="source-card">
                            <h4>Video {src['number']}: {src['title']}</h4>
                            <p>"{src['text'][:200]}..."</p>
                            <span class="badge">{format_timestamp(src['start'])} — {format_timestamp(src['end'])}</span>
                            <span class="badge badge-green">Relevance: {similarity:.0%}</span>
                        </div>""", unsafe_allow_html=True)

            if "perf" in message:
                p = message["perf"]
                st.markdown(f"""<div class="perf-bar">
                    <span class="perf-item"><span class="perf-dot dot-green"></span> Search: {p['search_ms']:.0f}ms</span>
                    <span class="perf-item"><span class="perf-dot dot-blue"></span> LLM: {p['llm_s']:.1f}s</span>
                    <span class="perf-item"><span class="perf-dot dot-purple"></span> Chunks: {p['chunks']}</span>
                </div>""", unsafe_allow_html=True)

            # Feedback buttons
            col1, col2, col3 = st.columns([1, 1, 20])
            with col1:
                if st.button("👍", key=f"up_{idx}"):
                    st.session_state.feedback[idx] = "positive"
            with col2:
                if st.button("👎", key=f"down_{idx}"):
                    st.session_state.feedback[idx] = "negative"
            if idx in st.session_state.feedback:
                fb = st.session_state.feedback[idx]
                st.caption(f"{'Thanks for the feedback!' if fb == 'positive' else 'Sorry, we will improve!'}")


# Handle input (chat box or suggestion click)
user_input = st.chat_input("Ask a question about your video content...")
if suggestion_clicked:
    user_input = suggestion_clicked

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # Search
        search_start = time.time()
        with st.spinner("Searching through video content..."):
            results = query_chromadb(collection, user_input, top_k=top_k)
        search_time = (time.time() - search_start) * 1000

        sources = []
        for i in range(len(results['ids'][0])):
            meta = results['metadatas'][0][i]
            sources.append({
                "number": meta['number'], "title": meta['title'],
                "start": meta['start'], "end": meta['end'],
                "text": results['documents'][0][i],
                "distance": results['distances'][0][i]
            })

        # LLM with conversation memory
        with st.spinner("Generating answer with AI..."):
            rag_prompt = build_prompt(user_input, results, chat_history=st.session_state.messages[:-1])
            llm_start = time.time()
            response = inference(rag_prompt)
            llm_time = time.time() - llm_start

        st.markdown(response)

        perf = {"search_ms": search_time, "llm_s": llm_time, "chunks": len(sources)}

        if show_sources:
            with st.expander("View Source Chunks", expanded=False):
                for src in sources:
                    similarity = max(0, 1 - src['distance'])
                    st.markdown(f"""<div class="source-card">
                        <h4>Video {src['number']}: {src['title']}</h4>
                        <p>"{src['text'][:200]}..."</p>
                        <span class="badge">{format_timestamp(src['start'])} — {format_timestamp(src['end'])}</span>
                        <span class="badge badge-green">Relevance: {similarity:.0%}</span>
                    </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="perf-bar">
            <span class="perf-item"><span class="perf-dot dot-green"></span> Search: {search_time:.0f}ms</span>
            <span class="perf-item"><span class="perf-dot dot-blue"></span> LLM: {llm_time:.1f}s</span>
            <span class="perf-item"><span class="perf-dot dot-purple"></span> Chunks: {len(sources)}</span>
        </div>""", unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant", "content": response,
        "sources": sources, "perf": perf
    })
    st.rerun()
