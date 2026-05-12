import streamlit as st
import chromadb
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from gemini_helper import generate
from theme import get_theme, get_common_css
CHROMA_PATH = "./chroma_db"

st.set_page_config(page_title="Compare Videos | Videx", page_icon="🔀", layout="wide")

t = get_theme()

page_css = f"""
<style>
    .compare-card {{
        background: {t['bg_card']}; border: 1px solid {t['border_card']};
        border-radius: 18px; padding: 1.5rem; margin: 0.5rem 0; color: {t['text_heading']}; line-height: 1.7;
    }}
    .compare-card h3 {{ color: {t['text_primary']} !important; margin-top: 0; }}
    .overlap-badge {{
        display: inline-block; background: rgba(240,147,251,0.15);
        border: 1px solid rgba(240,147,251,0.3); border-radius: 20px;
        padding: 3px 12px; font-size: 0.8rem; color: #f093fb; margin: 0.2rem;
    }}
    .unique-badge-a {{
        display: inline-block; background: {t['code_bg']};
        border: 1px solid rgba(102,126,234,0.3); border-radius: 20px;
        padding: 3px 12px; font-size: 0.8rem; color: {t['text_heading']}; margin: 0.2rem;
    }}
    .unique-badge-b {{
        display: inline-block; background: {t['success_bg']};
        border: 1px solid {t['success_border']}; border-radius: 20px;
        padding: 3px 12px; font-size: 0.8rem; color: {t['success']}; margin: 0.2rem;
    }}
</style>
"""
st.markdown(get_common_css(t) + page_css, unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 1rem 0;">
        <div style="font-size:2rem;">🎓</div>
        <div style="font-size:1.1rem; font-weight:700; color:{t['text_heading']};">VIDEX</div>
        <div style="font-size:0.7rem; color:{t['text_muted']}; letter-spacing:2px;">COMPARE VIDEOS</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <div class="page-title">Video Comparison</div>
    <div class="page-subtitle">Compare two videos to find overlapping and unique topics</div>
</div>
""", unsafe_allow_html=True)

# Load video list from JSON files + ChromaDB (for YouTube/uploaded videos)
jsons_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "jsons")
videos = {}
if os.path.isdir(jsons_dir):
    for json_file in sorted(os.listdir(jsons_dir)):
        with open(f"{jsons_dir}/{json_file}") as f:
            content = json.load(f)
        chunks = content['chunks']
        if chunks:
            title = chunks[0]['title']
            number = chunks[0]['number']
            full_text = " ".join(c['text'] for c in chunks)
            videos[f"Video {number}: {title}"] = full_text

# Also load YouTube/uploaded videos from ChromaDB
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection("video_chunks")
    all_data = collection.get(include=["documents", "metadatas"])
    yt_videos = {}
    for i, meta in enumerate(all_data["metadatas"]):
        if meta.get("source") == "youtube":
            title = meta["title"]
            if title not in yt_videos:
                yt_videos[title] = []
            yt_videos[title].append(all_data["documents"][i])
    for title, texts in yt_videos.items():
        videos[title] = " ".join(texts)
except:
    pass

video_list = list(videos.keys())

col1, col2 = st.columns(2)
with col1:
    video_a = st.selectbox("Video A", video_list, index=0 if video_list else 0)
with col2:
    video_b = st.selectbox("Video B", video_list, index=min(1, len(video_list)-1) if video_list else 0)

compare_btn = st.button("Compare Videos", type="primary", use_container_width=True)

if "comparison_result" not in st.session_state:
    st.session_state.comparison_result = None

if compare_btn and video_a != video_b:
    with st.spinner("Analyzing and comparing video content..."):
        text_a = videos[video_a][:4000]
        text_b = videos[video_b][:4000]

        prompt = f"""Compare these two video transcripts and identify:

VIDEO A - {video_a}:
{text_a}

VIDEO B - {video_b}:
{text_b}

Output ONLY valid JSON with this structure:
{{
    "video_a_summary": "2-3 sentence summary of Video A",
    "video_b_summary": "2-3 sentence summary of Video B",
    "overlapping_topics": ["topic1", "topic2"],
    "unique_to_a": ["topic1", "topic2"],
    "unique_to_b": ["topic1", "topic2"],
    "recommendation": "Which video to watch first and why",
    "similarity_score": 0.0 to 1.0
}}

Output valid JSON only, no markdown."""

        try:
            text = generate(prompt, status_callback=st.warning)
            text = text.strip()
            if text.startswith("```"):
                import re
                text = re.sub(r'^```\w*\n?', '', text)
                text = re.sub(r'\n?```$', '', text)
            st.session_state.comparison_result = json.loads(text)
        except Exception as e:
            st.error(f"Comparison failed: {e}")

    st.rerun()

elif compare_btn and video_a == video_b:
    st.warning("Please select two different videos to compare.")

if st.session_state.comparison_result:
    r = st.session_state.comparison_result
    sim = r.get('similarity_score', 0)

    st.markdown(f"""
    <div style="text-align:center; margin:1.5rem 0;">
        <div style="font-size:2.5rem; font-weight:800; background:linear-gradient(135deg,{t['accent']},{t['accent2']});
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            {sim:.0%} Similar
        </div>
        <div style="color:{t['text_secondary']}; font-size:0.9rem;">Content overlap between the two videos</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div class="compare-card">
            <h3 style="color:{t['text_heading']} !important;">Video A</h3>
            <p style="color:{t['text_explanation']};">{r.get('video_a_summary', '')}</p>
            <div style="margin-top:0.8rem;"><b style="color:{t['text_secondary']};">Unique topics:</b></div>
            {''.join(f'<span class="unique-badge-a">{topic}</span>' for topic in r.get('unique_to_a', []))}
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""<div class="compare-card">
            <h3 style="color:{t['success']} !important;">Video B</h3>
            <p style="color:{t['text_explanation']};">{r.get('video_b_summary', '')}</p>
            <div style="margin-top:0.8rem;"><b style="color:{t['text_secondary']};">Unique topics:</b></div>
            {''.join(f'<span class="unique-badge-b">{topic}</span>' for topic in r.get('unique_to_b', []))}
        </div>""", unsafe_allow_html=True)

    # Overlapping topics
    st.markdown(f"""<div style="text-align:center; margin:1rem 0;">
        <b style="color:{t['accent']};">Overlapping Topics</b>
    </div>""", unsafe_allow_html=True)
    overlap_html = ''.join(f'<span class="overlap-badge">{topic}</span>' for topic in r.get('overlapping_topics', []))
    st.markdown(f'<div style="text-align:center;">{overlap_html}</div>', unsafe_allow_html=True)

    # Recommendation
    st.markdown(f"""<div class="compare-card" style="border-color:rgba(240,147,251,0.2); margin-top:1.5rem;">
        <h3 style="color:{t['accent']} !important;">Recommendation</h3>
        <p style="color:{t['text_sidebar']};">{r.get('recommendation', '')}</p>
    </div>""", unsafe_allow_html=True)

    if st.button("Clear Comparison"):
        st.session_state.comparison_result = None
        st.rerun()
