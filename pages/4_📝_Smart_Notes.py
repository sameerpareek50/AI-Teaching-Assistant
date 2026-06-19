import streamlit as st
import chromadb
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from learner_profile import load_profile, get_weak_topics, get_overall_accuracy
from gemini_helper import generate
from theme import get_theme, get_common_css
CHROMA_PATH = "./chroma_db"

st.set_page_config(page_title="Smart Notes | Videx", page_icon="📝", layout="wide")

t = get_theme()
st.markdown(get_common_css(t), unsafe_allow_html=True)
st.markdown(f"""
<style>
    .notes-container {{
        background: {t['bg_card']}; border: 1px solid {t['border_card']};
        border-radius: 18px; padding: 2rem; margin: 1rem 0;
        color: {t['text_heading']}; line-height: 1.8;
    }}
    .notes-container h1, .notes-container h2, .notes-container h3 {{
        color: {t['text_primary']} !important;
        border-bottom: 1px solid {t['border_card']};
        padding-bottom: 0.5rem;
    }}
    .notes-container strong {{ color: {t['text_heading']}; }}
    .notes-container li {{ color: {t['text_primary']}; margin: 0.3rem 0; }}
    .notes-container code {{
        background: {t['code_bg']}; padding: 2px 6px;
        border-radius: 4px; color: {t['text_heading']}; font-size: 0.9rem;
    }}
    .notes-container blockquote {{
        border-left: 3px solid {t['accent']}; padding-left: 1rem;
        color: {t['text_explanation']}; font-style: italic;
    }}
</style>
""", unsafe_allow_html=True)

profile = load_profile()

with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 1rem 0;">
        <div style="font-size:2rem;">🎓</div>
        <div style="font-size:1.1rem; font-weight:700; color:#ffffff;">VIDEX</div>
        <div style="font-size:0.7rem; color:#7aa8cc; letter-spacing:2px;">SMART NOTES</div>
    </div>
    """, unsafe_allow_html=True)

    if profile["total_quizzes"] > 0:
        weak = get_weak_topics(profile, top_n=3)
        if weak:
            st.markdown('<div style="color:#e2eeff; font-size:1rem; font-weight:700; margin-bottom:0.5rem;">Your Weak Areas</div>', unsafe_allow_html=True)
            for w in weak:
                st.markdown(f'<div style="color:#b8d4f0; font-size:0.85rem;">— {w["topic"]} ({w["accuracy"]:.0%})</div>', unsafe_allow_html=True)
            st.markdown('<div style="color:#7aa8cc; font-size:0.78rem; margin-top:0.4rem;"><em>Use \'Focus on Weak Areas\' note style to target these.</em></div>', unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <div class="page-title">Smart Notes Generator</div>
    <div class="page-subtitle">Generate structured study notes from any indexed video</div>
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

# Selection
selected_video = st.selectbox("Select a Video", list(videos.keys()) if videos else ["No videos found"])

note_styles = [
    "Comprehensive Study Notes",
    "Quick Revision (Bullet Points)",
    "Key Concepts & Definitions",
    "Code Examples & Syntax",
    "Exam Prep (Q&A Format)",
]
if profile["total_quizzes"] > 0:
    note_styles.insert(0, "Focus on Weak Areas")

note_style = st.selectbox("Note Style", note_styles)

generate_btn = st.button("Generate Notes", type="primary", use_container_width=True)

if "generated_notes" not in st.session_state:
    st.session_state.generated_notes = None

if generate_btn and selected_video in videos:
    with st.spinner("Generating smart notes..."):
        content_text = videos[selected_video][:8000]

        style_prompts = {
            "Comprehensive Study Notes": "Create detailed, well-structured study notes with headings, subheadings, bullet points, key terms in bold, and code examples where relevant.",
            "Quick Revision (Bullet Points)": "Create concise bullet-point revision notes. Focus on key facts, definitions, and important points only. Keep it short and scannable.",
            "Key Concepts & Definitions": "Extract and explain all key concepts and technical terms. Format as a glossary with term: definition pairs.",
            "Code Examples & Syntax": "Focus on extracting all code examples, syntax patterns, and practical usage. Show code snippets with brief explanations.",
            "Exam Prep (Q&A Format)": "Create study material in Q&A format — important questions with concise answers that a student might be asked in an exam."
        }

        # Build weak areas context for adaptive notes
        weak_context = ""
        if note_style == "Focus on Weak Areas":
            weak = get_weak_topics(profile, top_n=5)
            weak_names = [w["topic"] for w in weak] if weak else []
            recent_wrong = profile.get("weak_questions", [])[:10]

            style_instruction = "Generate study notes that specifically target the learner's weak areas. "
            if weak_names:
                style_instruction += f"The learner struggles with these topics: {', '.join(weak_names)}. "
            if recent_wrong:
                style_instruction += "They recently got these questions wrong:\n"
                for wq in recent_wrong:
                    style_instruction += f"- Q: {wq['question']} (Correct answer: {wq['correct_answer']})\n"
            style_instruction += ("\nStructure the notes to:\n"
                "1. Start with a 'Key Gaps' section summarizing what the learner needs to review\n"
                "2. Explain each weak concept clearly with examples\n"
                "3. Include practice tips and common mistakes to avoid\n"
                "4. End with a quick self-check quiz (3-5 questions) targeting the weak areas")
        else:
            style_instruction = style_prompts[note_style]

        # Add general learner context if available
        if profile["total_quizzes"] > 0 and note_style != "Focus on Weak Areas":
            overall = get_overall_accuracy(profile)
            if overall is not None:
                weak_context = f"\n\n[Learner context: Overall quiz accuracy is {overall:.0f}%. Current level: {profile['current_difficulty']}.]"

        prompt = f"""You are a study notes generator for video-based learning content.

Video: {selected_video}
Transcript content:
{content_text}
{weak_context}
{style_instruction}

Use markdown formatting. Include relevant section headers. Make it visually organized and easy to study from."""

        try:
            st.session_state.generated_notes = generate(prompt, status_callback=st.warning)
        except Exception as e:
            st.error(f"Failed to generate notes: {e}")

    st.rerun()

if st.session_state.generated_notes:
    st.markdown("---")

    # Download button
    st.download_button(
        "Download Notes (.md)",
        st.session_state.generated_notes,
        file_name=f"notes_{selected_video.replace(' ', '_').replace(':', '')}.md",
        mime="text/markdown"
    )

    st.markdown(st.session_state.generated_notes)

    if st.button("Clear Notes"):
        st.session_state.generated_notes = None
        st.rerun()
