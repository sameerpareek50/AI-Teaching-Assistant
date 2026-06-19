import streamlit as st
import chromadb
import json
import re
import sys
import os

# Add parent directory to path so we can import shared modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from learner_profile import load_profile, save_profile, record_quiz_result, get_weak_topics
from gemini_helper import generate
from theme import get_theme, get_common_css
CHROMA_PATH = "./chroma_db"

st.set_page_config(page_title="Quiz Generator | Videx", page_icon="🧠", layout="wide")

t = get_theme()

PAGE_CSS = get_common_css(t) + f"""
<style>
    .quiz-card {{
        background: {t['bg_card']}; border: 1px solid {t['border_card']};
        border-radius: 18px; padding: 1.8rem; margin: 1rem 0;
    }}
    .quiz-question {{
        color: {t['text_primary']}; font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem; line-height: 1.6;
    }}
    .quiz-result-correct {{
        background: {t['success_bg']}; border: 1px solid {t['success_border']};
        border-radius: 12px; padding: 1rem; color: {t['success']}; margin: 0.5rem 0;
    }}
    .quiz-result-wrong {{
        background: rgba(234,102,102,0.15); border: 1px solid rgba(234,102,102,0.3);
        border-radius: 12px; padding: 1rem; color: {t['error']}; margin: 0.5rem 0;
    }}
    .score-card {{
        background: {t['card_gradient']};
        border: 1px solid rgba(59,130,246,0.3); border-radius: 18px;
        padding: 2rem; text-align: center; margin: 1.5rem 0;
    }}
    .score-value {{
        font-size: 3rem; font-weight: 800;
        background: linear-gradient(135deg, {t['accent']}, {t['accent2']});
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .score-label {{ color: {t['text_secondary']}; font-size: 1rem; margin-top: 0.5rem; }}
    .adaptive-banner {{
        background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(29,78,216,0.08));
        border: 1px solid rgba(59,130,246,0.2); border-radius: 14px;
        padding: 1rem 1.5rem; margin: 1rem 0; text-align: center;
    }}
    .difficulty-badge {{
        display: inline-block; padding: 4px 16px; border-radius: 20px;
        font-size: 0.85rem; font-weight: 600; letter-spacing: 0.5px;
    }}
    .diff-easy {{ background: {t['success_bg']}; border: 1px solid {t['success_border']}; color: {t['success']}; }}
    .diff-medium {{ background: {t['warning_bg']}; border: 1px solid {t['warning_border']}; color: {t['warning']}; }}
    .diff-hard {{ background: {t['error_bg']}; border: 1px solid {t['error_border']}; color: {t['error']}; }}
</style>
"""
st.markdown(PAGE_CSS, unsafe_allow_html=True)

# Load learner profile
profile = load_profile()

with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 1rem 0;">
        <div style="font-size:2rem;">🎓</div>
        <div style="font-size:1.1rem; font-weight:700; color:#ffffff;">VIDEX</div>
        <div style="font-size:0.7rem; color:#7aa8cc; letter-spacing:2px;">ADAPTIVE QUIZ</div>
    </div>
    """, unsafe_allow_html=True)

    # Show learner stats in sidebar
    if profile["total_quizzes"] > 0:
        overall_pct = profile["total_correct"] / profile["total_questions"] * 100 if profile["total_questions"] > 0 else 0
        st.markdown('<div style="color:#e2eeff; font-size:1rem; font-weight:700; margin-bottom:0.5rem;">Your Stats</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="color:#b8d4f0; font-size:0.85rem; line-height:1.8;">Quizzes taken: <strong style="color:#e2eeff;">{profile["total_quizzes"]}</strong><br>Overall accuracy: <strong style="color:#e2eeff;">{overall_pct:.0f}%</strong><br>Adaptive level: <strong style="color:#e2eeff;">{profile["current_difficulty"]}</strong></div>', unsafe_allow_html=True)

        weak = get_weak_topics(profile, top_n=3)
        if weak:
            st.markdown('<div style="color:#e2eeff; font-size:0.9rem; font-weight:600; margin-top:0.6rem;">Weak areas:</div>', unsafe_allow_html=True)
            for w in weak:
                st.markdown(f'<div style="color:#b8d4f0; font-size:0.85rem;">— {w["topic"]} ({w["accuracy"]:.0%})</div>', unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <div class="page-title">Adaptive Quiz</div>
    <div class="page-subtitle">Difficulty adjusts automatically based on your performance</div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_chromadb():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection("video_chunks")

try:
    collection = load_chromadb()
except:
    st.error("ChromaDB not available. Run the migration script first.")
    st.stop()

# Build video list from ChromaDB
all_data = collection.get(include=["metadatas"])
video_titles = {}
for meta in all_data["metadatas"]:
    title = meta.get("title", "Unknown")
    if title not in video_titles:
        video_titles[title] = meta.get("number", "?")

# Topic selection
topics = ["All Topics"] + [f"Video: {t}" for t in video_titles.keys()]

# Show adaptive difficulty banner
diff = profile["current_difficulty"]
diff_class = {"Easy": "diff-easy", "Medium": "diff-medium", "Hard": "diff-hard"}[diff]

if profile["total_quizzes"] > 0:
    st.markdown(f"""
    <div class="adaptive-banner">
        <span style="color:{t['text_secondary']}; font-size:0.9rem;">Adaptive difficulty:</span>
        <span class="difficulty-badge {diff_class}">{diff}</span>
        <span style="color:{t['text_muted']}; font-size:0.8rem; margin-left:0.5rem;">
            (based on your last {min(3, len(profile['quiz_history']))} quiz results)
        </span>
    </div>
    """, unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    selected_topic = st.selectbox("Select Topic or Video", topics)
with col2:
    num_questions = st.selectbox("Number of Questions", [5, 10, 15], index=0)
with col3:
    # Show adaptive difficulty as default, but let user override
    diff_options = ["Easy", "Medium", "Hard"]
    default_idx = diff_options.index(profile["current_difficulty"])
    difficulty = st.selectbox("Difficulty", diff_options, index=default_idx,
                              help="Auto-set by your performance. You can override.")

# Suggest weak areas as topics
weak_topics = get_weak_topics(profile, top_n=3)
if weak_topics and profile["total_quizzes"] >= 2:
    weak_names = [w["topic"] for w in weak_topics if w["accuracy"] < 0.7]
    if weak_names:
        st.markdown(f"""
        <div style="background:{t['error_bg']}; border:1px solid {t['error_border']};
            border-radius:12px; padding:0.8rem 1.2rem; margin:0.5rem 0; font-size:0.85rem; color:{t['error']};">
            Suggested focus areas based on your quiz history: <b>{', '.join(weak_names)}</b>
        </div>
        """, unsafe_allow_html=True)

generate_btn = st.button("Generate Quiz", type="primary", use_container_width=True)

if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
if "quiz_topic" not in st.session_state:
    st.session_state.quiz_topic = ""
if "quiz_video" not in st.session_state:
    st.session_state.quiz_video = ""
if "quiz_difficulty" not in st.session_state:
    st.session_state.quiz_difficulty = "Medium"

if generate_btn:
    st.session_state.quiz_submitted = False
    st.session_state.quiz_answers = {}
    st.session_state.quiz_topic = selected_topic
    st.session_state.quiz_difficulty = difficulty

    # Determine video title for profile tracking
    if selected_topic.startswith("Video: "):
        st.session_state.quiz_video = selected_topic[7:]
    else:
        st.session_state.quiz_video = selected_topic

    with st.spinner("Generating adaptive quiz..."):
        # Get relevant chunks
        if selected_topic.startswith("Video: "):
            video_title = selected_topic[7:]
            video_data = collection.get(
                where={"title": video_title},
                include=["documents", "metadatas"]
            )
            chunks_text = ""
            for i, doc in enumerate(video_data['documents']):
                chunks_text += f"[{video_title}]: {doc}\n\n"
        else:
            if selected_topic == "All Topics":
                query = "key concepts and topics covered in the videos"
            else:
                query = f"{selected_topic} concepts and features"
            results = collection.query(query_texts=[query], n_results=20, include=["documents", "metadatas"])
            chunks_text = ""
            for i, doc in enumerate(results['documents'][0]):
                title = results['metadatas'][0][i].get('title', '')
                chunks_text += f"[{title}]: {doc}\n\n"

        # Build adaptive prompt — include weak areas context
        weak_context = ""
        if profile["weak_questions"] and difficulty != "Easy":
            recent_wrong = profile["weak_questions"][:5]
            weak_context = "\n\nThe learner previously struggled with these questions (include similar concepts to help them practice):\n"
            for wq in recent_wrong:
                weak_context += f"- {wq['question']} (correct: {wq['correct_answer']})\n"

        prompt = f"""Based on this video content, generate exactly {num_questions} multiple choice questions at {difficulty} difficulty level.

Video content:
{chunks_text[:6000]}
{weak_context}
Output ONLY valid JSON array. Each element must have:
- "question": the question text
- "options": array of exactly 4 options (strings)
- "correct": index of correct option (0-3)
- "explanation": brief explanation of the correct answer
- "concept": the key concept being tested (1-3 words, e.g. "Flexbox", "DOM Events")

Topic focus: {selected_topic}
Difficulty: {difficulty} (Easy=basic recall with straightforward questions, Medium=understanding and application, Hard=analysis, edge cases, and tricky scenarios)

Output valid JSON only, no markdown."""

        try:
            text = generate(prompt, status_callback=st.warning)
            text = text.strip()
            if text.startswith("```"):
                text = re.sub(r'^```\w*\n?', '', text)
                text = re.sub(r'\n?```$', '', text)
            try:
                st.session_state.quiz_data = json.loads(text)
            except json.JSONDecodeError:
                # LLM added preamble text or the array is embedded — extract it
                match = re.search(r'\[.*\]', text, re.DOTALL)
                if match:
                    st.session_state.quiz_data = json.loads(match.group())
                else:
                    raise
            st.rerun()
        except Exception as e:
            st.error(f"Failed to generate quiz: {e}")
            st.session_state.quiz_data = None

# Display quiz
if st.session_state.quiz_data:
    questions = st.session_state.quiz_data

    if not st.session_state.quiz_submitted:
        with st.form("quiz_form"):
            for i, q in enumerate(questions):
                st.markdown(f"""<div class="quiz-card">
                    <div class="quiz-question">Q{i+1}. {q['question']}</div>
                </div>""", unsafe_allow_html=True)
                st.session_state.quiz_answers[i] = st.radio(
                    f"Select answer for Q{i+1}",
                    q['options'],
                    key=f"q_{i}",
                    label_visibility="collapsed"
                )

            submitted = st.form_submit_button("Submit Quiz", type="primary", use_container_width=True)
            if submitted:
                st.session_state.quiz_submitted = True
                st.rerun()
    else:
        # Record results in learner profile
        profile = load_profile()  # Reload fresh
        profile, score, total, wrong_questions = record_quiz_result(
            profile,
            topic=st.session_state.quiz_topic,
            video_title=st.session_state.quiz_video,
            questions=questions,
            user_answers=st.session_state.quiz_answers
        )

        # Show results
        for i, q in enumerate(questions):
            selected = st.session_state.quiz_answers.get(i, "")
            correct_option = q['options'][q['correct']]
            is_correct = selected == correct_option

            st.markdown(f"""<div class="quiz-card">
                <div class="quiz-question">Q{i+1}. {q['question']}</div>
            </div>""", unsafe_allow_html=True)

            if is_correct:
                st.markdown(f'<div class="quiz-result-correct">Your answer: {selected}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="quiz-result-wrong">Your answer: {selected}<br>Correct: {correct_option}</div>', unsafe_allow_html=True)

            st.markdown(f"<p style='color:{t['text_explanation']}; font-size:0.85rem; padding-left:1rem;'>{q.get('explanation', '')}</p>", unsafe_allow_html=True)

        pct = int((score / total) * 100)

        # Adaptive feedback
        new_diff = profile["current_difficulty"]
        old_diff = st.session_state.quiz_difficulty

        if pct >= 80:
            grade = "Excellent!"
            feedback_color = t['success']
        elif pct >= 60:
            grade = "Good job!"
            feedback_color = t['warning']
        else:
            grade = "Keep practicing!"
            feedback_color = t['error']

        # Score card
        st.markdown(f"""
        <div class="score-card">
            <div class="score-value">{score}/{total}</div>
            <div class="score-label">{pct}% — {grade}</div>
        </div>
        """, unsafe_allow_html=True)

        # Difficulty adjustment notification
        if new_diff != old_diff:
            direction = "increased" if diff_options.index(new_diff) > diff_options.index(old_diff) else "decreased"
            new_class = {"Easy": "diff-easy", "Medium": "diff-medium", "Hard": "diff-hard"}[new_diff]
            st.markdown(f"""
            <div class="adaptive-banner">
                <span style="color:{t['text_secondary']};">Difficulty {direction}:</span>
                <span class="difficulty-badge {new_class}">{new_diff}</span>
                <span style="color:{t['text_muted']}; font-size:0.8rem; margin-left:0.5rem;">
                    (next quiz will adapt to this level)
                </span>
            </div>
            """, unsafe_allow_html=True)

        # Show what to focus on if wrong answers exist
        if wrong_questions:
            concepts = list(set(q.get("concept", "General") for q in questions
                              if st.session_state.quiz_answers.get(questions.index(q), "") != q['options'][q['correct']]))
            if concepts:
                st.markdown(f"""
                <div style="background:{t['error_bg']}; border:1px solid {t['error_border']};
                    border-radius:12px; padding:1rem 1.2rem; margin:1rem 0; color:{t['error']}; font-size:0.9rem;">
                    <b>Focus areas from this quiz:</b> {', '.join(concepts)}<br>
                    <span style="color:{t['text_explanation']}; font-size:0.8rem;">
                        These concepts will be highlighted in your Smart Notes and revisited in future quizzes.
                    </span>
                </div>
                """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Retake Quiz", type="primary", use_container_width=True):
                st.session_state.quiz_submitted = False
                st.session_state.quiz_data = None
                st.session_state.quiz_answers = {}
                st.rerun()
        with col2:
            if st.button("New Quiz (Adaptive)", use_container_width=True):
                st.session_state.quiz_submitted = False
                st.session_state.quiz_data = None
                st.session_state.quiz_answers = {}
                st.rerun()
