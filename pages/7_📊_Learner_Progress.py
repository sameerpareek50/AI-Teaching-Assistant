import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from learner_profile import (
    load_profile, save_profile, get_weak_topics, get_weak_videos,
    get_overall_accuracy, _default_profile
)
from theme import get_theme, get_common_css
st.set_page_config(page_title="Learner Progress | Videx", page_icon="📊", layout="wide")

t = get_theme()

st.markdown(get_common_css(t) + f"""
<style>
    .stat-card {{
        background: {t['bg_card']}; border: 1px solid {t['border_card']};
        border-radius: 18px; padding: 1.5rem; text-align: center;
    }}
    .stat-value {{
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(135deg, {t['accent']}, {t['accent2']});
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .stat-label {{ color: {t['text_secondary']}; font-size: 0.85rem; margin-top: 0.3rem; text-transform: uppercase; letter-spacing: 1px; }}
    .progress-card {{
        background: {t['bg_card']}; border: 1px solid {t['border_card']};
        border-radius: 18px; padding: 1.5rem; margin: 0.8rem 0;
    }}
    .section-title {{ color: {t['text_primary']}; font-size: 1.2rem; font-weight: 700; margin-bottom: 1rem; }}
    .topic-row {{
        display: flex; align-items: center; gap: 1rem; margin: 0.6rem 0;
    }}
    .topic-name {{ color: {t['text_primary']}; font-size: 0.9rem; min-width: 200px; }}
    .topic-bar-bg {{
        flex: 1; height: 8px; background: {t['bg_card_hover']};
        border-radius: 4px; overflow: hidden;
    }}
    .topic-bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.5s ease; }}
    .topic-pct {{ color: {t['text_secondary']}; font-size: 0.8rem; min-width: 50px; text-align: right; }}
    .diff-badge {{
        display: inline-block; padding: 4px 16px; border-radius: 20px;
        font-size: 0.85rem; font-weight: 600; letter-spacing: 0.5px;
    }}
    .diff-easy {{ background: {t['success_bg']}; border: 1px solid {t['success_border']}; color: {t['success']}; }}
    .diff-medium {{ background: {t['warning_bg']}; border: 1px solid {t['warning_border']}; color: {t['warning']}; }}
    .diff-hard {{ background: {t['error_bg']}; border: 1px solid {t['error_border']}; color: {t['error']}; }}
    .history-row {{
        display: flex; justify-content: space-between; align-items: center;
        padding: 0.6rem 0; border-bottom: 1px solid {t['border_card']};
    }}
    .history-topic {{ color: {t['text_primary']}; font-size: 0.9rem; }}
    .history-score {{ font-weight: 600; font-size: 0.9rem; }}
    .history-date {{ color: {t['text_muted']}; font-size: 0.75rem; }}
    .recommendation-card {{
        background: {t['card_gradient']};
        border: 1px solid {t['border_card']}; border-radius: 14px;
        padding: 1.2rem; margin: 0.5rem 0; color: {t['text_heading']}; font-size: 0.9rem;
    }}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 1rem 0;">
        <div style="font-size:2rem;">🎓</div>
        <div style="font-size:1.1rem; font-weight:700; color:#ffffff;">VIDEX</div>
        <div style="font-size:0.7rem; color:#7aa8cc; letter-spacing:2px;">LEARNER PROGRESS</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <div class="page-title">Learner Progress</div>
    <div class="page-subtitle">Track your strengths, weaknesses, and quiz performance over time</div>
</div>
""", unsafe_allow_html=True)

profile = load_profile()

if profile["total_quizzes"] == 0:
    st.markdown(f"""
    <div style="text-align:center; padding:3rem; color:{t['text_secondary']};">
        <div style="font-size:3rem; margin-bottom:1rem;">📊</div>
        <div style="font-size:1.2rem; color:{t['text_primary']}; font-weight:600;">No quiz data yet</div>
        <div style="margin-top:0.5rem;">Take your first quiz to start tracking your progress!</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# --- Overview Stats ---
overall_acc = get_overall_accuracy(profile)
diff = profile["current_difficulty"]
diff_class = {"Easy": "diff-easy", "Medium": "diff-medium", "Hard": "diff-hard"}[diff]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class="stat-card">
        <div class="stat-value">{profile['total_quizzes']}</div>
        <div class="stat-label">Quizzes Taken</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="stat-card">
        <div class="stat-value">{overall_acc:.0f}%</div>
        <div class="stat-label">Overall Accuracy</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="stat-card">
        <div class="stat-value">{profile['total_correct']}/{profile['total_questions']}</div>
        <div class="stat-label">Correct Answers</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="stat-card">
        <div class="stat-value"><span class="diff-badge {diff_class}">{diff}</span></div>
        <div class="stat-label">Adaptive Level</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Performance Trend ---
if len(profile["quiz_history"]) >= 2:
    st.markdown('<div class="progress-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Performance Trend</div>', unsafe_allow_html=True)

    history = profile["quiz_history"]
    chart_data = []
    for i, h in enumerate(history):
        pct = (h["score"] / h["total"] * 100) if h["total"] > 0 else 0
        chart_data.append({"Quiz": i + 1, "Score (%)": round(pct, 1)})

    import pandas as pd
    df = pd.DataFrame(chart_data)
    st.line_chart(df, x="Quiz", y="Score (%)", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Topic Strengths & Weaknesses ---
col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<div class="progress-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Topic Performance</div>', unsafe_allow_html=True)

    topic_scores = profile.get("topic_scores", {})
    if topic_scores:
        sorted_topics = sorted(
            [(tp, d) for tp, d in topic_scores.items() if d["total"] >= 1],
            key=lambda x: x[1]["correct"] / x[1]["total"] if x[1]["total"] > 0 else 0,
            reverse=True
        )
        for topic, data in sorted_topics:
            acc = data["correct"] / data["total"] * 100 if data["total"] > 0 else 0
            if acc >= 70:
                bar_color = t['success']
            elif acc >= 50:
                bar_color = t['warning']
            else:
                bar_color = t['error']

            display_topic = topic[:30] + "..." if len(topic) > 30 else topic
            st.markdown(f"""
            <div class="topic-row">
                <span class="topic-name">{display_topic}</span>
                <div class="topic-bar-bg">
                    <div class="topic-bar-fill" style="width:{acc}%; background:{bar_color};"></div>
                </div>
                <span class="topic-pct" style="color:{bar_color};">{acc:.0f}%</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"<p style='color:{t['text_muted']};'>No topic data yet.</p>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="progress-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Video Performance</div>', unsafe_allow_html=True)

    video_scores = profile.get("video_scores", {})
    if video_scores:
        sorted_videos = sorted(
            [(v, d) for v, d in video_scores.items() if d["total"] >= 1],
            key=lambda x: x[1]["correct"] / x[1]["total"] if x[1]["total"] > 0 else 0,
            reverse=True
        )
        for video, data in sorted_videos:
            acc = data["correct"] / data["total"] * 100 if data["total"] > 0 else 0
            if acc >= 70:
                bar_color = t['success']
            elif acc >= 50:
                bar_color = t['warning']
            else:
                bar_color = t['error']

            display_video = video[:30] + "..." if len(video) > 30 else video
            st.markdown(f"""
            <div class="topic-row">
                <span class="topic-name">{display_video}</span>
                <div class="topic-bar-bg">
                    <div class="topic-bar-fill" style="width:{acc}%; background:{bar_color};"></div>
                </div>
                <span class="topic-pct" style="color:{bar_color};">{acc:.0f}%</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"<p style='color:{t['text_muted']};'>No video data yet.</p>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# --- Personalized Recommendations ---
st.markdown('<div class="progress-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Personalized Recommendations</div>', unsafe_allow_html=True)

weak_topics = get_weak_topics(profile, top_n=3)
weak_videos = get_weak_videos(profile, top_n=3)

if weak_topics:
    for w in weak_topics:
        if w["accuracy"] < 0.7:
            st.markdown(f"""
            <div class="recommendation-card">
                <b>Review: {w['topic']}</b> — Your accuracy is {w['accuracy']:.0%} ({w['correct']}/{w['total']} correct).
                Try taking a focused quiz on this topic or generate notes with "Focus on Weak Areas" style.
            </div>
            """, unsafe_allow_html=True)

if weak_videos:
    for w in weak_videos:
        if w["accuracy"] < 0.7:
            st.markdown(f"""
            <div class="recommendation-card">
                <b>Rewatch: {w['video']}</b> — Your accuracy is {w['accuracy']:.0%}.
                Consider rewatching this video and taking another quiz.
            </div>
            """, unsafe_allow_html=True)

if not weak_topics and not weak_videos:
    st.markdown("""
    <div class="recommendation-card">
        <b>Great work!</b> You're performing well across all topics. Keep challenging yourself with harder quizzes!
    </div>
    """, unsafe_allow_html=True)

if overall_acc >= 80:
    st.markdown("""
    <div class="recommendation-card">
        You're scoring above 80% overall — try Hard difficulty quizzes to push your understanding further.
    </div>
    """, unsafe_allow_html=True)
elif overall_acc < 50:
    st.markdown("""
    <div class="recommendation-card">
        Your overall accuracy is below 50%. Focus on one topic at a time — use Smart Notes with "Focus on Weak Areas" to build your foundation.
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- Quiz History ---
st.markdown('<div class="progress-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Recent Quiz History</div>', unsafe_allow_html=True)

history = profile["quiz_history"]
for h in reversed(history[-15:]):
    pct = (h["score"] / h["total"] * 100) if h["total"] > 0 else 0
    if pct >= 70:
        score_color = t['success']
    elif pct >= 50:
        score_color = t['warning']
    else:
        score_color = t['error']

    diff_label = h.get("difficulty", "—")
    date_str = h.get("timestamp", "")[:10]
    topic_display = h.get("topic", "—")
    if len(topic_display) > 40:
        topic_display = topic_display[:40] + "..."

    st.markdown(f"""
    <div class="history-row">
        <span class="history-topic">{topic_display}</span>
        <span class="history-score" style="color:{score_color};">{h['score']}/{h['total']} ({pct:.0f}%)</span>
        <span style="color:{t['text_secondary']}; font-size:0.8rem;">{diff_label}</span>
        <span class="history-date">{date_str}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- Recently Missed Questions ---
if profile.get("weak_questions"):
    st.markdown('<div class="progress-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Recently Missed Questions</div>', unsafe_allow_html=True)

    for wq in profile["weak_questions"][:10]:
        st.markdown(f"""
        <div style="background:{t['error_bg']}; border:1px solid {t['error_border']};
            border-radius:12px; padding:0.8rem 1rem; margin:0.5rem 0;">
            <div style="color:{t['text_primary']}; font-size:0.9rem; font-weight:500;">{wq['question']}</div>
            <div style="color:{t['error']}; font-size:0.8rem; margin-top:0.3rem;">Your answer: {wq['user_answer']}</div>
            <div style="color:{t['success']}; font-size:0.8rem;">Correct: {wq['correct_answer']}</div>
            <div style="color:{t['text_muted']}; font-size:0.75rem; margin-top:0.2rem;">{wq.get('explanation', '')}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Reset progress button
st.markdown("<br>", unsafe_allow_html=True)
if st.button("Reset All Progress", type="secondary"):
    st.warning("This will delete all your quiz history and learner data.")
    if st.button("Confirm Reset", type="primary"):
        save_profile(_default_profile())
        st.rerun()
