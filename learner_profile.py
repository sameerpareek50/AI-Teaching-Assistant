"""
Learner Profile System for Videx AI Teaching Assistant.
Tracks quiz performance, weak areas, and adapts difficulty automatically.
Persists data to a JSON file so progress survives across sessions.
"""

import json
import os
from datetime import datetime

PROFILE_PATH = "./learner_data.json"


def _default_profile():
    return {
        "quiz_history": [],          # List of quiz attempts
        "topic_scores": {},          # topic -> {"correct": N, "total": N}
        "video_scores": {},          # video_title -> {"correct": N, "total": N}
        "weak_questions": [],        # Questions the learner got wrong (recent)
        "current_difficulty": "Medium",
        "total_quizzes": 0,
        "total_correct": 0,
        "total_questions": 0,
    }


def load_profile():
    """Load learner profile from disk, or create a fresh one."""
    if os.path.exists(PROFILE_PATH):
        try:
            with open(PROFILE_PATH) as f:
                return json.load(f)
        except:
            pass
    return _default_profile()


def save_profile(profile):
    """Persist learner profile to disk."""
    with open(PROFILE_PATH, "w") as f:
        json.dump(profile, f, indent=2)


def record_quiz_result(profile, topic, video_title, questions, user_answers):
    """
    Record results of a quiz attempt. Updates scores and weak areas.

    questions: list of dicts with 'question', 'options', 'correct', 'explanation'
    user_answers: dict of {question_index: selected_option_text}

    Returns (profile, score, total, wrong_questions_list)
    """
    score = 0
    total = len(questions)
    wrong_questions = []

    for i, q in enumerate(questions):
        correct_option = q['options'][q['correct']]
        user_answer = user_answers.get(i, "")
        is_correct = user_answer == correct_option

        if is_correct:
            score += 1
        else:
            wrong_questions.append({
                "question": q['question'],
                "user_answer": user_answer,
                "correct_answer": correct_option,
                "explanation": q.get('explanation', ''),
                "topic": topic,
                "video": video_title,
            })

    # Update topic scores
    if topic not in profile["topic_scores"]:
        profile["topic_scores"][topic] = {"correct": 0, "total": 0}
    profile["topic_scores"][topic]["correct"] += score
    profile["topic_scores"][topic]["total"] += total

    # Update video scores
    if video_title and video_title != "All Topics":
        if video_title not in profile["video_scores"]:
            profile["video_scores"][video_title] = {"correct": 0, "total": 0}
        profile["video_scores"][video_title]["correct"] += score
        profile["video_scores"][video_title]["total"] += total

    # Update weak questions (keep last 30)
    profile["weak_questions"] = (wrong_questions + profile["weak_questions"])[:30]

    # Update totals
    profile["total_quizzes"] += 1
    profile["total_correct"] += score
    profile["total_questions"] += total

    # Record in quiz history
    profile["quiz_history"].append({
        "timestamp": datetime.now().isoformat(),
        "topic": topic,
        "video": video_title,
        "score": score,
        "total": total,
        "difficulty": profile["current_difficulty"],
    })
    # Keep last 50 quiz attempts
    profile["quiz_history"] = profile["quiz_history"][-50:]

    # Auto-adjust difficulty based on recent performance
    profile["current_difficulty"] = compute_adaptive_difficulty(profile)

    save_profile(profile)
    return profile, score, total, wrong_questions


def compute_adaptive_difficulty(profile):
    """
    Determine the next difficulty level based on recent quiz performance.
    Uses the last 3 quizzes to decide.
    """
    recent = profile["quiz_history"][-3:]
    if not recent:
        return "Medium"

    recent_pcts = [r["score"] / r["total"] * 100 for r in recent if r["total"] > 0]
    if not recent_pcts:
        return "Medium"

    avg = sum(recent_pcts) / len(recent_pcts)

    if avg >= 80:
        return "Hard"
    elif avg >= 50:
        return "Medium"
    else:
        return "Easy"


def get_weak_topics(profile, top_n=5):
    """Return topics sorted by weakness (lowest accuracy first)."""
    weak = []
    for topic, data in profile["topic_scores"].items():
        if data["total"] >= 2:  # Need at least 2 questions to judge
            accuracy = data["correct"] / data["total"]
            weak.append({"topic": topic, "accuracy": accuracy, **data})

    weak.sort(key=lambda x: x["accuracy"])
    return weak[:top_n]


def get_weak_videos(profile, top_n=5):
    """Return videos sorted by weakness (lowest accuracy first)."""
    weak = []
    for video, data in profile["video_scores"].items():
        if data["total"] >= 2:
            accuracy = data["correct"] / data["total"]
            weak.append({"video": video, "accuracy": accuracy, **data})

    weak.sort(key=lambda x: x["accuracy"])
    return weak[:top_n]


def get_overall_accuracy(profile):
    """Return overall accuracy percentage."""
    if profile["total_questions"] == 0:
        return None
    return profile["total_correct"] / profile["total_questions"] * 100


def get_learner_summary_for_prompt(profile):
    """
    Generate a concise learner context string to inject into RAG/chat prompts.
    Helps the AI focus on what the learner struggles with.
    """
    if profile["total_questions"] == 0:
        return ""

    lines = ["[Learner Profile]"]

    overall = get_overall_accuracy(profile)
    lines.append(f"Overall accuracy: {overall:.0f}% across {profile['total_quizzes']} quizzes")
    lines.append(f"Current difficulty level: {profile['current_difficulty']}")

    weak_topics = get_weak_topics(profile, top_n=3)
    if weak_topics:
        topics_str = ", ".join(f"{w['topic']} ({w['accuracy']:.0%})" for w in weak_topics)
        lines.append(f"Weak areas: {topics_str}")

    recent_wrong = profile["weak_questions"][:5]
    if recent_wrong:
        lines.append("Recently missed questions:")
        for wq in recent_wrong:
            lines.append(f"  - {wq['question']}")

    return "\n".join(lines)
