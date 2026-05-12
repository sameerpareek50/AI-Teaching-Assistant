import streamlit as st
import chromadb
import json
import re
import math
import random
import sys
import os
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from gemini_helper import generate
from theme import get_theme, get_common_css
CHROMA_PATH = "./chroma_db"

st.set_page_config(page_title="Knowledge Graph | Videx", page_icon="🕸️", layout="wide")

t = get_theme()
st.markdown(get_common_css(t), unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 1rem 0;">
        <div style="font-size:2rem;">🎓</div>
        <div style="font-size:1.1rem; font-weight:700; color:{t['text_heading']};">VIDEX</div>
        <div style="font-size:0.7rem; color:{t['text_muted']}; letter-spacing:2px;">KNOWLEDGE GRAPH</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <div class="page-title">Topic Knowledge Graph</div>
    <div class="page-subtitle">Visual map of how topics connect across all your indexed videos</div>
</div>
""", unsafe_allow_html=True)


# --- Dynamically build knowledge graph from all indexed videos ---
@st.cache_data(ttl=120)
def build_knowledge_graph():
    """Extract topics from ALL indexed videos using Gemini to build a knowledge graph."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection("video_chunks")
        all_data = collection.get(include=["documents", "metadatas"])

        # Group chunks by video title
        videos = {}
        for i, meta in enumerate(all_data["metadatas"]):
            title = meta.get("title", "Unknown")
            if title not in videos:
                videos[title] = []
            videos[title].append(all_data["documents"][i])

        if not videos:
            return [], []

        # Build a combined content summary for Gemini
        video_summaries = []
        for title, chunks in videos.items():
            content = " ".join(chunks)[:1500]
            video_summaries.append(f"Video: {title}\nContent: {content}\n")

        combined = "\n---\n".join(video_summaries)[:12000]

        prompt = f"""Analyze these video transcripts and build a knowledge graph.

{combined}

Output ONLY valid JSON with this exact structure:
{{
    "nodes": [
        {{"id": "Topic Name", "group": "category", "size": 20}}
    ],
    "edges": [
        {{"from": "Topic A", "to": "Topic B", "strength": 2}}
    ]
}}

Rules:
- Extract 15-30 key topics/concepts from across all videos
- Each node needs: id (short name, 1-3 words), group (a category like "core", "subtopic", "tool", "project"), size (15-40 based on importance)
- Include edges between related topics. Strength: 1=weak, 2=medium, 3=strong
- Make the graph connected — no isolated nodes
- Output valid JSON only, no markdown."""

        text = generate(prompt)
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r'^```\w*\n?', '', text)
            text = re.sub(r'\n?```$', '', text)
        data = json.loads(text)

        nodes = data.get("nodes", [])
        edges = [(e["from"], e["to"], e.get("strength", 1)) for e in data.get("edges", [])]
        return nodes, edges

    except Exception:
        return [], []


# --- Build full graph ---
all_nodes, all_edges = build_knowledge_graph()

if not all_nodes:
    st.info("No videos indexed yet. Process some videos first to see the knowledge graph.")
    st.stop()

st.markdown(f"""
<div style="text-align:center; margin-bottom:1rem;">
    <span style="background:{t['code_bg']}; border:1px solid rgba(102,126,234,0.3);
        color:{t['accent']}; padding:4px 14px; border-radius:20px; font-size:0.85rem;">
        {len(all_nodes)} topics extracted from indexed videos
    </span>
</div>
""", unsafe_allow_html=True)

# --- Layout --- dynamically assign group centers based on discovered groups
unique_groups = list(set(n["group"] for n in all_nodes))
group_centers = {}
angle_step = 2 * math.pi / max(len(unique_groups), 1)
for i, group in enumerate(unique_groups):
    angle = i * angle_step
    group_centers[group] = (3 * math.cos(angle), 3 * math.sin(angle))

random.seed(42)
positions = {}

for node in all_nodes:
    cx, cy = group_centers.get(node["group"], (0, 0))
    positions[node["id"]] = (
        cx + random.uniform(-1.5, 1.5),
        cy + random.uniform(-1.5, 1.5)
    )

# Force-directed relaxation
all_node_ids = {n["id"] for n in all_nodes}
for _ in range(50):
    for edge in all_edges:
        a, b, w = edge
        if a in positions and b in positions:
            ax, ay = positions[a]
            bx, by = positions[b]
            dx, dy = bx - ax, by - ay
            dist = math.sqrt(dx*dx + dy*dy) + 0.01
            target = 1.5
            force = (dist - target) * 0.05
            fx, fy = (dx/dist) * force, (dy/dist) * force
            positions[a] = (ax + fx, ay + fy)
            positions[b] = (bx - fx, by - fy)

# Dynamically assign colors to groups
_palette = ["#f093fb", "#68d391", "#ea6666", "#667eea", "#f0c83c", "#c084fc", "#ffa500", "#38bdf8", "#fb923c", "#a78bfa"]
group_colors = {g: _palette[i % len(_palette)] for i, g in enumerate(unique_groups)}

# --- Build plotly figure ---
fig = go.Figure()

# Draw edges
for edge in all_edges:
    a, b, w = edge
    if a in positions and b in positions:
        ax, ay = positions[a]
        bx, by = positions[b]
        edge_color = f'rgba(102,126,234,{0.1 + w*0.1})'
        fig.add_trace(go.Scatter(
            x=[ax, bx, None], y=[ay, by, None],
            mode='lines',
            line=dict(width=w * 0.8, color=edge_color),
            hoverinfo='none', showlegend=False
        ))

# Draw nodes
for node in all_nodes:
    if node["id"] not in positions:
        continue
    x, y = positions[node["id"]]
    color = group_colors.get(node["group"], "#667eea")
    fig.add_trace(go.Scatter(
        x=[x], y=[y],
        mode='markers+text',
        marker=dict(size=node["size"], color=color, opacity=0.85,
                    line=dict(width=2, color='rgba(255,255,255,0.2)')),
        text=[node["id"]],
        textposition="top center",
        textfont=dict(size=10, color=t['text_primary']),
        hovertext=f"{node['id']} ({node['group'].upper()})",
        hoverinfo='text',
        showlegend=False
    ))

fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    height=650,
    margin=dict(l=20, r=20, t=20, b=20),
    hoverlabel=dict(bgcolor=t['bg_card'], font_color=t['text_heading'], bordercolor=t['accent'])
)

st.plotly_chart(fig, use_container_width=True)

# Legend
legend_html = ""
for group, color in group_colors.items():
    legend_html += f'<span style="display:inline-flex; align-items:center; margin:0 0.8rem;"><span style="width:12px; height:12px; border-radius:50%; background:{color}; display:inline-block; margin-right:6px;"></span><span style="color:{t["text_explanation"]}; font-size:0.85rem;">{group.upper()}</span></span>'

st.markdown(f'<div style="text-align:center; margin:1rem 0;">{legend_html}</div>', unsafe_allow_html=True)

st.markdown(f"""
<div style="text-align:center; color:{t['text_muted']}; font-size:0.8rem; margin-top:1rem;">
    Node size represents topic importance. Edge thickness represents connection strength.
    <br>Topics that are taught together or build on each other are connected.
    <br>All topics are dynamically extracted from your indexed videos using AI.
</div>
""", unsafe_allow_html=True)

# Refresh button
if st.button("Refresh Graph", use_container_width=True):
    st.cache_data.clear()
    st.rerun()
