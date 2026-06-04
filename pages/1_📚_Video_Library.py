import streamlit as st
import chromadb
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from theme import get_theme, get_common_css
CHROMA_PATH = "./chroma_db"

st.set_page_config(page_title="Video Library | Videx", page_icon="📚", layout="wide")

t = get_theme()

st.markdown(get_common_css(t) + f"""
<style>
    .page-header {{
        animation: fadeIn 0.8s ease;
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(-15px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .video-item {{
        background: {t['bg_card']};
        border: 1px solid {t['border_card']};
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        margin: 0.8rem 0;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        gap: 1.2rem;
        animation: slideUp 0.5s ease;
    }}

    @keyframes slideUp {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .video-item:hover {{
        border-color: rgba(102,126,234,0.3);
        transform: translateX(6px);
        box-shadow: 0 4px 20px rgba(102,126,234,0.1);
    }}

    .video-number {{
        background: linear-gradient(135deg, {t['accent']}, {t['accent2']});
        color: white;
        font-size: 1.2rem;
        font-weight: 700;
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }}

    .video-info h4 {{
        color: {t['text_primary']} !important;
        margin: 0 0 0.3rem;
        font-size: 1rem;
        font-weight: 600;
    }}

    .video-info p {{
        color: {t['text_muted']} !important;
        margin: 0;
        font-size: 0.85rem;
    }}

    .video-badge {{
        background: {t['code_bg']};
        border: 1px solid rgba(102,126,234,0.25);
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        color: {t['text_heading']};
        display: inline-block;
        margin-right: 0.3rem;
    }}

    .search-stats {{
        text-align: center;
        color: {t['text_muted']};
        font-size: 0.85rem;
        margin: 1rem 0;
    }}

    .delete-banner {{
        background: {t['success_bg']};
        border: 1px solid {t['success_border']};
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        color: {t['success']};
        margin: 0.5rem 0;
    }}

</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 1rem 0;">
        <div style="font-size:2rem;">🎓</div>
        <div style="font-size:1.1rem; font-weight:700; color:{t['text_heading']};">VIDEX</div>
        <div style="font-size:0.7rem; color:{t['text_muted']}; letter-spacing:2px;">VIDEO LIBRARY</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <div class="page-title">Video Library</div>
    <div class="page-subtitle">Browse all indexed videos in your knowledge base</div>
</div>
""", unsafe_allow_html=True)

# Build video list from ALL sources: JSON files + ChromaDB (YouTube/uploaded)
jsons_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "jsons")
videos = []
seen_titles = set()

# Source 1: JSON files (pre-processed videos)
if os.path.isdir(jsons_dir):
    json_files = sorted(os.listdir(jsons_dir))
    for json_file in json_files:
        with open(f"{jsons_dir}/{json_file}") as f:
            content = json.load(f)
        chunks = content['chunks']
        if chunks:
            number = chunks[0]['number']
            title = chunks[0]['title']
            total_chunks = len(chunks)
            duration_s = chunks[-1]['end']
            duration_m = int(duration_s) // 60
            videos.append({
                "number": number, "title": title,
                "chunks": total_chunks, "duration": duration_m,
                "source": "local",
                "full_text_preview": chunks[0]['text'][:100] + "..."
            })
            seen_titles.add(title)

# Source 2: YouTube/uploaded videos from ChromaDB
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection("video_chunks")
    all_data = collection.get(include=["documents", "metadatas"])

    yt_videos = {}
    for i, meta in enumerate(all_data["metadatas"]):
        src = meta.get("source")
        if src in ("youtube", "upload"):
            title = meta["title"]
            if title not in yt_videos:
                yt_videos[title] = {"chunks": 0, "max_end": 0, "first_text": "", "number": meta.get("number", "YT"), "ids": [], "source": src}
            yt_videos[title]["chunks"] += 1
            yt_videos[title]["ids"].append(all_data["ids"][i])
            end = meta.get("end", 0)
            if end > yt_videos[title]["max_end"]:
                yt_videos[title]["max_end"] = end
            if not yt_videos[title]["first_text"]:
                yt_videos[title]["first_text"] = all_data["documents"][i][:100] + "..."

    for title, info in yt_videos.items():
        if title not in seen_titles:
            videos.append({
                "number": info["number"],
                "title": title,
                "chunks": info["chunks"],
                "duration": int(info["max_end"]) // 60,
                "source": info["source"],
                "full_text_preview": info["first_text"],
                "chroma_ids": info["ids"],
            })
except Exception:
    pass


def delete_video(video):
    """Delete a video from all sources (JSON file + ChromaDB)."""
    deleted_from = []

    # Delete from ChromaDB — match by title across all chunks
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        col = client.get_collection("video_chunks")

        if video.get("chroma_ids"):
            col.delete(ids=video["chroma_ids"])
            deleted_from.append(f"ChromaDB ({len(video['chroma_ids'])} chunks)")
        else:
            all_items = col.get(include=["metadatas"])
            ids_to_delete = [
                all_items["ids"][i]
                for i, meta in enumerate(all_items["metadatas"])
                if meta.get("title") == video["title"]
            ]
            if ids_to_delete:
                col.delete(ids=ids_to_delete)
                deleted_from.append(f"ChromaDB ({len(ids_to_delete)} chunks)")
    except Exception:
        pass

    # Delete JSON file if local source
    if video.get("source") == "local":
        jsons_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "jsons")
        if os.path.isdir(jsons_path):
            for fname in os.listdir(jsons_path):
                fpath = os.path.join(jsons_path, fname)
                try:
                    with open(fpath) as f:
                        content = json.load(f)
                    if content.get("chunks") and content["chunks"][0].get("title") == video["title"]:
                        os.remove(fpath)
                        deleted_from.append(f"JSON file ({fname})")
                        break
                except Exception:
                    continue

    return deleted_from


# Search filter
search = st.text_input("Search videos...", placeholder="Type to filter videos...")

if search:
    videos = [v for v in videos if search.lower() in v['title'].lower()]

st.markdown(f'<div class="search-stats">{len(videos)} videos found</div>', unsafe_allow_html=True)

# Handle deletion
if "delete_confirm" not in st.session_state:
    st.session_state.delete_confirm = None

for idx, v in enumerate(videos):
    source_badge = ""
    if v.get("source") == "youtube":
        source_badge = f'<span class="video-badge" style="background:{t["error_bg"]};border-color:{t["error_border"]};color:{t["error"]};">YouTube</span>'
    elif v.get("source") == "upload":
        source_badge = f'<span class="video-badge" style="background:{t["warning_bg"]};border-color:{t["warning_border"]};color:{t["warning"]};">Uploaded</span>'
    elif v.get("source") == "local":
        source_badge = f'<span class="video-badge" style="background:{t["success_bg"]};border-color:{t["success_border"]};color:{t["success"]};">Local</span>'

    st.markdown(f"""
    <div class="video-item">
        <div class="video-number">{v['number']}</div>
        <div class="video-info" style="flex:1;">
            <h4>{v['title']}</h4>
            <p>{v['full_text_preview']}</p>
            <div style="margin-top: 0.5rem;">
                {source_badge}
                <span class="video-badge">{v['chunks']} segments</span>
                <span class="video-badge">{v['duration']} min</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.delete_confirm == idx:
        st.warning(f"Are you sure you want to delete **{v['title']}**? This will remove all chunks and data permanently.")
        col_yes, col_no, _ = st.columns([1, 1, 6])
        with col_yes:
            if st.button("Yes, delete", key=f"confirm_{idx}", type="primary"):
                result = delete_video(v)
                st.session_state.delete_confirm = None
                st.session_state.delete_success = f"Deleted **{v['title']}** from: {', '.join(result)}"
                st.rerun()
        with col_no:
            if st.button("Cancel", key=f"cancel_{idx}"):
                st.session_state.delete_confirm = None
                st.rerun()
    else:
        if st.button("Delete", key=f"delete_{idx}", type="secondary"):
            st.session_state.delete_confirm = idx
            st.rerun()

if st.session_state.get("delete_success"):
    st.success(st.session_state.delete_success)
    st.session_state.delete_success = None
