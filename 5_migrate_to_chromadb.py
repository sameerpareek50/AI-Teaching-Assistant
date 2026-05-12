# Step 5: Migrate embeddings from JSON files to ChromaDB vector database
# Uses ChromaDB's default embedding model (all-MiniLM-L6-v2) which runs locally
# Merges consecutive small chunks into larger chunks for better semantic context
# No API keys or external services needed for embedding

import json
import os
import sys
import chromadb

CHROMA_PATH = "./chroma_db"
MERGE_SIZE = 5  # Merge every N consecutive whisper segments into one semantic chunk


def log(msg):
    print(msg)
    sys.stdout.flush()


def merge_chunks(chunks, merge_size=MERGE_SIZE):
    """Merge consecutive small whisper segments into larger semantic chunks.
    Whisper outputs very short segments (2-5 seconds). Merging gives better context
    for embedding and retrieval — this is a chunking strategy optimization."""
    merged = []
    for i in range(0, len(chunks), merge_size):
        group = chunks[i:i + merge_size]
        merged.append({
            "number": group[0]["number"],
            "title": group[0]["title"],
            "start": group[0]["start"],
            "end": group[-1]["end"],
            "text": " ".join(c["text"].strip() for c in group)
        })
    return merged


def migrate():
    # ChromaDB with default embedding function (all-MiniLM-L6-v2, runs locally)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection("video_chunks")
    except:
        pass

    # Using default embedding function — ChromaDB will auto-embed documents
    collection = client.create_collection(
        name="video_chunks",
        metadata={"description": "Video transcript chunks for AI Teaching Assistant"}
    )

    jsons_dir = "jsons"
    json_files = sorted(os.listdir(jsons_dir))

    total_original = 0
    total_merged = 0

    for json_file in json_files:
        with open(f"{jsons_dir}/{json_file}") as f:
            content = json.load(f)

        original_count = len(content['chunks'])
        merged = merge_chunks(content['chunks'])
        total_original += original_count
        total_merged += len(merged)
        log(f"Processing: {json_file} ({original_count} -> {len(merged)} chunks)")

        ids = []
        texts = []
        metadatas = []

        for i, chunk in enumerate(merged):
            chunk_id = total_merged - len(merged) + i
            ids.append(f"chunk_{chunk_id}")
            texts.append(chunk['text'])
            metadatas.append({
                "number": str(chunk['number']),
                "title": chunk['title'],
                "start": float(chunk['start']),
                "end": float(chunk['end']),
                "chunk_id": chunk_id
            })

        # ChromaDB auto-embeds the documents using its default model
        collection.add(ids=ids, documents=texts, metadatas=metadatas)
        log(f"  Added {len(ids)} chunks to ChromaDB")

    log(f"\n{'='*50}")
    log(f"Migration complete!")
    log(f"Original whisper segments: {total_original}")
    log(f"Merged semantic chunks:    {total_merged}")
    log(f"Compression ratio:         {total_original/total_merged:.1f}x")
    log(f"ChromaDB path:             {CHROMA_PATH}")
    log(f"Collection count:          {collection.count()}")
    log(f"{'='*50}")


if __name__ == "__main__":
    migrate()
