# Step 6: Evaluation & Benchmarking of the RAG Pipeline
# Tests retrieval quality across different top_k values
# Measures hit rate, MRR (Mean Reciprocal Rank), topic-wise accuracy
# Uses ChromaDB's built-in embedding for retrieval, Gemini for LLM

import chromadb
import json
import sys
import time
from datetime import datetime
from google import genai

GEMINI_API_KEY = "AIzaSyAgGOmuCxNFv_5LrhW9bhzVgcRJZ4Hfzkg"
LLM_MODEL = "gemini-2.5-flash"
CHROMA_PATH = "./chroma_db"

client_genai = genai.Client(api_key=GEMINI_API_KEY)

# 20 test questions with ground-truth expected video numbers
TEST_QUESTIONS = [
    {"question": "How to install VS Code?", "expected_videos": ["1"], "topic": "Setup"},
    {"question": "What is HTML?", "expected_videos": ["2", "3"], "topic": "HTML"},
    {"question": "How to create headings in HTML?", "expected_videos": ["4"], "topic": "HTML"},
    {"question": "How to add images in HTML?", "expected_videos": ["5"], "topic": "HTML"},
    {"question": "What is SEO?", "expected_videos": ["6"], "topic": "HTML"},
    {"question": "How to create forms in HTML?", "expected_videos": ["7"], "topic": "HTML"},
    {"question": "What are inline and block elements?", "expected_videos": ["8"], "topic": "HTML"},
    {"question": "What are id and classes in HTML?", "expected_videos": ["9"], "topic": "HTML"},
    {"question": "How to add video in HTML?", "expected_videos": ["10"], "topic": "HTML"},
    {"question": "What are semantic tags?", "expected_videos": ["11"], "topic": "HTML"},
    {"question": "What is CSS and how to use it?", "expected_videos": ["14"], "topic": "CSS"},
    {"question": "What are CSS selectors?", "expected_videos": ["17"], "topic": "CSS"},
    {"question": "What is the CSS box model?", "expected_videos": ["18"], "topic": "CSS"},
    {"question": "Difference between inline and external CSS?", "expected_videos": ["15"], "topic": "CSS"},
    {"question": "How to create a media player in HTML?", "expected_videos": ["12"], "topic": "HTML"},
    {"question": "What are HTML entities?", "expected_videos": ["13"], "topic": "HTML"},
    {"question": "What is margin and padding?", "expected_videos": ["18"], "topic": "CSS"},
    {"question": "How websites work?", "expected_videos": ["1"], "topic": "Setup"},
    {"question": "How to create tables in HTML?", "expected_videos": ["5"], "topic": "HTML"},
    {"question": "What are internal stylesheets?", "expected_videos": ["15"], "topic": "CSS"},
    {"question": "How does CSS Flexbox work?", "expected_videos": ["19"], "topic": "CSS"},
    {"question": "What is CSS Grid?", "expected_videos": ["20"], "topic": "CSS"},
    {"question": "How to make a responsive website?", "expected_videos": ["21"], "topic": "CSS"},
    {"question": "How to create CSS animations?", "expected_videos": ["22"], "topic": "CSS"},
    {"question": "What is JavaScript?", "expected_videos": ["25"], "topic": "JavaScript"},
    {"question": "What are JavaScript variables?", "expected_videos": ["26"], "topic": "JavaScript"},
    {"question": "How do JavaScript arrays work?", "expected_videos": ["28"], "topic": "JavaScript"},
    {"question": "What are arrow functions in JavaScript?", "expected_videos": ["29"], "topic": "JavaScript"},
    {"question": "What is DOM manipulation?", "expected_videos": ["31"], "topic": "JavaScript"},
    {"question": "How do event listeners work in JavaScript?", "expected_videos": ["32"], "topic": "JavaScript"},
    {"question": "What are JavaScript Promises?", "expected_videos": ["35"], "topic": "JavaScript"},
    {"question": "How to use fetch API?", "expected_videos": ["36"], "topic": "JavaScript"},
    {"question": "What is localStorage in JavaScript?", "expected_videos": ["37"], "topic": "JavaScript"},
    {"question": "What are ES6 features?", "expected_videos": ["38"], "topic": "JavaScript"},
]


def log(msg):
    print(msg)
    sys.stdout.flush()


def query_chromadb(collection, query_text, top_k=5):
    return collection.query(
        query_texts=[query_text],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )


def inference(prompt):
    response = client_genai.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )
    return response.text


def evaluate_retrieval(collection, top_k_values=[3, 5, 7, 10]):
    """Evaluate retrieval accuracy for different top_k values"""
    log("=" * 70)
    log("RETRIEVAL EVALUATION")
    log("=" * 70)

    results_summary = []

    for top_k in top_k_values:
        hits = 0
        total = len(TEST_QUESTIONS)
        mrr_sum = 0
        latencies = []

        for tq in TEST_QUESTIONS:
            start = time.time()
            results = query_chromadb(collection, tq["question"], top_k=top_k)
            latency = time.time() - start
            latencies.append(latency)

            retrieved_videos = [m['number'] for m in results['metadatas'][0]]
            hit = False
            for rank, vid in enumerate(retrieved_videos):
                if vid in tq["expected_videos"]:
                    if not hit:
                        mrr_sum += 1.0 / (rank + 1)
                        hit = True
            if hit:
                hits += 1

        accuracy = hits / total
        mrr = mrr_sum / total
        avg_latency = sum(latencies) / len(latencies)

        result = {
            "top_k": top_k,
            "accuracy": accuracy,
            "mrr": mrr,
            "avg_latency_ms": avg_latency * 1000,
            "hits": hits,
            "total": total
        }
        results_summary.append(result)

        log(f"\nTop-K = {top_k}:")
        log(f"  Hit Rate (Accuracy): {accuracy:.1%} ({hits}/{total})")
        log(f"  Mean Reciprocal Rank: {mrr:.3f}")
        log(f"  Avg Retrieval Latency: {avg_latency*1000:.0f}ms")

    return results_summary


def evaluate_by_topic(collection, top_k=5):
    """Break down accuracy by topic"""
    log("\n" + "=" * 70)
    log("TOPIC-WISE BREAKDOWN (Top-K = 5)")
    log("=" * 70)

    topic_results = {}
    for tq in TEST_QUESTIONS:
        topic = tq["topic"]
        if topic not in topic_results:
            topic_results[topic] = {"hits": 0, "total": 0}

        results = query_chromadb(collection, tq["question"], top_k=top_k)
        retrieved_videos = [m['number'] for m in results['metadatas'][0]]
        hit = any(vid in tq["expected_videos"] for vid in retrieved_videos)
        topic_results[topic]["total"] += 1
        if hit:
            topic_results[topic]["hits"] += 1

    for topic, data in topic_results.items():
        acc = data["hits"] / data["total"]
        log(f"  {topic}: {acc:.1%} ({data['hits']}/{data['total']})")

    return topic_results


def evaluate_end_to_end(collection, num_samples=5):
    """End-to-end evaluation: retrieval + LLM response quality"""
    log("\n" + "=" * 70)
    log(f"END-TO-END EVALUATION (first {num_samples} questions)")
    log("=" * 70)

    e2e_results = []
    for tq in TEST_QUESTIONS[:num_samples]:
        results = query_chromadb(collection, tq["question"], top_k=5)

        chunks_info = []
        for i in range(len(results['ids'][0])):
            meta = results['metadatas'][0][i]
            chunks_info.append({
                "title": meta['title'], "number": meta['number'],
                "start": meta['start'], "end": meta['end'],
                "text": results['documents'][0][i]
            })

        prompt = f'''You are an AI video assistant. Here are video transcript chunks containing video title, video number, start time in seconds, end time in seconds, and the text at that time:

{json.dumps(chunks_info)}
---------------------------------
"{tq['question']}"
User asked this question related to the video content. Answer in a helpful, human way — explain where and how much content is taught in which video (including video number and timestamp) and guide the user to the right video. If the question is unrelated to the indexed video content, let them know politely.'''

        start = time.time()
        response = inference(prompt)
        latency = time.time() - start

        mentions_correct = any(
            f"video {v}" in response.lower() or
            f"video number {v}" in response.lower() or
            f"video #{v}" in response.lower()
            for v in tq["expected_videos"]
        )

        log(f"\nQ: {tq['question']}")
        log(f"  Expected Videos: {tq['expected_videos']}")
        log(f"  Mentions Correct Video: {'Yes' if mentions_correct else 'No'}")
        log(f"  LLM Latency: {latency:.1f}s")
        log(f"  Response Preview: {response[:150]}...")

        e2e_results.append({
            "question": tq["question"],
            "expected_videos": tq["expected_videos"],
            "mentions_correct": mentions_correct,
            "latency": latency,
            "response": response
        })

        time.sleep(2)  # Rate limit protection for Gemini LLM

    correct = sum(1 for r in e2e_results if r["mentions_correct"])
    log(f"\nEnd-to-End Accuracy: {correct}/{len(e2e_results)} ({correct/len(e2e_results):.1%})")
    return e2e_results


def save_results(retrieval_results, topic_results, e2e_results):
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "embedding_model": "all-MiniLM-L6-v2 (ChromaDB default)",
            "llm_model": LLM_MODEL,
            "test_questions": len(TEST_QUESTIONS),
            "chunk_merge_size": 5
        },
        "retrieval_evaluation": retrieval_results,
        "topic_evaluation": topic_results,
        "e2e_evaluation": [
            {k: v for k, v in r.items() if k != "response"}
            for r in e2e_results
        ]
    }

    with open("evaluation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    log(f"\nResults saved to evaluation_results.json")


if __name__ == "__main__":
    log("Loading ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection("video_chunks")
    log(f"Loaded {collection.count()} chunks\n")

    retrieval_results = evaluate_retrieval(collection, top_k_values=[3, 5, 7, 10])
    topic_results = evaluate_by_topic(collection)
    e2e_results = evaluate_end_to_end(collection, num_samples=5)
    save_results(retrieval_results, topic_results, e2e_results)

    log("\n" + "=" * 70)
    log("EVALUATION COMPLETE")
    log("=" * 70)
