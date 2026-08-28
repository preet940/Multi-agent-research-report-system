"""
Episodic memory for the worker agent: a vector DB cache of past
(sub_question -> summary) pairs. Before researching a question, check
if something SEMANTICALLY SIMILAR (not exact-match) was already
researched -- if so, reuse it instead of re-searching and re-summarizing.

Uses Chroma (local, no server needed) with its default embedding
function (sentence-transformers all-MiniLM-L6-v2, runs locally, no
API key required) -- keeps this decoupled from Groq/OpenAI entirely.

This is a real, if small, example of:
  - retrieval internals: embeddings + similarity search + a distance
    threshold, not just calling search_web every time
  - episodic memory: knowledge persists ACROSS runs, not just within
    one orchestrator call (unlike worker_outputs, which resets every run)
  - cost optimization: a cache hit skips BOTH the search_web call and
    the LLM summarization call -- the cheapest possible research step
    is the one you don't have to do
"""

import chromadb

# Persistent = survives between runs, stored on disk in ./chroma_data
client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(name="research_cache")

# Chroma returns DISTANCE, not similarity -- lower distance = more similar.
# This threshold is empirical: start conservative (small = stricter match)
# and loosen it if you're getting too few cache hits on genuinely similar
# questions. Tune by inspecting distances Chroma returns during testing.
DISTANCE_THRESHOLD = 0.25


def check_cache(sub_question: str) -> dict:
    """
    Returns {"hit": True, "summary": ..., "sources": [...], "distance": ...}
    if a similar-enough past question exists, else {"hit": False}.
    """
    count = collection.count()
    if count == 0:
        return {"hit": False}

    results = collection.query(
        query_texts=[sub_question],
        n_results=1,
    )

    if not results["ids"][0]:
        return {"hit": False}

    distance = results["distances"][0][0]
    if distance > DISTANCE_THRESHOLD:
        return {"hit": False, "closest_distance": distance}  # too dissimilar

    metadata = results["metadatas"][0][0]
    return {
        "hit": True,
        "matched_question": results["documents"][0][0],
        "summary": metadata["summary"],
        "sources": metadata["sources"].split("|"),
        "distance": distance,
    }


def store_result(sub_question: str, summary: str, sources: list):
    """Called after a real (non-cached) research step, so next time a
    similar question comes in, it can be reused."""
    collection.add(
        documents=[sub_question],
        metadatas=[{"summary": summary, "sources": "|".join(sources)}],
        ids=[f"q_{collection.count()}"],
    )