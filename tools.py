"""
Tools are plain functions. They know nothing about LLMs or agents.
Keeping this separation means:
  - you can unit test tools without ever calling the model
  - swapping a tool implementation never touches agent logic
"""

import os

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")


def search_web(query: str, max_results: int = 3) -> list[dict]:
    """
    Returns a list of {title, url, content} dicts.
    Uses Tavily if you have a key (free tier: tavily.com), else falls
    back to a stub so the pipeline still runs end-to-end for testing.
    """
    if TAVILY_API_KEY:
        import requests
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": max_results,
            },
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return [
            {"title": r["title"], "url": r["url"], "content": r["content"]}
            for r in results
        ]

    # --- Stub fallback: lets you test the ORCHESTRATION before wiring
    # a real API key. This is a good habit generally: mock the tool,
    # prove the control flow works, THEN plug in the real thing.
    return [{
        "title": f"[STUB] result for: {query}",
        "url": "https://example.com",
        "content": f"No TAVILY_API_KEY set. This is placeholder content "
                    f"standing in for real search results about '{query}'.",
    }]