"""The local model judges results against the request and can ask for better queries."""
import json

from .config import JUDGE_CHUNK, JUDGE_ENABLED, MAX_JUDGED

from .planning import JUDGE_SCHEMA, REFINE_SCHEMA, llm_json


async def judge(llm_url, request, queries, items):
    """items: [(source label, result)]. The local model sorts them into good / partial / irrelevant by reading the
    user's request. Returns {"good": set, "partial": set, "unknown": set} of positions in `items`;
    positions of a chunk the model failed to answer land in "unknown"."""
    from datetime import date
    good, partial, unknown = set(), set(), set()
    for start in range(0, len(items), JUDGE_CHUNK):
        chunk = items[start:start + JUDGE_CHUNK]
        lines = "\n".join(f"{i}. [{lab}] {it['title'][:120]} | {it['snippet'][:160]}"
                          + (f" | {it['date']}" if it["date"] else "") for i, (lab, it) in enumerate(chunk))
        prompt = (
            f"Today is {date.today():%d %B %Y}. You filter web search results for a voice assistant.\n"
            f"User request: {request}\nSearches run: {' | '.join(queries)}\n\n"
            "Below are numbered results, each as '[source] title | snippet'.\n"
            "- good: clearly what the user asked for: the right topic, the right kind of page (an actual posting, "
            "article or page about the thing, not a generic category page, homepage or login page), and the right "
            "place, country and language if the user specified one. A result in another country is NOT good.\n"
            "- partial: related but weaker (adjacent topic, only partly matching, or unclear).\n"
            "Everything else is irrelevant: wrong field, wrong country, wrong kind of page, spam.\n"
            'Reply with JSON {"good": [numbers], "partial": [numbers]}. Be selective with good, but do not leave '
            "both lists empty when something is related.\n\nResults:\n" + lines)
        out = await llm_json(llm_url, prompt, JUDGE_SCHEMA, "judge", 400, 120)
        if not out:
            unknown.update(range(start, start + len(chunk)))
            continue
        for key, bucket in (("good", good), ("partial", partial)):
            for i in out.get(key) or []:
                if isinstance(i, int) and 0 <= i < len(chunk):
                    bucket.add(start + i)
    return {"good": good, "partial": partial - good, "unknown": unknown}


async def refine_queries(llm_url, request, tried, samples):
    """When results are weak the model writes new queries instead of repeating the old ones."""
    prompt = (
        "A web search for a voice assistant found too few good results.\n"
        f"User request: {request}\nQueries already tried: {' | '.join(tried)}\n"
        "Titles found so far (many are off-topic):\n" + ("\n".join("- " + s[:90] for s in samples[:10]) or "(none)") +
        "\n\nWrite 3 NEW short keyword queries likely to find better results: each 2 to 5 words, SHORTER and more "
        "focused than the old ones (long queries match nothing: every word must be present). If the old queries "
        "stacked several skills, split them into one query per skill. Use synonyms, the other language (French, "
        "English or Arabic when relevant), more specific or more general wording. No site names, no "
        'site: operators, and do not repeat the old queries. Reply with JSON {"queries": [...]}.')
    out = await llm_json(llm_url, prompt, REFINE_SCHEMA, "refine", 200, 90)
    return [q.strip() for q in (out or {}).get("queries", []) if isinstance(q, str) and q.strip()][:3]
