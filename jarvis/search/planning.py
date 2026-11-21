"""Query planning: the local model turns a sentence into sites, queries and limits."""
import json
import re

import httpx

from .config import JUDGE_CHUNK, MAX_LIMIT
from .memory import memory_line
from .sites import detect_sites
from .utils import JOBS_RE, LIMIT_RE, NEWS_WORDS, RETRY_RE, fallback_job_query, fold


async def llm_json(llm_url, prompt, schema, name, max_tokens=300, timeout=90):
    """Asks the local model for JSON that follows `schema`. Constrained output first; looser formats if the
    server rejects it. Returns the parsed dict, or None when every attempt failed."""
    base = {"messages": [{"role": "user", "content": prompt}], "stream": False, "temperature": 0,
            "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}}
    formats = [{"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}},
               {"type": "json_object"}, None]
    for fmt in formats:
        payload = dict(base, **({"response_format": fmt} if fmt else {}))
        try:
            async with httpx.AsyncClient(timeout=timeout) as c:
                r = await c.post(llm_url, json=payload)
            if r.status_code >= 400:
                continue
            txt = r.json()["choices"][0]["message"].get("content") or ""
            txt = re.sub(r"</think>.*?</think>", "", txt, flags=re.S)   # some models emit a thinking block before the JSON
            outer = re.search(r"\{.*\}", txt, re.S)         # outermost object first...
            cands = ([outer.group(0)] if outer else [])     # ...then small brace-free ones as a salvage
            cands += re.findall(r"\{[^{}]*\}", txt, re.S)
            for c in cands:
                try:
                    return json.loads(c)
                except ValueError:
                    continue
            raise ValueError("no JSON object found in the answer")
        except Exception as e:
            print(f"[{name} attempt failed]", repr(e))
    return None


PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "search": {"type": "boolean"},
        "intent": {"enum": ["news", "jobs", "general"]},
        "queries": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "sites": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
        "suggest": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
        "recency": {"enum": ["none", "day", "week", "month", "year"]},
        "limit": {"type": "integer", "minimum": 0, "maximum": MAX_LIMIT},
        "limit_scope": {"enum": ["per_site", "total"]},
        "display": {"enum": ["answer", "links"]},
        "refs": {"type": "array", "items": {"type": "integer", "minimum": 1}, "maxItems": 3},
    },
    "required": ["search", "intent", "queries", "sites", "suggest", "recency", "limit", "limit_scope", "display",
                 "refs"],
}


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"good": {"type": "array", "items": {"type": "integer", "minimum": 0}, "maxItems": JUDGE_CHUNK},
                   "partial": {"type": "array", "items": {"type": "integer", "minimum": 0}, "maxItems": JUDGE_CHUNK}},
    "required": ["good", "partial"],
}


REFINE_SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
    "required": ["queries"],
}


PICK_SCHEMA = {
    "type": "object",
    "properties": {"pick": {"type": "integer", "minimum": 0}, "why": {"type": "string"}},
    "required": ["pick", "why"],
}


def norm_plan(raw, user_text):
    intent = raw.get("intent") if raw.get("intent") in ("news", "jobs", "general") else "general"
    queries = [re.sub(r"\bsite:\S+", " ", q).strip() for q in (raw.get("queries") or []) if isinstance(q, str)]
    queries = [re.sub(r"\s+", " ", q) for q in queries if q.strip()][:3]
    try:
        limit = max(0, min(MAX_LIMIT, int(raw.get("limit") or 0)))
    except (TypeError, ValueError):
        limit = 0
    refs = [int(x) for x in (raw.get("refs") or []) if isinstance(x, (int, float)) and x >= 1][:3]
    return {"search": bool(raw.get("search")), "intent": intent, "queries": queries,
            "sites": [str(s) for s in (raw.get("sites") or [])][:10],
            "suggest": [str(s) for s in (raw.get("suggest") or [])][:8],
            "recency": raw.get("recency") if raw.get("recency") in ("day", "week", "month", "year") else None,
            "limit": limit, "limit_scope": "total" if raw.get("limit_scope") == "total" else "per_site",
            "display": raw.get("display") if raw.get("display") in ("answer", "links")
            else ("links" if intent == "jobs" else "answer"),
            "refs": refs}


def heuristic_plan(text, last_plan):
    """Used when the planner call fails. Keyword rules, same output shape."""
    f = fold(text)
    news, jobs = bool(NEWS_WORDS.search(f)), bool(JOBS_RE.search(f))
    if RETRY_RE.search(f) and last_plan and last_plan.get("queries"):
        intent, queries = last_plan.get("intent", "general"), list(last_plan["queries"])
    elif jobs:
        intent, queries = "jobs", [fallback_job_query(text)]
    elif news:
        intent, queries = "news", ["top news today", "breaking news headlines"]
    else:
        intent, queries = "general", [text]
    m = LIMIT_RE.search(f)
    return norm_plan({"search": True, "intent": intent, "queries": queries,
                      "sites": [s["name"] for s in detect_sites(text)], "suggest": [],
                      "recency": "day" if news else "none",
                      "limit": int(m.group(1) or m.group(2)) if m else 0,
                      "limit_scope": "total" if re.search(r"\b(total|au total|in total|overall)\b", f) else "per_site",
                      "display": "links" if (jobs or re.search(r"\b(links?|liens?|list|liste)\b", f)) else "answer",
                      "refs": []}, text)


async def make_plan(llm_url, user_text, history, last_plan, last_results):
    from datetime import date
    recent = [m for m in history[1:-1] if m["role"] in ("user", "assistant")][-4:]
    ctx = "\n".join(f'{m["role"]}: {m["content"][:300]}' for m in recent) or "(none)"
    shown = "\n".join(f"#{r['n']} {r['title'][:80]} ({r['source']})" for r in last_results[:30]) or "(none)"
    prev = json.dumps({k: last_plan[k] for k in ("intent", "queries", "sites", "recency")}) if last_plan else "(none)"
    prompt = (
        f"Today is {date.today():%A %d %B %Y}. You plan web searches for a voice assistant.\n"
        "Fill the JSON fields for the user's last message:\n"
        "- search: true if it needs the web (news, current events, jobs, prices, scores, weather, recent facts, "
        "products, people, research) or if the user wants something played, watched or opened (music, a playlist, "
        "a video, an album, a page, a product) — that always needs a real link. false for greetings, chit-chat, "
        "coding and timeless knowledge.\n"
        "- intent: news, jobs (jobs, internships, stages, PFE, recruitment) or general.\n"
        "- queries: 1 to 4 SHORT keyword queries, each 2 to 5 words. If the message names several skills, subjects "
        "or fields, give each one its own query instead of stacking them ('data science, AI, data analyst, BI' = "
        "separate queries, one per field). Keep the place in the query when the user names one ('Maroc', "
        "'Toulouse'). Write each query in the language of the country or content sought (French for Morocco, "
        "English otherwise). Long queries find almost nothing: search engines match every word. "
        "NEVER put site names or 'site:' in queries. Resolve words like 'he' or 'that' from the conversation.\n"
        "- sites: only websites the user names (for example linkedin, indeed, reddit, a domain). Empty list otherwise.\n"
        "- suggest: only when the user wants listings (jobs, internships, products, events...) and names no site. "
        "First reason about WHO actually publishes this kind of listing (the organizations themselves, specialized "
        "boards for that country/field, institutions, associations) — not just the most famous generic sites — then "
        "name up to 6 real websites you are confident exist. Names only. Otherwise [].\n"
        "- recency: none unless the user wants recent results: day for today/latest news, week, month or year.\n"
        "- limit: how many results the user asks for, 0 if they do not say. limit_scope: per_site if it is per site "
        "or the user names no total, total if it is an overall count.\n"
        "- display: links if the user wants a list of links, offers or listings (job searches always); "
        "answer if they want information explained.\n"
        "- refs: numbers of previously shown results the user refers to ('number 3', 'the Capgemini one', "
        "'summarize the second'), otherwise [].\n"
        "A follow-up like 'retry with linkedin' or 'again' keeps the previous queries and changes only what the "
        "user changed.\n"
        "Domains that produced good results in past searches (your own experience, learned automatically — "
        "prefer them in `suggest` when they fit the country/field): " + memory_line() + "\n\n"
        f"Previous search: {prev}\nPreviously shown results:\n{shown}\n\n"
        f"Conversation:\n{ctx}\n\nLast message: {user_text}")
    raw = await llm_json(llm_url, prompt, PLAN_SCHEMA, "search_plan", 400, 60)
    if raw:
        plan = norm_plan(raw, user_text)
        if plan["search"] and not plan["queries"]:
            plan["queries"] = [fallback_job_query(user_text) if plan["intent"] == "jobs" else user_text]
        return plan
    print("[planner failed, using keyword rules]")
    return heuristic_plan(user_text, last_plan)
