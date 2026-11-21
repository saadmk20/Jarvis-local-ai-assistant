"""Fusion, relevance scoring and per-source collection."""
import asyncio
import re
from urllib.parse import quote_plus, urlparse

from .config import BLOCKED, DEEP_FETCH, MAX_JUDGED
from .engines import active_providers, run_search
from .utils import (GENERIC, STOP, SYN, fold, fuse, item, norm_url, short, word_in)


def fuse(lists, k=60):
    """Reciprocal rank fusion over the results of several query variants, de-duplicated by URL."""
    score, best = {}, {}
    for lst in lists:
        for rank, r in enumerate(lst):
            key = norm_url(r["url"])
            score[key] = score.get(key, 0) + 1 / (k + rank + 1)
            if key not in best or len(r["snippet"]) > len(best[key]["snippet"]):
                best[key] = r
    return [best[x] for x in sorted(score, key=score.get, reverse=True)]


def topic_tokens(queries, sites):
    skip = {w for s in sites for a in s.get("alias", []) for w in re.findall(r"[a-z0-9]+", fold(a))}
    toks = set()
    for q in queries:
        for t in re.findall(r"[a-z0-9]+", fold(q)):
            if len(t) >= 2 and t not in STOP:
                toks.add(t)
    return (toks - GENERIC - skip) or None       # None = nothing specific to check, so no filtering


def relevant(r, toks):
    if not toks:
        return True
    text = fold(r["title"] + " " + r["snippet"] + " " + r["url"])
    return any(word_in(s, text) for t in toks for s in SYN.get(t, [t]))


def keep_basic(r):
    """Drops bare homepages and junk domains. Whether a result is on topic is decided by the model, not here."""
    u = urlparse(r["url"])
    return bool(u.path.strip("/")) and not any(b in u.netloc for b in BLOCKED)


def keep(r, toks):          # keyword check: only used when the model is unavailable
    return keep_basic(r) and relevant(r, toks)


async def collect(site, queries, n, recency, news, lang, label, fresh=False, budget=None, auto=False, deep=False):
    """Gathers the candidates for one site (or the open web when site is None): every query variant, fused and
    de-duplicated, real postings first. Nothing is judged yet. An empty source gets one rescue attempt."""
    domain = site["domain"] if site else None
    fetch = max(n * 3, DEEP_FETCH) if deep else max(n * 2, 8)

    async def go(rescue):
        return await asyncio.gather(*(run_search(q, fetch, recency, news, lang, domain, fresh=fresh or rescue,
                                                 rescue=rescue, budget=budget, deep=deep, fuse_all=deep)
                                      for q in queries))

    runs = await go(False)
    cands = [r for r in fuse([x[0] for x in runs]) if keep_basic(r)]
    used_open = False
    if not cands and site:                       # site-restricted search found nothing: the open web
        try:                                     # with the site's name as keyword often does better
            open_runs = await asyncio.gather(*(run_search(q + " " + site["name"], fetch, recency, news,
                                                          lang, None, fresh=True, rescue=False, budget=None,
                                                          deep=deep, fuse_all=deep)
                                               for q in queries))
            cands = [r for r in fuse([x[0] for x in open_runs]) if keep_basic(r)]
            used_open = bool(cands)
            if used_open:
                print(f"[search] {site['name']}: {len(cands)} candidates via open-web fallback")
        except Exception as e:
            print(f"[search {site['name']} open fallback failed: {short(e)}]")
    tavily_left = bool(budget and budget.get("tavily", 0) > 0 and "tavily" in active_providers())
    if not cands and (any(x[2] for x in runs) or tavily_left):
        runs = await go(True)
        cands = [r for r in fuse([x[0] for x in runs]) if keep_basic(r)]
    post = site.get("post") if site else ""
    if post:
        cands.sort(key=lambda r: 0 if re.search(post, r["url"]) else 1)         # real postings first
    if cands:
        status = "ok"
    elif runs and all(x[2] for x in runs):
        status = "failed"
    else:
        status = "empty"
    only_if = site.get("only_if") if site else None
    shortcut = site.get("search") if site else None
    if shortcut and only_if and not re.search(only_if, fold(queries[0])):
        shortcut = None                                  # e.g. the Indeed Maroc shortcut only makes sense for Morocco
    print(f"[search] {site['name'] if site else label}: candidates={len(cands)} status={status} "
          f"providers={[x[1] or x[2] for x in runs]}")
    return {"site": site["name"] if site else label, "status": status, "auto": auto,
            "provider": next((x[1] for x in runs if x[0]), None) or ("open-web" if used_open else None),
            "error": next((x[2] for x in runs if x[2]), None) if status == "failed" else None,
            "search_url": shortcut.format(q=quote_plus(queries[0])) if shortcut else None,
            "cands": cands, "seen": {norm_url(r["url"]) for r in cands}, "items": []}
