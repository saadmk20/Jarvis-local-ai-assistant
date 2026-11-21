"""Search providers (ddgs, searxng, tavily): health, pacing, caching and the fetch itself."""
import asyncio
import time

import httpx

from .config import (CACHE_TTL, COOLDOWN, DDG_REGION, DEEP_GAPS, GAPS, PROVIDER_ORDER,
                     SEARXNG_URL, TAVILY_API_KEY, TAVILY_ONLY_ON_FAILURE, UA)
from .utils import fold, fuse, item, short


try:
    from ddgs import DDGS
except ImportError:
    DDGS = None


class Gate:
    """Spaces out requests to one provider so we stay under its rate limit."""

    def __init__(self, gap):
        self.gap, self.lock, self.next = gap, asyncio.Lock(), 0.0

    async def wait(self):
        if self.gap <= 0:
            return
        async with self.lock:
            now = time.monotonic()
            if self.next > now:
                await asyncio.sleep(self.next - now)
            self.next = max(now, self.next) + self.gap


_gates, _http = {}, None      # created lazily, inside the running event loop


def gate(name, deep=False):
    """One gate per (provider, mode): deep research talks to the providers at a gentler pace."""
    key = (name, bool(deep))
    if key not in _gates:
        _gates[key] = Gate((DEEP_GAPS if deep else GAPS).get(name, 0.0))
    return _gates[key]


def http():
    global _http
    if _http is None:
        _http = httpx.AsyncClient(timeout=15, follow_redirects=True, headers=UA)
    return _http


def with_site(q, domain):
    return f"{q} site:{domain}" if domain else q


class EnginesDown(RuntimeError):
    """The provider answered, but its own upstream engines were blocked. Retrying right away will not help."""


async def p_searxng(q, n, recency, news, lang, domain):
    params = {"q": with_site(q, domain), "format": "json", "language": lang if lang in DDG_REGION else "all",
              "safesearch": 0, "categories": "news" if news else "general"}
    if recency:
        params["time_range"] = recency
    r = await http().get(SEARXNG_URL.rstrip("/") + "/search", params=params)
    r.raise_for_status()
    data = r.json()
    rows = [item(x.get("title"), x.get("url"), x.get("content"), x.get("publishedDate"))
            for x in data.get("results", [])][:n]
    bad = data.get("unresponsive_engines") or []
    if not rows and bad:        # SearXNG's own engines were blocked or timed out: not a real "no results"
        raise EnginesDown("searxng engines unresponsive: " + ", ".join(f"{e[0]} ({e[1]})" for e in bad[:4]))
    return rows


async def p_tavily(q, n, recency, news, lang, domain):
    body = {"query": q, "max_results": min(n, 20), "topic": "news" if news else "general", "search_depth": "basic"}
    if recency:
        body["time_range"] = recency
    if domain:
        body["include_domains"] = [domain.split("/")[0]]
    r = await http().post("https://api.tavily.com/search", json=body,
                          headers={"Authorization": f"Bearer {TAVILY_API_KEY}"})
    r.raise_for_status()
    return [item(x.get("title"), x.get("url"), x.get("content"), x.get("published_date") or "")
            for x in r.json().get("results", [])][:n]


def _ddgs_sync(q, n, recency, news, lang, domain):
    tl = {"day": "d", "week": "w", "month": "m", "year": "y"}.get(recency)
    region = DDG_REGION.get(lang, "wt-wt")
    with DDGS() as d:
        if news and not domain:
            rows = d.news(q, region=region, safesearch="off", timelimit=tl, max_results=n)
        else:
            rows = d.text(with_site(q, domain), region=region, safesearch="off", timelimit=tl, max_results=n)
    return [item(x.get("title"), x.get("url") or x.get("href"), x.get("body"), x.get("date") or "", x.get("source") or "")
            for x in rows or []]


async def p_ddgs(q, n, recency, news, lang, domain):
    try:
        return await asyncio.to_thread(_ddgs_sync, q, n, recency, news, lang, domain)
    except Exception as e:
        if "no results" in str(e).lower():
            return []
        raise


PROVIDERS = {"searxng": p_searxng, "tavily": p_tavily, "ddgs": p_ddgs}


def active_providers():
    ok = {"searxng": bool(SEARXNG_URL), "tavily": bool(TAVILY_API_KEY), "ddgs": DDGS is not None}
    return [p for p in PROVIDER_ORDER if ok.get(p)]


CACHE = {}


COOL = {}


_diagnosed = False


async def diagnose():
    """Runs once per start: prints which providers are active and whether SearXNG is reachable and configured."""
    global _diagnosed
    if _diagnosed:
        return
    _diagnosed = True
    print("[search] providers in order:", active_providers())
    if not SEARXNG_URL:
        print("[search] SEARXNG_URL is empty, so SearXNG is not used")
        return
    try:
        r = await http().get(SEARXNG_URL.rstrip("/") + "/search", params={"q": "test", "format": "json"}, timeout=10)
        if r.status_code == 403:
            print("[search] SearXNG answered 403: put json under search: formats: in settings.yml and set "
                  "server: limiter: false, then run: docker restart searxng")
        elif r.status_code != 200:
            print(f"[search] SearXNG answered HTTP {r.status_code}")
        else:
            d = r.json()
            print(f"[search] SearXNG OK: {len(d.get('results', []))} results for a test query, "
                  f"blocked engines: {d.get('unresponsive_engines') or 'none'}")
    except Exception as e:
        print(f"[search] SearXNG not reachable at {SEARXNG_URL}: {short(e)} (is the container running? docker ps)")


async def run_search(q, n, recency, news, lang, domain, fresh=False, rescue=False, budget=None, deep=False,
                     fuse_all=False):
    """One query through the provider chain. Returns (rows, provider, error).
    error is None for a normal answer, including a genuine 'no results'; it is set only when every provider failed.
    rescue=True is the last attempt for a source that came back empty: every provider is tried again, cooldowns
    are ignored, and Tavily is allowed even after an empty answer. budget caps the Tavily calls per question.
    deep=True spaces the requests further apart (deep research mode). fuse_all=True (deep research) does not stop
    at the first provider that answers: every provider runs and the results are fused into one ranked list."""
    key = (q.lower(), n, recency, news, lang, domain, fuse_all)
    hit = CACHE.get(key)
    if hit and not fresh and not rescue and time.time() - hit[0] < CACHE_TTL:
        return hit[1], hit[2], None
    provs = active_providers()
    if not provs:
        return [], None, "no search provider available (pip install ddgs, or set a key / SEARXNG_URL)"
    last_err, clean_empty, used, hard_err = None, False, None, False
    got = []                               # (provider, rows) collected when fuse_all
    ready = provs if rescue else [x for x in provs if COOL.get(x, 0) <= time.time()]
    for name in (ready or provs):          # skip providers that just failed, unless every one of them did
        if name == "tavily":
            if TAVILY_ONLY_ON_FAILURE and not rescue and not hard_err:
                continue                   # paid credits: only when a previous provider really failed
            if budget is not None and budget.get("tavily", 0) <= 0:
                continue
            if budget is not None:
                budget["tavily"] -= 1          # one credit per call, not per retry attempt
        for attempt in range(2):
            try:
                await gate(name, deep).wait()
                rows = await PROVIDERS[name](q, n, recency, news, lang, domain)
                if not rows and news:                     # nothing in the news index: try the normal index
                    await gate(name, deep).wait()
                    rows = await PROVIDERS[name](q, n, recency, False, lang, domain)
                if domain:
                    rows = [r for r in rows if fold(domain) in fold(r["url"])]
                if rows:
                    CACHE[key] = (time.time(), rows, name)
                    if not fuse_all:
                        return rows, name, None
                    got.append((name, rows))
                    break                  # this provider answered: deep mode moves on to the next one
                clean_empty, used = True, name
                break
            except Exception as e:
                last_err = short(e)
                print(f"[search {name} failed: {last_err}]")
                if not isinstance(e, EnginesDown):
                    hard_err = True        # searxng upstream blocked is noise, not a reason to spend credits
                if isinstance(e, EnginesDown) or attempt == 1:
                    COOL[name] = time.time() + COOLDOWN
                    break
                await asyncio.sleep(2.0)
    if fuse_all and got:
        rows = fuse([r for _n, r in got])
        label = "+".join(n2 for n2, _r in got)
        CACHE[key] = (time.time(), rows, label)
        return rows, label, None
    return ([], used, None) if clean_empty else ([], None, last_err)
