"""Page fetching ladder and reading: plain HTTP, trafilatura, then a real stealth browser."""
import asyncio
import os
import time
from urllib.parse import urlparse

from .config import PAGE_CHARS
from .engines import http
from .planning import llm_json


try:
    import trafilatura
except ImportError:
    trafilatura = None


try:
    from scrapling.fetchers import AsyncFetcher, StealthyFetcher
except ImportError:
    AsyncFetcher = StealthyFetcher = None


STEALTH_MAX = 3                    # stealth-browser fetches per deep search (slow, one at a time)

READ_LOG = os.environ.get("JARVIS_READ_LOG", "1") != "0"   # per-page ladder log; set 0 to silence


_scrap_sem = asyncio.Semaphore(1)


JOBS_EXTRACT_MAX = 3               # LLM job-offer extraction per deep search ("scrapegraph" node)


def _scrap_html(page):
    """Best-effort: pull the HTML string out of a Scrapling response (API varies by version)."""
    for attr in ("html_content", "body", "html"):
        v = getattr(page, attr, None)
        if isinstance(v, bytes):
            v = v.decode("utf-8", "ignore")
        if isinstance(v, str) and len(v) > 50:
            return v
    return ""


async def read_pages(results, total=None, chars=None, deep=False):
    """Downloads pages and extracts their text (needs trafilatura). Returns one string per result.
    total caps each page in seconds, chars the text kept per page.

    Fetch ladder for pages that plain httpx cannot read:
      1. plain httpx (fast, free) — as before
      2. Scrapling AsyncFetcher: curl_cffi with browser-like TLS/headers (defeats naive bot checks)
      3. deep mode only: Scrapling StealthyFetcher — a real stealth browser (defeats JS challenges),
         max STEALTH_MAX per search, one at a time."""
    if trafilatura is None or not results:
        return [""] * len(results)

    sem = asyncio.Semaphore(4)                   # download several pages at once, politely
    stealth_left = [STEALTH_MAX if deep else 0]
    per_page = (total or 12) + (45 if deep and (AsyncFetcher or StealthyFetcher) else 0)

    def extract(html):
        return trafilatura.extract(html, include_comments=False, include_tables=False) or ""

    def host_of(r):
        return urlparse(r.get("url") or "").netloc.removeprefix("www.")

    def note(r, rung, text, t0, capped=False):
        """One line per page: which rung produced the text, how much, and how long it took."""
        if not READ_LOG:
            return
        print(f"[read] {rung:<12}{len(text):>6} chars {time.perf_counter() - t0:>5.1f}s  {host_of(r)}"
              + ("  (stealth cap reached)" if capped else ""))

    async def plain(url):
        r = await http().get(url, timeout=7)
        if r.status_code != 200 or "html" not in r.headers.get("content-type", ""):
            return ""
        return await asyncio.to_thread(extract, r.text)

    async def one(r):
        async with sem:
            url, t0, capped = r["url"], time.perf_counter(), False

            def done(text, rung):
                note(r, rung, text, t0, capped)
                return text[:chars or PAGE_CHARS], rung, capped

            try:
                text = await plain(url)
            except Exception:
                text = ""
            if len(text) >= 250 or (AsyncFetcher is None and StealthyFetcher is None):
                return done(text, "http")
            try:                                 # page failed, blocked or is a JS shell: ladder
                for stealth in (False, True):
                    if stealth:
                        if StealthyFetcher is None:
                            continue
                        if stealth_left[0] <= 0:
                            capped = True        # budget spent: this page keeps what httpx got
                            continue
                        async with _scrap_sem:
                            stealth_left[0] -= 1
                            html = _scrap_html(await asyncio.to_thread(
                                StealthyFetcher.fetch, url, headless=True, timeout=60000))
                    else:
                        if AsyncFetcher is None:
                            continue
                        html = _scrap_html(await AsyncFetcher.get(url, stealthy_headers=True, timeout=30))
                    t = await asyncio.to_thread(extract, html)
                    if t:
                        return done(t, "stealth" if stealth else "curl_cffi")
            except Exception:
                pass
            return done(text, "http-partial" if text else "failed")   # whatever the plain attempt got

    # per-page timeouts instead of killing the whole batch: one slow stealth fetch must not
    # cost us the pages that already succeeded
    async def safe(i, r):
        try:
            text, rung, capped = await asyncio.wait_for(one(r), per_page)
        except Exception:
            text, rung, capped = "", "timeout", False
            if READ_LOG:
                print(f"[read] {'timeout':<12}{0:>6} chars {per_page:>5.0f}s  {host_of(r)}")
        return i, text, rung, capped

    out = await asyncio.gather(*(safe(i, r) for i, r in enumerate(results)))
    pages = [""] * len(results)
    tally = {}
    for i, text, rung, capped in out:
        pages[i] = text
        key = rung + ("+cap" if capped else "")
        tally[key] = tally.get(key, 0) + 1
    if READ_LOG:
        print(f"[read] {len(pages)} pages: " + ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    return pages


JOB_SCHEMA = {"type": "object",
              "properties": {"offers": {"type": "array", "items": {"type": "object", "properties": {
                  "title": {"type": "string"}, "company": {"type": "string"},
                  "location": {"type": "string"}, "link": {"type": "string"},
                  "deadline": {"type": "string"}},
                  "required": ["title"]}}},
              "required": ["offers"]}


async def extract_job_offers(llm_url, results, pages):
    """The 'scrapegraph' node, lean edition: one structured LLM call per fetched job-board page
    (max JOBS_EXTRACT_MAX) to pull out the individual offers the page contains.
    Returns a list aligned with the results: "" where nothing was extracted."""
    notes = [""] * len(results)
    done = 0
    for i, (r, page) in enumerate(zip(results, pages)):
        if done >= JOBS_EXTRACT_MAX or not page:
            continue
        done += 1
        prompt = ("This is the text of a job or internship listing page (it may be messy). Extract every "
                  "job or internship offer you can find. Return JSON exactly like "
                  '{"offers": [{"title", "company", "location", "link", "deadline"}]}. '
                  'Use "link" only when a full URL is visible in the text; "deadline"/"location"/"company" '
                  'may be "" if unknown. If there are no offers, return {"offers": []}. Page text:\n\n'
                  + page[:3000])
        raw = await llm_json(llm_url, prompt, JOB_SCHEMA, "job_extract", 500, 60)
        offers = (raw or {}).get("offers") or []
        offers = [o for o in offers if isinstance(o, dict) and o.get("title")]
        if offers:
            lines = []
            for o in offers[:8]:
                bits = [o.get("title", ""), "@" + o.get("company", "") if o.get("company") else "",
                        o.get("location", ""), ("deadline: " + o["deadline"]) if o.get("deadline") else ""]
                s = " — ".join(b for b in bits if b)
                if o.get("link", "").startswith("http"):
                    s += " [" + o["link"] + "]"
                lines.append("• " + s)
            notes[i] = "\nOffers extracted from this page:\n" + "\n".join(lines)
    return notes


def describe(r, page):
    head = f"[{r['n']}] {r['title']} ({r['source']}{', ' + r['date'] if r['date'] else ''})"
    return head + "\n" + (page or r["snippet"])
