"""Web search engine for JARVIS. One general pipeline for every kind of question (news, jobs, facts, research).

    question -> planner (schema-constrained JSON) -> query variants x sites -> provider chain with retries
             -> rank fusion + dedupe + relevance filter -> links card on screen and/or page text for the model

"Web search" (UI) runs this quickly. "Deep research" runs the same pipeline with more query variants, gentler
request rates (fewer captchas), more Tavily credits and several pages actually read: much slower, for demanding
questions like a full job hunt.

Providers are tried in PROVIDER_ORDER and skipped when not configured:
    searxng : self-hosted meta search, free.  Set SEARXNG_URL (e.g. http://127.0.0.1:8888)
              and enable the json format in SearXNG's settings.yml:  search: formats: [html, json]
    ddgs    : DuckDuckGo scraper, no key (pip install ddgs).
    tavily  : Tavily API (paid credits). Set TAVILY_API_KEY. Last resort, only when the others fail.
Put your keys in a .env file next to this module (see .env.example) or as environment variables.
Optional: pip install trafilatura   (extracts the full text of pages)
"""
import asyncio
import json
import os
import difflib
import re
import time
import unicodedata
from pathlib import Path
from urllib.parse import parse_qsl, quote_plus, urlencode, urlparse

import httpx

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None
try:
    import trafilatura
except ImportError:
    trafilatura = None
# Optional page-fetch ladder (pip install scrapling): pages that plain httpx cannot read
# (JavaScript-rendered shells, anti-bot sites) are retried with browser-like TLS headers,
# then — deep mode only, capped — with a real stealth browser. Absent library = old behavior.
try:
    from scrapling.fetchers import AsyncFetcher, StealthyFetcher
except ImportError:
    AsyncFetcher = StealthyFetcher = None
STEALTH_MAX = 3                    # stealth-browser fetches per deep search (slow, one at a time)
_scrap_sem = asyncio.Semaphore(1)
JOBS_EXTRACT_MAX = 3               # LLM job-offer extraction per deep search ("scrapegraph" node)

# ------------------------------------------------------------------ settings
def _load_env():
    """Reads KEY=VALUE lines from a .env file next to this module (real environment variables win)."""
    f = Path(__file__).with_name(".env")
    try:
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass
    except Exception as e:
        print("[.env not read]", e)


_load_env()
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://127.0.0.1:8888")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
# ddgs first: it rotates between several search backends and has a real news index, which gave the best results.
# SearXNG (only Bing and Yahoo work from this PC) is the second source. Tavily is paid: last resort only.
PROVIDER_ORDER = ["ddgs", "searxng", "tavily"]
RELEVANCE_FILTER = True    # keyword check, used only when the local model cannot judge the results
JUDGE_ENABLED = True       # the local model checks every result against what you asked (country, field, kind of page)
MAX_ROUNDS = 2             # 1 = no second attempt; 2 = weak results make the model write new queries and search again
WEAK_BELOW = 3             # a source with fewer good results than this (or your limit, if smaller) is searched again
JUDGE_CHUNK = 40           # results shown to the model per judging call (about 3 seconds each on a fast GPU)
MAX_JUDGED = 120           # never judge more than this many results per round
TAVILY_MAX_PER_SEARCH = 4  # paid credits: at most this many Tavily calls per question, even for empty sources
FALLBACK_JOB_BOARDS = True # job search, no site named and none suggested by the model: use the boards marked jobs in SITES
COOLDOWN = 60               # seconds a provider is skipped after it failed, so a blocked one is not hammered
TAVILY_ONLY_ON_FAILURE = True   # paid credits: used only when the other providers errored, not on a genuine "no results"
DEFAULT_LIMIT = 5          # results per site (or in total for an open-web search); changeable by voice or in the UI
MAX_LIMIT = 20
PAGES_TO_READ = 5          # pages downloaded and read when the answer is spoken (not for plain link lists)
PAGE_CHARS = 3000
CACHE_TTL = 600            # seconds: the same search gives the same results for 10 minutes
GAPS = {"searxng": 2.0, "tavily": 0.15, "ddgs": 0.8}   # minimum seconds between requests per provider

# "Deep research" (UI toggle): the same pipeline, slower and far more thorough. For demanding searches
# (a full job hunt) where quality matters more than the wait.
DEEP_MAX_ROUNDS = 4        # refine rounds instead of MAX_ROUNDS
DEEP_MAX_Q = 5             # query variants per source instead of 2-3
DEEP_FETCH = 14            # candidates fetched per query instead of max(n*2, 8)
DEEP_PAGES = 8             # pages actually read instead of PAGES_TO_READ
DEEP_PAGE_CHARS = 3500     # characters kept per page in deep mode
DEEP_GAPS = {"searxng": 3.0, "ddgs": 1.5, "tavily": 0.15}   # gentler pace: fewer captchas and suspensions
DEEP_TAVILY = 8            # tavily credits for the whole question instead of TAVILY_MAX_PER_SEARCH
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"}
DDG_REGION = {"en": "us-en", "fr": "fr-fr", "es": "es-es"}

# Sites the user can name ("on LinkedIn and ReKrute"). "domain" feeds the site: filter, "post" recognises a real
# posting URL (those are listed first), "search" is a live-search shortcut ({q} = keywords), "jobs" marks the
# boards used when a job search names no site. Any domain the user says (e.g. "welovedevs.com") also works.
SITES = {
    "indeed": {"name": "Indeed", "domain": "indeed.com", "post": r"viewjob|/rc/clk|jk=", "jobs": True,
               "search": "https://ma.indeed.com/jobs?q={q}&l=Maroc", "only_if": r"maroc|morocco", "alias": ["indeed"]},
    "linkedin": {"name": "LinkedIn", "domain": "linkedin.com/jobs", "post": r"/jobs/view/", "jobs": True,
                 "search": "https://www.linkedin.com/jobs/search/?keywords={q}&location=Morocco",
                 "only_if": r"maroc|morocco", "alias": ["linkedin", "linkdin", "linkedn", "linked in", "لينكدان", "لينكد ان"]},
    "rekrute": {"name": "ReKrute", "domain": "rekrute.com", "post": r"offre-emploi", "jobs": True,
                "search": None, "alias": ["rekrute"]},
    "emploi": {"name": "Emploi.ma", "domain": "emploi.ma", "post": r"offre", "jobs": True,
               "search": None, "alias": ["emploi.ma"]},
    "marocannonces": {"name": "MarocAnnonces", "domain": "marocannonces.com", "post": "", "jobs": True,
                      "search": None, "alias": ["marocannonces", "maroc annonces"]},
    "jobzyn": {"name": "Jobzyn", "domain": "jobzyn.com", "post": "", "jobs": True, "search": None,
               "alias": ["jobzyn"]},
    "bayt": {"name": "Bayt", "domain": "bayt.com", "post": r"/jobs?/", "jobs": True, "search": None,
             "alias": ["bayt"]},
    "joblia": {"name": "Joblia", "domain": "joblia.ma", "post": "", "jobs": True, "search": None,
               "alias": ["joblia"]},
    "reddit": {"name": "Reddit", "domain": "reddit.com", "post": "", "search": None, "alias": ["reddit"]},
    "github": {"name": "GitHub", "domain": "github.com", "post": "", "search": None, "alias": ["github"]},
    "stackoverflow": {"name": "Stack Overflow", "domain": "stackoverflow.com", "post": "", "search": None,
                      "alias": ["stackoverflow", "stack overflow"]},
    "wikipedia": {"name": "Wikipedia", "domain": "wikipedia.org", "post": "", "search": None,
                  "alias": ["wikipedia"]},
    "youtube": {"name": "YouTube", "domain": "youtube.com", "post": "", "search": None, "alias": ["youtube"]},
}
ALIAS_MAP = {a: k for k, s in SITES.items() for a in s["alias"]}
BLOCKED = ("pinterest.", "facebook.com", "instagram.com", "tiktok.com")


# ------------------------------------------------------------------ text helpers
def fold(s):
    """Lower-case and strip accents/diacritics so matching works across French, Spanish and Arabic."""
    s = unicodedata.normalize("NFKD", (s or "").lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def word_in(word, text):
    return re.search(r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])", text) is not None


NEWS_WORDS = re.compile(r"\b(news|headlines?|latest|breaking|happening|going on|this week|yesterday|"
                        r"actualit\w*|noticias|hoy)\b|اخبار", re.I)
JOBS_RE = re.compile(r"\b(stages?|pfe|stagiaires?|internships?|jobs?|emplois?|recrutement|recruitment|alternance)\b"
                     r"|وظائف|توظيف|تدريب", re.I)
RETRY_RE = re.compile(r"\b(retry|again|reessaie\w*|reessayer|encore|refais|relance\w*|redo)\b", re.I)
LIMIT_RE = re.compile(r"\b(\d{1,2})\s*(?:par site|per site|each|chacun|resultats?|results?|liens|links|offres|"
                      r"offers|jobs|annonces)\b|\btop\s*(\d{1,2})\b", re.I)
TOPIC_RE = re.compile(r"data science|data engineer\w*|data analyst|business intelligence|machine learning|"
                      r"deep learning|power ?bi|big data|\bbi\b|\bia\b|\bai\b|\bdata\b|\bml\b|\bnlp\b|devops|"
                      r"cyber\w*|python|java|full ?stack", re.I)
STOP = set("le la les de des du un une et ou en pour sur au aux the a an and or of in for to on with d l je tu veux que "
           "cherche chercher trouve trouver donne moi me show give find search please stp svp".split())
GENERIC = set("stage stages pfe stagiaire stagiaires emploi emplois offre offres job jobs internship internships "
              "maroc morocco marocain recrutement fin etudes news today headlines breaking top latest actualite "
              "actualites 2025 2026 2027 web site sites".split())
AI_SYN = ["ia", "ai", "intelligence artificielle", "artificial intelligence", "machine learning", "deep learning",
          "ml", "llm", "genai", "nlp"]
SYN = {"ia": AI_SYN, "ai": AI_SYN, "ml": AI_SYN,
       "bi": ["bi", "business intelligence", "power bi", "powerbi", "msbi", "ssis", "ssas", "decisionnel", "dashboard"],
       "data": ["data", "donnees", "big data", "analytics", "analyst"]}


def fallback_job_query(text):
    topics = []
    for m in TOPIC_RE.finditer(text):
        t = m.group(0).strip()
        if t.lower() not in [x.lower() for x in topics]:
            topics.append(t)
    kind = "stage PFE" if re.search(r"pfe|stage", text, re.I) else "emploi"
    return " ".join([kind, *topics, "Maroc"])


def norm_url(u):
    p = urlparse(u)
    host = p.netloc.lower().removeprefix("www.")
    q = [(k, v) for k, v in parse_qsl(p.query) if not k.startswith("utm_") and k not in ("fbclid", "gclid")]
    return host + p.path.rstrip("/") + ("?" + urlencode(q) if q else "")


def item(title, url, snippet, date="", source=""):
    url = url or ""
    return {"title": (title or "").strip()[:160], "url": url, "snippet": re.sub(r"\s+", " ", snippet or "").strip()[:300],
            "date": (date or "")[:10], "source": source or urlparse(url).netloc.removeprefix("www.")}


def short(e):
    if isinstance(e, httpx.HTTPStatusError):
        return f"HTTP {e.response.status_code}"
    return f"{type(e).__name__}: {e}"[:120]


# ------------------------------------------------------------------ sites
def detect_sites(text):
    """Finds named sites in the user's message. Tolerates typos and speech-recognition spellings
    ('Linked-in', 'link din', 'rekrut')."""
    f = fold(text)
    out = []

    def add(key):
        if SITES[key] not in out:
            out.append(SITES[key])

    for alias, key in ALIAS_MAP.items():
        if word_in(alias, f):
            add(key)
    words = re.findall(r"[a-z0-9]+", f)
    cands = set(words) | {a + b for a, b in zip(words, words[1:])}      # 'link din' -> 'linkdin'
    for key, s in SITES.items():
        alias = s["alias"][0]                    # canonical name only; typo aliases are too close to real words
        if len(alias) >= 5 and re.fullmatch(r"[a-z0-9]+", alias):
            if any(len(c) >= 5 and difflib.SequenceMatcher(None, c, alias).ratio() >= 0.88 for c in cands):
                add(key)
    return out


def site_from_name(name):
    f = fold(name).strip()
    if not f:
        return None
    if f in ALIAS_MAP:
        return SITES[ALIAS_MAP[f]]
    for alias, key in ALIAS_MAP.items():
        if word_in(alias, f):
            return SITES[key]
    if re.fullmatch(r"[a-z0-9.-]+\.[a-z]{2,}(/\S*)?", f):
        return custom_site(f, f)
    return None


KNOWN_FILE = Path(__file__).with_name("known_sites.json")    # learned name -> domain; you can edit it by hand
GENERIC_DOMAINS = ("wikipedia.", "facebook.", "linkedin.", "youtube.", "instagram.", "twitter.", "x.com", "reddit.",
                   "pinterest.", "tiktok.")
GROUP_WORDS = re.compile(r"\b(sites?|all|tous|toutes|moroccan|marocains?|boards?|plateformes?|websites?|web)\b", re.I)


def _load_known():
    try:
        return {fold(k): v for k, v in json.loads(KNOWN_FILE.read_text(encoding="utf-8")).items()}
    except Exception:
        return {}


KNOWN = _load_known()


def _save_known():
    try:
        KNOWN_FILE.write_text(json.dumps(KNOWN, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print("[known_sites.json not saved]", e)


def custom_site(name, domain):
    return {"name": name, "domain": domain, "post": "", "search": None, "alias": [fold(name).strip()]}


async def resolve_domain(name, lang):
    """Finds the website of a site the user named but that is not in SITES: look it up once, remember it."""
    key = fold(name).strip()
    if key in KNOWN:
        return KNOWN[key]
    slug = re.sub(r"[^a-z0-9]", "", key)
    if len(slug) < 3:
        return None
    rows, _, _ = await run_search(f"{name} official website", 8, None, False, lang, None)
    for r in rows:
        host = urlparse(r["url"]).netloc.lower().removeprefix("www.")
        if any(g in host for g in GENERIC_DOMAINS):
            continue
        if slug in re.sub(r"[^a-z0-9]", "", host):          # the domain must contain the name, or we do not guess
            KNOWN[key] = host
            _save_known()
            return host
    return None


async def resolve_sites(names, text, lang):
    """Returns (sites, unresolved_names). Known sites and typed domains are instant; other names are looked up."""
    sites, unresolved = [], []

    def add(s):
        if s and all(s["domain"] != x["domain"] for x in sites):
            sites.append(s)

    for n in names or []:
        if GROUP_WORDS.search(n) or len(n.split()) > 4:      # "all moroccan sites" etc. is not one site
            continue
        s = site_from_name(n)
        if not s:
            d = await resolve_domain(n, lang)
            s = custom_site(n, d) if d else None
        if s:
            add(s)
        else:
            unresolved.append(n)
    for s in detect_sites(text):
        add(s)
    return sites, unresolved


def strip_sites(queries, sites):
    """Site names are filters, never search keywords."""
    words = [a for s in sites for a in s.get("alias", [])]
    out = []
    for q in queries:
        q = re.sub(r"\bsite:\S+", " ", q)
        for w in words:
            q = re.sub(r"(?<![A-Za-z0-9])" + re.escape(w) + r"(?![A-Za-z0-9])", " ", q, flags=re.I)
        q = re.sub(r"\s+", " ", q).strip()
        if q:
            out.append(q)
    return out


# ------------------------------------------------------------------ providers
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


# ------------------------------------------------------------------ fusion and filtering
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


def public(g):
    return {k: g[k] for k in ("site", "status", "provider", "error", "search_url", "items")}


def address_rows(items, n=6):
    """The exact URLs of some results, numbered as on screen, for the model to copy if it opens one."""
    rows, k = [], 0
    for r in items:
        if not r.get("url"):
            continue
        k += 1
        rows.append(f"{r.get('n', k)}. {r['url']}")
        if k >= n:
            break
    return "\n".join(rows)


OPEN_HINT = ("\nIf the user asked you to open, play, show or take them to one of these, or is asking for something "
             "to watch, listen to, read or visit rather than for a fact, end your reply with a final line "
             "'OPEN: <address>' copied exactly from this list. Otherwise do not write the word OPEN.\n")


def number(groups, start=1):
    n, flat, seen = start, [], set()
    for g in groups:
        keep_items = []
        for it in g["items"]:
            key = norm_url(it["url"])
            if key in seen:
                continue                      # the same page found under two sources
            seen.add(key)
            it["n"] = n
            n += 1
            flat.append(it)
            keep_items.append(it)
        g["items"] = keep_items
    return flat


def interleave(groups):
    out, i = [], 0
    while any(i < len(g["items"]) for g in groups):
        out += [g["items"][i] for g in groups if i < len(g["items"])]
        i += 1
    return out


def trim_total(groups, n):
    keep_ids = {id(x) for x in interleave(groups)[:n]}
    for g in groups:
        g["items"] = [x for x in g["items"] if id(x) in keep_ids]


# ------------------------------------------------------------------ experience: sources that worked
# The assistant's own learned source map, no human input: every time a search ends, the domains of
# results the judge called "good" are counted per intent. The planner reads this on the next search
# ("domains that worked before") and may reuse them — that is how it learns sites by itself.
MEMORY_FILE = Path(__file__).with_name("source_memory.json")
SOURCE_MEM = {}          # "intent" -> {"domain": count}


def _load_memory():
    try:
        if MEMORY_FILE.is_file():
            d = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
            if isinstance(d, dict):
                SOURCE_MEM.update({str(k): v for k, v in d.items() if isinstance(v, dict)})
    except Exception as e:
        print("[source_memory.json not loaded]", e)


def _save_memory():
    try:
        top = sorted(SOURCE_MEM.items(), key=lambda kv: -sum(kv[1].values()))[:40]
        MEMORY_FILE.write_text(json.dumps(dict(top), ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as e:
        print("[source_memory.json not saved]", e)


def memory_line():
    """Compact view of the learned domains for the planner prompt."""
    parts = []
    for k, v in SOURCE_MEM.items():
        if v:
            best = ", ".join(f"{d}({c})" for d, c in sorted(v.items(), key=lambda x: -x[1])[:6])
            parts.append(f"{k}: {best}")
    return " | ".join(parts) or "(none yet — you will learn them yourself)"


def remember_sources(intent, groups):
    """Count domains of good results for this intent, and remember each domain in known_sites.json
    so the planner's suggestions resolve instantly next time — the assistant learns its own sites."""
    d = SOURCE_MEM.setdefault(intent, {})
    learned = False
    for g in groups:
        for it in g.get("items") or []:
            if it.get("match") != "good":
                continue
            dom = urlparse(it["url"]).netloc.lower()
            dom = dom[4:] if dom.startswith("www.") else dom
            if dom:
                d[dom] = min(d.get(dom, 0) + 1, 50)
                if dom not in KNOWN:             # learned domain becomes a resolvable site name
                    KNOWN[dom] = dom
                    learned = True
    if learned:
        _save_known()
    _save_memory()


_load_memory()                                   # the learned source map survives restarts

# ------------------------------------------------------------------ planner
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
        "products, people, research). false for greetings, chit-chat, coding and timeless knowledge.\n"
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


# ------------------------------------------------------------------ the model judges and improves the search
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


# ------------------------------------------------------------------ page reading
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

    async def plain(url):
        r = await http().get(url, timeout=7)
        if r.status_code != 200 or "html" not in r.headers.get("content-type", ""):
            return ""
        return await asyncio.to_thread(extract, r.text)

    async def one(r):
        async with sem:
            url = r["url"]
            try:
                text = await plain(url)
            except Exception:
                text = ""
            if len(text) >= 250 or (AsyncFetcher is None and StealthyFetcher is None):
                return text[:chars or PAGE_CHARS]
            try:                                 # page failed, blocked or is a JS shell: ladder
                for stealth in (False, True):
                    if stealth:
                        if StealthyFetcher is None or stealth_left[0] <= 0:
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
                        return t[:chars or PAGE_CHARS]
            except Exception:
                pass
            return text[:chars or PAGE_CHARS]    # whatever the plain attempt got

    # per-page timeouts instead of killing the whole batch: one slow stealth fetch must not
    # cost us the pages that already succeeded
    async def safe(i, r):
        try:
            return i, await asyncio.wait_for(one(r), per_page)
        except Exception:
            return i, ""

    out = await asyncio.gather(*(safe(i, r) for i, r in enumerate(results)))
    pages = [""] * len(results)
    for i, t in out:
        pages[i] = t
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


# ------------------------------------------------------------------ per-session search state
class WebSearch:
    def __init__(self, send, llm_url, limit=DEFAULT_LIMIT):
        self.send, self.llm_url = send, llm_url
        self.limit = max(1, min(MAX_LIMIT, int(limit)))
        self.reset()

    def reset(self):
        self.last_plan, self.last_results, self.last_groups, self.last_ctx = None, [], [], None

    def prefs(self):
        return {"type": "search_prefs", "limit": self.limit, "providers": active_providers()}

    async def _judge_groups(self, groups, request, queries, n, pools):
        """Lets the model pick the results that match the request. pools: id(group) -> candidates to judge.
        Picked results are merged with what the group already has (good ones first), up to n per group."""
        flat = [(g, r) for g in groups for r in pools.get(id(g), [])][:MAX_JUDGED]
        if not flat:
            return
        toks = topic_tokens(queries, [])
        verdict = None
        if JUDGE_ENABLED:
            await self.send({"type": "note", "text": f"Checking {len(flat)} results with the local model…"})
            verdict = await judge(self.llm_url, request, queries, [(g["site"], r) for g, r in flat])
        picked = {id(g): {"good": [], "partial": [], "unchecked": []} for g in groups}
        for i, (g, r) in enumerate(flat):
            if verdict is None or i in verdict["unknown"]:
                kind = "unchecked" if (relevant(r, toks) or not RELEVANCE_FILTER) else None   # model unavailable
            else:
                kind = "good" if i in verdict["good"] else "partial" if i in verdict["partial"] else None
            if kind:
                picked[id(g)][kind].append(dict(r, match=kind))
        if verdict is not None:
            print(f"[judge] {len(flat)} results -> good={len(verdict['good'])} partial={len(verdict['partial'])} "
                  f"unchecked={len(verdict['unknown'])}")
        for g in groups:
            p = picked[id(g)]
            old = g["items"]
            ranked = ([x for x in old if x["match"] == "good"] + p["good"] +
                      [x for x in old if x["match"] != "good"] + p["partial"] + p["unchecked"])
            seen, out = set(), []
            for x in ranked:
                k = norm_url(x["url"])
                if k not in seen:
                    seen.add(k)
                    out.append(x)
            g["items"] = out[:n]

    def _weak(self, groups, n):
        want = min(n, WEAK_BELOW)
        return [g for g in groups if sum(1 for x in g["items"] if x["match"] == "good") < want]

    async def context(self, user_text, history, lang, deep=False):
        """Runs the whole pipeline. Returns text to append to the user's message ('' = nothing to add)."""
        await diagnose()
        plan = await make_plan(self.llm_url, user_text, history, self.last_plan, self.last_results)
        if plan["refs"] and self.last_results:
            return await self._followup(plan)
        sites, unresolved = await resolve_sites(plan["sites"], user_text, lang)
        named = bool(sites or unresolved)
        print(f"[plan] intent={plan['intent']} search={plan['search']} sites={[s['name'] for s in sites]} "
              f"unresolved={unresolved} suggest={plan['suggest']} queries={plan['queries']} limit={plan['limit']}")
        if not plan["search"] and not named:
            return ""
        jobs, news = plan["intent"] == "jobs", plan["intent"] == "news"
        auto = False
        if not named and (jobs or plan["suggest"]):
            # nobody named a site: the model suggests the boards for this country and field, and the names are
            # turned into domains by a search (so a made-up name is simply dropped)
            sites, _ = await resolve_sites(plan["suggest"], "", lang)
            if not sites and jobs and FALLBACK_JOB_BOARDS:
                sites = [s for s in SITES.values() if s.get("jobs")]
            auto = True
        if plan["limit"]:
            self.limit = plan["limit"]                           # sticky for the rest of the session
            await self.send(self.prefs())
        n = self.limit
        recency = plan["recency"] or ("week" if news else None)
        queries = strip_sites(plan["queries"], sites)
        if not queries:
            queries = (self.last_plan or {}).get("queries") or [fallback_job_query(user_text) if jobs else user_text]
        if unresolved:
            await self.send({"type": "note", "text": "Could not find the website for: " + ", ".join(unresolved) +
                             ". Using the name as a keyword in a general web search instead."})
            queries = [q + " " + " ".join(unresolved) for q in queries]
        self.last_plan = dict(plan, queries=queries)
        label = "News" if news else ("Other sources" if sites else "Web")
        targets = [(s, auto) for s in sites] + ([(None, False)] if (auto or not sites) else [])
        max_q = DEEP_MAX_Q if deep else (3 if len(targets) <= 2 else 2)
        cap = max(n * 3, 12) if deep else max(n * 2, 8)   # candidates shown to the judge per source
        await self.send({"type": "note", "text": ("Deep research — searching: " if deep else "Searching: ") +
                         " | ".join(queries[:max_q]) +
                         (" on " + ", ".join(s["name"] for s in sites) if sites else "")})
        budget = {"tavily": DEEP_TAVILY if deep else TAVILY_MAX_PER_SEARCH}
        sem = asyncio.Semaphore(3 if deep else 4)   # fewer parallel sites: searxng's upstream engines rate limit fast

        async def go(s, a):
            async with sem:
                return await collect(s, queries[:max_q], n, recency, news, lang, label, budget=budget, auto=a,
                                     deep=deep)

        groups = list(await asyncio.gather(*(go(s, a) for s, a in targets)))
        await self._judge_groups(groups, user_text, queries, n, {id(g): g["cands"][:cap] for g in groups})
        for _ in range((DEEP_MAX_ROUNDS if deep else MAX_ROUNDS) - 1):   # weak results: the model writes better queries
            weak = self._weak(groups, n)
            if not weak:
                break
            samples = [r["title"] for g in weak for r in (g["items"] + g["cands"])]
            new_q = [q for q in strip_sites(await refine_queries(self.llm_url, user_text, queries, samples), sites)
                     if q.lower() not in {x.lower() for x in queries}]
            if not new_q:
                break
            await self.send({"type": "note", "text": "Refining the search: " + " | ".join(new_q[:2])})
            pools = {}
            for g in weak:
                site = next((s for s, _a in targets if (s["name"] if s else label) == g["site"]), None)
                more = await collect(site, new_q[:max_q] if deep else new_q[:2], n, recency, news, lang, label,
                                     budget=budget, deep=deep)
                fresh_rows = [r for r in more["cands"] if norm_url(r["url"]) not in g["seen"]]
                g["seen"].update(norm_url(r["url"]) for r in fresh_rows)
                g["cands"] += fresh_rows
                if fresh_rows and g["status"] != "ok":
                    g["status"], g["provider"], g["error"] = "ok", more["provider"], None
                pools[id(g)] = fresh_rows[:cap]
            queries = queries + new_q
            await self._judge_groups(weak, user_text, queries, n, pools)
        if not any(g["items"] for g in groups):                   # never blank: show the best raw candidates
            for g in groups:
                g["items"] = [dict(r, match="unchecked") for r in g["cands"][:n]]
        if plan["limit"] and plan["limit_scope"] == "total" and len(groups) > 1:
            trim_total(groups, n)
        shown = [g for g in groups if g["items"] or g["status"] == "failed" or not g["auto"]]
        self.last_results = number(shown)
        remember_sources(plan["intent"], shown)      # learn which domains delivered good results
        self.last_groups = shown
        self.last_ctx = {"queries": queries[:max_q], "recency": recency, "news": news, "n": n, "lang": lang,
                         "request": user_text, "deep": deep,
                         "sites": {(s["name"] if s else label): s for s, _a in targets}}
        title = (queries[0] + (f" · {n} per site" if len(sites) > 1 else "")) if plan["display"] == "links" \
            else "Sources"
        await self.send({"type": "links", "title": title, "groups": [public(g) for g in shown]})
        flat = interleave(shown)
        failed = [g["site"] for g in shown if g["status"] == "failed" and not g["items"]]
        if not flat:
            return ("\n\n[The search returned no results" +
                    (" because the search sources failed or were rate limited" if failed else "") +
                    ". Say so in one sentence: the screen shows each site's status and a Retry button.]")
        read_n = DEEP_PAGES if deep else PAGES_TO_READ
        page_of, job_of = {}, {}                 # page text keyed by URL, so read order is free
        if deep or plan["display"] != "links":    # deep research reads pages even for plain link lists
            order = {"good": 0, "partial": 1, "unchecked": 2}
            best = sorted(flat, key=lambda r: order.get(r.get("match", "unchecked"), 2))
            top, seen_r = [], set()
            for r in best:                        # judged-best pages first, one copy of each URL
                k = norm_url(r["url"])
                if k not in seen_r:
                    seen_r.add(k)
                    top.append(r)
                    if len(top) >= read_n:
                        break
            if deep:
                await self.send({"type": "note", "text": f"Deep research: reading {len(top)} pages…"})
            pages = await read_pages(top, total=40 if deep else None, chars=DEEP_PAGE_CHARS if deep else None,
                                     deep=deep)
            page_of = {norm_url(r["url"]): t for r, t in zip(top, pages)}
            if deep and plan.get("intent") == "jobs" and any(pages):
                await self.send({"type": "note", "text": "Deep research: extracting the offers from the pages…"})
                try:
                    job_notes = await extract_job_offers(self.llm_url, top, pages)
                    job_of = {norm_url(r["url"]): t for r, t in zip(top, job_notes)}
                except Exception as e:
                    print("[job extraction failed]", repr(e))

        def blk(i, r):
            """One result for the LLM: header + page text (+ structured offers when extracted)."""
            k = norm_url(r["url"])
            s = describe(r, page_of.get(k, ""))
            return s + job_of.get(k, "")
        from datetime import date
        if plan["display"] == "links":
            found = "\n".join(f"{g['site']}: " + "; ".join(i["title"] for i in g["items"][:4])
                              for g in shown if g["items"])
            body = f"\n\n[{'DEEP RESEARCH' if deep else 'SEARCH'} RESULTS for \"{queries[0]}\"]\n{found}\n"
            if deep:
                body += "\n\n".join(blk(i, r) for i, r in enumerate(top)) + "\n"
            rows = address_rows(flat)
            if rows:
                body += ("\n[EXACT ADDRESSES, in the order shown on screen]\n" + rows +
                         "\n[END ADDRESSES]\n" + OPEN_HINT)
            how = ("The pages of the best results were opened and read: point out the strongest ones and say what "
                   "each really contains (requirements, dates, salary, what the page is about), in a few short "
                   "sentences. " if deep else
                   "In two or three short sentences, say how many results were found and on which sites, mention two "
                   "or three notable ones by title or company, and ")
            return (body + "\n[END]\n\nThe full list with clickable links is already displayed on screen. " + how +
                    "say the links are on screen." +
                    (f" These sites failed and have a Retry button: {', '.join(failed)}." if failed else "") +
                    " Never read URLs aloud.")
        blocks = [blk(i, r) for i, r in enumerate(flat)]
        rows = address_rows(flat, 5)
        addrs = ("\n[EXACT ADDRESSES]\n" + rows + "\n[END ADDRESSES]\n" + OPEN_HINT) if rows else ""
        return (f"\n\n[{'DEEP RESEARCH' if deep else 'WEB SEARCH'} RESULTS, fetched {date.today():%d %B %Y}]\n" +
                "\n\n".join(blocks) + "\n[END OF RESULTS]" + addrs + "\n\nAnswer the message above using these "
                "results. Give the "
                "actual facts, not a list of websites. For news, summarize the 3 or 4 most important stories in one "
                "sentence each, naming the outlet naturally. Never read URLs or result numbers aloud. Keep it "
                "suitable for speech. Only if the results truly lack the answer, say what is missing.")

    async def _followup(self, plan):
        chosen = [r for r in self.last_results if r.get("n") in plan["refs"]][:3]
        if not chosen:
            return ""
        await self.send({"type": "note", "text": "Reading result " + ", ".join(f"#{r['n']}" for r in chosen)})
        pages = await read_pages(chosen)
        blocks = [describe(r, pages[i] if i < len(pages) else "") for i, r in enumerate(chosen)]
        rows = address_rows(chosen)
        addrs = ("\n[EXACT ADDRESSES]\n" + rows + "\n[END ADDRESSES]\n" + OPEN_HINT) if rows else ""
        return ("\n\n[SELECTED RESULTS]\n" + "\n\n".join(blocks) + "\n[END]" + addrs +
                "\n\nAnswer the user's request about these "
                "selected results (summarize, list requirements, compare...). Never read URLs aloud. If a page text "
                "is missing, work from the snippet and say so.")

    def address_block(self, n=5):
        """Addresses from the previous search, so 'open the second one' still works without searching again."""
        rows = address_rows(self.last_results, n)
        if not rows:
            return ""
        return ("\n\n[ADDRESSES FROM THE LAST SEARCH — the results numbered on screen]\n" + rows +
                "\n[END ADDRESSES]" + OPEN_HINT)

    async def retry(self, site_name, lang=None):
        """Re-runs one failed site with the same queries, bypassing the cache, and updates its card."""
        try:
            ctx = self.last_ctx
            old = next((g for g in self.last_groups if g["site"] == site_name), None)
            if not ctx or not old:
                return
            deep = ctx.get("deep", False)
            budget = {"tavily": DEEP_TAVILY if deep else TAVILY_MAX_PER_SEARCH}
            new = await collect(ctx["sites"].get(site_name), ctx["queries"], ctx["n"], ctx["recency"], ctx["news"],
                                lang or ctx["lang"], site_name, fresh=True, budget=budget, deep=deep)
            cap = max(ctx["n"] * 3, 12) if deep else max(ctx["n"] * 2, 8)
            await self._judge_groups([new], ctx["request"], ctx["queries"], ctx["n"], {id(new): new["cands"][:cap]})
            self.last_results = [r for r in self.last_results if r not in old["items"]]
            number([new], max([r.get("n", 0) for r in self.last_results] + [0]) + 1)
            self.last_results += new["items"]
            self.last_groups[self.last_groups.index(old)] = new
            await self.send({"type": "links_update", "group": public(new)})
        except Exception as e:
            print("[retry error]", repr(e))
            await self.send({"type": "error", "text": f"Retry failed: {e}"})