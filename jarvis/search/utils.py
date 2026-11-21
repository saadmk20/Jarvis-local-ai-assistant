"""Text normalisation, URL identity and the numbering of displayed results."""
import difflib
import re
import unicodedata
from urllib.parse import parse_qsl, quote_plus, urlencode, urlparse, urlsplit

import httpx


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
