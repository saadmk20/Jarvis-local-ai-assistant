"""Text normalisation, URL identity and the numbering of displayed results."""
import re
import unicodedata
from urllib.parse import parse_qsl, urlencode, urlparse

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


# ---- what the user wants opened, and what each page actually is -------------------------------

PLAY_ACTION = re.compile(r"\b(play|listen|put on|take me|go to|open|show|watch|regarde|ouvre|ouvrir|joue|jouer|"
                         r"ecoute|ecouter|montre|abre|abrir|reproduce|reproducir|escucha|escuchar|muestra|ver)\b", re.I)
QUESTION = re.compile(r"^\s*(who|what|which|when|where|why|how|did|does|do|is|are|was|were|who's|"
                      r"qui|que|quoi|quel|quelle|quand|ou|comment|pourquoi|quien|como|donde|cuando)\b", re.I)
COLLECTION = re.compile(r"\b(playlists?|albums?|hits?|greatest|best of|mix|mixtape|compilation|discography|"
                        r"tous les|toutes les|todas las|all (?:his|her|their) (?:songs?|music|albums?))\b", re.I)
SINGLE = re.compile(r"\b(song|songs?|track|tracks?|single|this one|that one|this song|that song|ce titre|cette chanson|"
                    r"esta cancion|este tema)\b", re.I)
CHANNEL = re.compile(r"\b(channel|chaine|canal|artist|artiste|artista|profile|profil|perfil|account|compte|cuenta)\b", re.I)


def wanted_kind(text):
    """What the user expects to be opened: a collection, one item, a channel, or any page.

    A question that merely mentions an album ('who produced those albums?') is not an action.
    """
    t = fold(text or "")
    if QUESTION.match(t) and not PLAY_ACTION.search(t):
        return None
    if COLLECTION.search(t):
        return "collection"
    if SINGLE.search(t):
        return "item"
    if CHANNEL.search(t):
        return "channel"
    if PLAY_ACTION.search(t):
        return "open"
    return None


# Words that point back at something already on screen rather than naming a new subject.
DEICTIC = re.compile(r"\b(his|her|their|its|that|this|these|those|it|them|the one|the other|the first|the second|"
                     r"the third|the last|same|celui|celle|cet|cette|son|sa|ses|ese|esa|eso|este|esta)\b", re.I)

ACTION_LEAD = re.compile(r"^(?:hey\s+\w+[,\s]+)?(?:please\s+)?(?:can you\s+|could you\s+|would you\s+)?"
                         r"(?:play|put on|listen to|open|show me|show|take me to|go to|watch|find)\s+", re.I)

# A bare follow-up that still asks for a page: 'the second one', 'number 3', 'yes, that one'.
REF_NUM = re.compile(r"\b(?:number|no\.?|num[eé]ro|#)\s*\d{1,2}\b|\b(?:first|second|third|fourth|fifth|sixth|last)\b",
                     re.I)
AFFIRM = re.compile(r"^\s*(?:yes|yeah|yep|sure|ok(?:ay)?|go ahead|do it|please do|that one|oui|vas[- ]y|"
                    r"d'accord|bien s[uû]r|si|dale|vale)\b", re.I)


def refers_to_results(text):
    """The user points at what is on screen without naming an action ('the second one', 'yes')."""
    return bool(REF_NUM.search(text or "") or AFFIRM.match(text or ""))


def action_query(text):
    """'play me some Kanye West music' -> 'Kanye West music': words a search engine can use.

    Used only when the planner called an action request chit-chat, so there are no planned queries.
    """
    q = ACTION_LEAD.sub("", (text or "").strip())
    while True:
        shorter = re.sub(r"^(?:me|us|for me|some|a|an|the)\s+", "", q, flags=re.I)
        if shorter == q:
            break
        q = shorter
    q = re.sub(r"[\s.!,]*(?:please|for me|right now|now)[?.!]*\s*$", "", q, flags=re.I)
    return (q.strip(" .!?") or (text or "").strip(" .!?"))


def can_satisfy(candidates, want):
    """Do the candidates on screen include a page of the kind this request needs?"""
    allowed = OPENABLE.get(want or "") or set()
    return any(c.get("kind") in allowed and c.get("kind") != "search" for c in (candidates or []))


def needs_forced_search(user_text, plan, candidates, named):
    """True when the planner wrongly treated an action as chit-chat.

    'Play Kanye West's greatest hits' cannot be answered without a page, and the planner sometimes
    files it under chit-chat. A follow-up that points at what is already on screen ('open his
    channel') must not trigger a fresh search — but only when the screen really holds that kind.
    """
    want = wanted_kind(user_text)
    if plan.get("search") or named or not want:
        return False
    return not (DEICTIC.search(user_text or "") and can_satisfy(candidates, want))


def stays_local(user_text, candidates):
    """The mirror case: 'open his channel', 'play that one', 'the other one'.

    The page is already on screen, so a search on words like 'his channel' would throw away good
    candidates. If nothing of the wanted kind is there, let the search happen instead.
    """
    want = wanted_kind(user_text)
    return bool(want) and bool(DEICTIC.search(user_text or "")) and can_satisfy(candidates, want)


def page_kind(url):
    """Classify a URL well enough to know whether opening it can satisfy 'play this'."""
    p = urlparse(url or "")
    host = p.netloc.lower().removeprefix("www.")
    path = p.path.rstrip("/")
    q = dict(parse_qsl(p.query))
    if host == "youtu.be":
        return "track"
    if host.endswith("youtube.com"):
        if path == "/playlist" and q.get("list"):
            return "playlist"
        if path == "/watch" and q.get("v"):
            return "track"
        if path in ("/playlist", "/watch") or path.startswith("/results"):
            return "search"                     # missing id / search page: nothing to play
        if path.startswith(("/channel", "/@", "/user", "/c/")):
            return "channel"
        return "page"
    if host.endswith("spotify.com"):
        for part, kind in (("/playlist/", "playlist"), ("/album/", "album"), ("/track/", "track"),
                           ("/episode/", "track"), ("/artist/", "artist")):
            if part in path:
                return kind
    if host.endswith("deezer.com"):
        for part, kind in (("/playlist/", "playlist"), ("/album/", "album"),
                           ("/track/", "track"), ("/artist/", "artist")):
            if part in path:
                return kind
    if host.endswith("music.apple.com"):
        for part, kind in (("/playlist/", "playlist"), ("/album/", "album"),
                           ("/song/", "track"), ("/artist/", "artist")):
            if part in path:
                return kind
    if host.endswith("soundcloud.com"):
        if "/sets/" in path:
            return "playlist"
        return "artist" if path.count("/") == 1 else "track"
    if host.endswith("bandcamp.com"):
        return "album" if path.count("/") == 2 else "artist"
    if re.search(r"/(search|results|find|recherche|buscar)\b", path) or (q.get("q") and path in ("", "/")):
        return "search"
    if re.search(r"/(watch|video|videos|film|movie)/", path + "/"):
        return "video"
    if re.search(r"/(playlist|playlists)/", path + "/"):
        return "playlist"
    if re.search(r"/(album|discography|discographie)/", path + "/"):
        return "album"
    if re.search(r"/(channel|artist|artiste|artista|profile|profil|user|u)/", path + "/"):
        return "profile"
    if re.search(r"/(product|produit|item|dp)/", path + "/"):
        return "product"
    if re.search(r"/(article|post|news|story|blog)/", path + "/"):
        return "article"
    return "page"


OPENABLE = {
    "collection": {"playlist", "album", "track", "video", "page"},
    "item": {"track", "video", "playlist", "album", "page"},
    "open": {"playlist", "album", "track", "video", "page", "article", "product", "channel", "artist", "profile"},
    "channel": {"channel", "artist", "profile"},
}


def kind_rank(kind, want):
    """Which kind best satisfies the request, used when falling back to the next candidate."""
    order = {
        "collection": {"playlist": 0, "album": 0, "video": 1, "track": 2, "page": 3},
        "item": {"track": 0, "video": 0, "album": 1, "playlist": 2, "page": 3},
        "channel": {"channel": 0, "artist": 1, "profile": 2},
    }.get(want or "", {})
    return order.get(kind, 9)


def open_candidates(items, text=None, limit=6):
    """The results the model may open, labelled by kind, filtered by what was asked for.

    A channel/artist/search page can never satisfy 'play his greatest hits', so it is not even
    offered to the model. Numbers stay the ones shown on screen, so 'OPEN: 7' is unambiguous.
    """
    want = wanted_kind(text) if text else None
    allowed = OPENABLE.get(want or "")
    out = []
    for r in items or []:
        u = r.get("url")
        if not u:
            continue
        out.append({"n": r.get("n"), "url": u, "title": (r.get("title") or u)[:120],
                    "kind": page_kind(u), "source": r.get("source") or "",
                    "match": r.get("match") or ""})
    usable = [c for c in out if c["kind"] != "search"] or out   # a dead search page is never worth a tab
    keep = [c for c in usable if not allowed or c["kind"] in allowed]
    if not keep and want != "channel":         # nothing of the wanted type: take the best of the rest
        keep = usable
    return keep[:limit]


def candidate_rows(cands):
    """The labelled list the model reads: number, kind, title, source."""
    return "\n".join(f"{c['n']}. [{c['kind']}] {c['title']}" + (f" — {c['source']}" if c["source"] else "")
                     for c in cands)


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
             "'OPEN: <number>' using the candidate number exactly as shown (for example 'OPEN: 3'). "
             "Prefer a playlist or album when they asked for a collection (hits, mix, all his songs) and a single "
             "track or video when they named one song. Never write a URL, and do not write OPEN for anything else.\n")


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
