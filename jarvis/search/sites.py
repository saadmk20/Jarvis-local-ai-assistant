"""Named-site detection and resolution: "on LinkedIn and ReKrute" -> domains to search."""
import difflib
import json
import re
from urllib.parse import urlparse

from .config import ALIAS_MAP, ROOT, SITES
from .engines import run_search
from .utils import fold, word_in


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


KNOWN_FILE = ROOT / "known_sites.json"    # learned name -> domain; you can edit it by hand


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
