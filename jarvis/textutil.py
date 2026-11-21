"""Pure text helpers: display cleaning, sentence splitting, language guessing, URL identity."""
import re
from urllib.parse import parse_qsl, urlparse

from .config import JUNK, settings


def is_junk(text):
    norm = re.sub(r"[\W_]+", " ", text.lower(), flags=re.UNICODE).strip()
    return norm in JUNK


def guess_text_lang(text, last):
    mode = settings["lang_mode"]
    if mode != "auto":
        return mode
    low = text.lower()
    if re.search(r"[ñ¿¡]", low):
        return "es"
    if re.search(r"[çœêèùà]", low):
        return "fr"
    return last


def clean(s):
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    s = re.sub(r"https?://\S+", "link", s)
    s = re.sub(r"[*_#`>|~]", "", s)
    s = re.sub(r"^\s*[-+]\s+", "", s)
    return re.sub(r"\s+", " ", s).strip()


SENT = re.compile(r"(.+?(?:[.!?؟]\s|\n))", re.S)


def pop_sentences(buf):
    out = []
    while True:
        m = SENT.match(buf)
        if not m:
            break
        out.append(m.group(1))
        buf = buf[m.end():]
    return out, buf


# Params that never change which page is shown: asking for a timestamp must not break the match.
HARMLESS_PARAMS = frozenset({
    "t", "start", "si", "feature", "ref", "fbclid", "gclid",
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
})


def url_key(u):
    """Loose identity for a URL: scheme, www, fragment and trailing slash do not matter.
    Host is lower-cased; path and query keep their case, because ids such as YouTube's
    `v=` and `list=` are case-sensitive."""
    u = (u or "").split("#")[0].strip().rstrip("/")
    m = re.match(r"^https?://([^/]+)(/.*)?$", u, re.I)
    if not m:
        return u
    return m.group(1).lower().removeprefix("www.") + (m.group(2) or "")


def _target(u):
    p = urlparse(u or "")
    host = p.netloc.lower().removeprefix("www.")
    path = p.path.rstrip("/")
    q = tuple(sorted((k, v) for k, v in parse_qsl(p.query) if k not in HARMLESS_PARAMS))
    return host, path, q


def same_target(typed, known):
    """The model's address must point at exactly the page we know, not a truncation of it.

    `https://youtube.com/playlist` must NOT match `…/playlist?list=PL1` (that tab would open
    YouTube's generic page), while adding `&t=30` to a known video still matches.
    """
    a, b = _target(typed), _target(known)
    return bool(a[0]) and a == b


def content_key(u):
    """Identity of the actual media, so youtube.com and music.youtube.com copies collapse."""
    q = dict(parse_qsl(urlparse(u or "").query))
    if q.get("list"):
        return "list:" + q["list"]
    if q.get("v"):
        return "video:" + q["v"]
    return "url:" + url_key(u)
