"""Pure text helpers: display cleaning, sentence splitting, language guessing, URL identity."""
import re

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


def url_key(u):
    """Loose identity for a URL: scheme, www, fragment and trailing slash do not matter."""
    u = (u or "").split("#")[0].strip().rstrip("/").lower()
    return re.sub(r"^https?://(www\.)?", "", u)


def same_url(a, b):
    a, b = url_key(a), url_key(b)
    if not a or not b:
        return False
    if a == b:
        return True
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) >= 12 and long_.startswith(short)   # a dropped ?query or extra path segment
