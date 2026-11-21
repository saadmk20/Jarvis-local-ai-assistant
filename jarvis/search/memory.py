"""Experience memory: which domains actually delivered for a kind of question."""
import json
from urllib.parse import urlparse

from .config import ROOT
from .sites import KNOWN, _save_known


MEMORY_FILE = ROOT / "source_memory.json"


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
