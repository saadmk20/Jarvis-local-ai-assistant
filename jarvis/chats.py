"""Saved conversations: the on-disk store, chat summaries and rich -> LLM conversion."""
import json
import secrets

from .config import CHATS_FILE, SYSTEM_PROMPT


CHATS = {}                     # id -> {"title": str, "created": ts, "messages": [rich msg...]}


def load_chats():
    try:
        if CHATS_FILE.is_file():
            d = json.loads(CHATS_FILE.read_text(encoding="utf-8"))
            for c in d:
                if isinstance(c, dict) and isinstance(c.get("messages"), list):
                    if not c["messages"]:
                        continue                     # empty chats from older versions are dropped
                    cid = c.get("id") or new_chat_id()   # tolerate records saved without an id
                    c["id"] = cid
                    CHATS[cid] = c
    except Exception as e:
        print("[chats.json not loaded]", e)


def save_chats():
    try:
        for cid, c in CHATS.items():
            c.setdefault("id", cid)          # the id is embedded in every saved record
        data = sorted((c for c in CHATS.values() if c.get("messages")),
                      key=lambda c: c.get("created", 0))[-100:]
        CHATS_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print("[chats.json not saved]", e)


def new_chat_id():
    return secrets.token_hex(6)


def chats_msg():
    lst = sorted((c for c in CHATS.values() if c.get("messages")),
                 key=lambda c: c.get("created", 0), reverse=True)
    return {"type": "chats", "chats": [{"id": c.get("id", ""), "title": c.get("title", "")} for c in lst]}


def rich_to_llm(msgs):
    """Stored rich messages -> the LLM-facing history (system prompt + content strings)."""
    h = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in msgs:
        e = {"role": m["role"], "content": m["text"] + (" [interrupted by the user]" if m.get("cut") else "")}
        if m.get("images"):
            e["images"] = m["images"]
        h.append(e)
    return h


def llm_message(e):
    """One history entry -> the chat.completions message (multimodal when images are attached)."""
    if not e.get("images"):
        return {"role": e["role"], "content": e["content"]}
    parts = [{"type": "text", "text": e["content"]}]
    parts += [{"type": "image_url", "image_url": {"url": u}} for u in e["images"]]
    return {"role": e["role"], "content": parts}
