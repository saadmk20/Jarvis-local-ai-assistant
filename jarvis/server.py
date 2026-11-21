"""FastAPI app: token gate, index page, websocket endpoint and the launcher."""
import asyncio
import json
import os
import secrets
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from faster_whisper import WhisperModel

from . import speech
from .chats import CHATS, load_chats
from .config import (HERE, LANGS, MAX_LIMIT, STT_MODEL, VOICES_DIR, WEB_PORT,
                     load_settings, save_settings, settings)
from .hub import broadcast, sessions
from .llm import list_models, llama, startup_llama
from .session import Session
from .speech import get_voice, list_voices, voices_msg


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(startup_llama())
    yield
    task.cancel()
    if llama.owned:
        await llama.stop_proc()


app = FastAPI(lifespan=lifespan)


def _load_token():
    """Access key: JARVIS_TOKEN env var if set, else token.txt in the project folder
    (created automatically on first run with a random value)."""
    tok = os.environ.get("JARVIS_TOKEN", "").strip()
    if tok:
        return tok, "JARVIS_TOKEN env var"
    f = HERE / "token.txt"
    try:
        if f.is_file():
            tok = f.read_text(encoding="utf-8").strip()
            if tok:
                return tok, f"token.txt"
        tok = secrets.token_urlsafe(24)
        f.write_text(tok + "\n", encoding="utf-8")
        return tok, f"generated, saved to {f.name}"
    except OSError as e:
        print("[token.txt not writable]", e)
        return "", "unavailable"


TOKEN, TOKEN_FROM = _load_token()


def _match(k: str) -> bool:
    return bool(TOKEN) and bool(k) and secrets.compare_digest(k.encode(), TOKEN.encode())


@app.get("/")
async def index(request: Request):
    if _match(request.query_params.get("key")):          # first visit with the key: remember it for 90 days
        proto = request.headers.get("x-forwarded-proto") or request.url.scheme
        resp = FileResponse(HERE / "Index.html")
        resp.set_cookie("jarvis_key", request.query_params.get("key"),
                        max_age=90 * 24 * 3600, samesite="lax", secure=(proto == "https"))
        return resp
    if _match(request.cookies.get("jarvis_key", "")):
        return FileResponse(HERE / "Index.html")
    return HTMLResponse(
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>JARVIS &mdash; access</title><style>"
        "body{margin:0;min-height:100vh;display:flex;flex-direction:column;gap:16px;align-items:center;"
        "justify-content:center;background:radial-gradient(1100px 650px at 50% -10%,#1b2244 0%,#0b0d14 62%);"
        "color:#e8eaf2;font:15px system-ui,'Segoe UI',sans-serif;text-align:center;padding:20px}"
        "h1{margin:0;letter-spacing:.32em;font-weight:700;font-size:26px;"
        "background:linear-gradient(90deg,#8ab4ff,#d29bff);-webkit-background-clip:text;background-clip:text;color:transparent}"
        "input{background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.16);color:inherit;border-radius:10px;"
        "padding:11px 14px;font:inherit;outline:none;width:min(320px,80vw);text-align:center}"
        "input:focus{border-color:#8ab4ff}"
        "button{margin-top:10px;padding:11px 30px;border-radius:99px;border:0;color:#fff;"
        "background:linear-gradient(135deg,#5b9dff,#a78bfa);font:inherit;cursor:pointer}"
        "p{margin:0;color:#8a90a6;font-size:13px}</style></head><body>"
        "<h1>JARVIS</h1>"
        "<form method='get' action='/'>"
        "<input type='password' name='key' placeholder='Access key' autofocus autocomplete='off'>"
        "<br><button>Enter</button></form>"
        "<p>This assistant is private. The key is remembered by this browser afterwards.</p>"
        "</body></html>", status_code=401)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    if not _match(ws.query_params.get("key")) and not _match(ws.cookies.get("jarvis_key", "")):
        await ws.close(code=4401)          # missing/invalid key: refuse the connection
        return
    await ws.accept()
    sess = Session(ws)
    sessions.add(sess)
    await sess.send({"type": "hello", "models": list_models(),
                     "reasoning_options": ["auto", "off", "low", "medium", "high"],
                     "langs": [{"code": c, "name": v["name"]} for c, v in LANGS.items()]})
    await sess.send(voices_msg())
    await sess.send(llama.status_msg())
    await sess.send(sess.search.prefs())
    latest = sorted(CHATS.values(), key=lambda c: c.get("created", 0))
    await sess.open_chat(latest[-1]["id"] if latest else None, create=not latest)
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes"):
                await sess.start_turn(sess.audio_turn(msg["bytes"]))
            elif msg.get("text"):
                d = json.loads(msg["text"])
                t = d.get("type")
                if t == "interrupt":
                    await sess.stop_turn()
                elif t == "text" and d.get("text", "").strip():
                    await sess.start_turn(sess.text_turn(d["text"].strip()))
                elif t == "clear":
                    await sess.stop_turn()
                    await sess.open_chat(None, create=True)
                elif t == "chat_open":
                    await sess.open_chat(d.get("id"))
                elif t == "chat_new":
                    await sess.open_chat(None, create=True)
                elif t == "chat_del":
                    await sess.delete_chat(d.get("id"))
                elif t == "image":
                    uri = d.get("data") or ""
                    if uri.startswith("data:image/") and len(uri) < 6_000_000:
                        if len(sess.pending_images) < 4:
                            sess.pending_images.append(uri)
                            await sess.send({"type": "images_pending", "count": len(sess.pending_images)})
                        else:
                            await sess.send({"type": "error", "text": "Up to 4 images per message."})
                    else:
                        await sess.send({"type": "error", "text": "Image rejected (too large or not an image)."})
                elif t == "set_lang" and d.get("lang") in ("auto", *LANGS):
                    settings["lang_mode"] = d["lang"]
                    save_settings()
                    await broadcast(voices_msg())
                elif t == "set_voice" and d.get("lang") in LANGS:
                    lang, vid = d["lang"], d.get("voice")
                    if vid in [v["id"] for v in list_voices()[lang]]:
                        settings["voices"][lang] = vid
                        save_settings()
                        await broadcast(voices_msg())
                        await sess.start_turn(sess.preview(lang))     # plays a short sample in the new voice
                elif t == "set_limit":
                    try:
                        sess.search.limit = settings["search_limit"] = max(1, min(MAX_LIMIT, int(d.get("limit"))))
                        save_settings()
                    except (TypeError, ValueError):
                        pass
                    await sess.send(sess.search.prefs())
                elif t == "search_retry":
                    asyncio.create_task(sess.search.retry(d.get("site"), sess.last_lang))
                elif t == "set_web":
                    sess.web = bool(d.get("on"))
                elif t == "set_deep":
                    sess.deep = bool(d.get("on"))
                    if sess.deep:
                        sess.web = True          # deep research implies web search
                elif t == "rescan":
                    await broadcast(voices_msg())
                elif t == "set_model":
                    if not llama.owned:
                        await sess.send({"type": "error", "text": "Model switching is off because llama-server "
                                                                  "was already running. Close it and restart this app."})
                    elif d.get("model") in [m["name"] for m in list_models()]:
                        await sess.stop_turn()
                        asyncio.create_task(llama.start(d["model"], d.get("reasoning", "auto")))
    except WebSocketDisconnect:
        pass
    finally:
        sessions.discard(sess)
        await sess.stop_turn()


# ------------------------------------------------------------------ launcher
def run():
    """Warm everything up, then serve the UI on 127.0.0.1 (a tunnel exposes it to other devices)."""
    VOICES_DIR.mkdir(exist_ok=True)
    load_settings()               # needs the voice catalog, so it runs after the package is importable
    load_chats()
    print(f"Loading speech recognition ({STT_MODEL})...")
    speech.stt = WhisperModel(STT_MODEL, device="cpu", compute_type="int8")
    try:
        get_voice("en", settings["voices"]["en"])     # warm up English so the first answer is quick
    except Exception as e:
        print("[could not load the English voice]", repr(e))
    print(f"Ready. On this PC:   http://127.0.0.1:{WEB_PORT}   (Cloudflare tunnel for other devices)")
    if TOKEN:
        print(f"Access key ({TOKEN_FROM}): {TOKEN}")
    uvicorn.run(app, host="127.0.0.1", port=WEB_PORT, log_level="warning")


if __name__ == "__main__":
    run()
