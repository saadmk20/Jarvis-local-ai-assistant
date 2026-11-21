"""JARVIS voice assistant (multilingual): web UI + Whisper (CPU) + llama-server + Pocket TTS + web search.

Run:  python voice_chat.py   then open http://127.0.0.1:7860
Needs: pip install ddgs trafilatura   (websearch.py must sit next to this file; see its header for
SearXNG / Brave / Tavily setup)
Keep index.html next to this file. Put your voices in the "voices" folder next to it.
"""
import asyncio
import json
import os
import re
import secrets
import struct
import subprocess
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from faster_whisper import WhisperModel
from pocket_tts import TTSModel, export_model_state
from websearch import DEFAULT_LIMIT, MAX_LIMIT, WebSearch   # web search engine: see websearch.py

# ------------------------------------------------------------------ settings
HERE = Path(__file__).parent
VOICES_DIR = HERE / "voices"
SETTINGS_FILE = HERE / "app_settings.json"
LLAMA_EXE = Path(r"F:\AI\Prism\llama-server.exe")
MODELS_DIR = Path(r"F:\AI\Models")
MMPROJ = Path(r"F:\AI\Models\mmproj-Q8_0.gguf")     # vision projector: images in chat when present
CHATS_FILE = HERE / "chats.json"                     # saved conversations (multiple chats)
DEFAULT_MODEL = "Ternary-Bonsai-2-27B-Uncensored-Heretic-v2-PTQ1_0.gguf"  # falls back to the first .gguf found
DEFAULT_REASONING = "medium"        # auto | off | low | medium | high
CTX = 49152
MAX_HISTORY = 61                     # messages kept in a session (system + 30 turns);
                                     # older turns are dropped so the context never silently overflows
LLM_BASE = "http://127.0.0.1:8080"
LLM_URL = LLM_BASE + "/v1/chat/completions"
WEB_PORT = 7860
MIN_CHARS = 30                      # merge very short sentences before speaking
STT_MODEL = "small"                 # multilingual. "medium" is better for Arabic but slower on CPU
VRAM_WARN_GB = 7.3                  # models bigger than this are flagged in the UI
DEFAULT_VOICE_AUDIO = "hf://kyutai/tts-voices/alba-mackenna/casual.wav"   # voice used by models without a catalog
SYSTEM_PROMPT = (
    "You are JARVIS, the calm, dry-witted, impeccably polite AI butler of a brilliant engineer — in the spirit "
    "of Tony Stark's JARVIS: unflappable, precise, understated, quietly confident. You are never condescending, "
    "never sycophantic, and you never ramble. "
    "Address the user the way a trusted butler would, using the form of address of the language you are replying "
    "in: 'Sir' in English, 'Monsieur' in French, 'Señor' in Spanish. Never use another "
    "language's form of address inside a reply — in particular, never say 'Sir' in a French sentence. "
    "Answer for speech: one to three short sentences by default, no bullet lists, no markdown, no URLs read "
    "aloud. A brief acknowledgement ('Very good, Sir.', 'Bien sûr, Monsieur.') is welcome when it fits, but do "
    "not repeat the user's question back to them. Dry humour is allowed; flattery is not. If you do not know "
    "something, say so plainly and offer the next useful step. "
    "You are also an expert coding assistant: when you write code, put it in code blocks and explain it briefly "
    "in plain sentences. "
    "You can open a web page in the user's own browser. When the user asks you to open, play, show or take them "
    "to something — or when they are asking for something to watch, listen to, read or visit (a playlist, a video, "
    "a track, an article, a page, a product) rather than for an answer to a question — and you have the exact "
    "address of the best match, finish your reply with one extra line, on its own line, of the form "
    "OPEN: https://... . Copy the address verbatim, put no punctuation after it, and never wrap it in backticks. "
    "That line is removed automatically before the reply is shown or spoken, so do not mention it or read it "
    "aloud. For purely factual questions, open nothing. Never guess an address: if you do not have the exact one, "
    "say so instead. Do not write the word OPEN for anything else."
)

# One entry per language. "tts" is passed to TTSModel.load_model(...).
# Add "_24l" to a language name (e.g. "french_24l") for the bigger, higher-quality but slower variant.
LANGS = {
    "en": {"name": "English", "tts": {"language": "english"},
           "prompt": "Always reply in English. Address the user only as 'Sir'.",
           "say_code": "Code block skipped.",
           "sample": "Good evening. How can I help you today?"},
    "fr": {"name": "Français", "tts": {"language": "french"},
           "prompt": ("Always reply in French. Address the user as 'Monsieur' — never 'Sir', never "
                      "'Monsieur Sir'. Use 'vous', never 'tu'."),
           "say_code": "Bloc de code ignoré.",
           "sample": "Bonsoir. Comment puis-je vous aider ?"},
    "es": {"name": "Español", "tts": {"language": "spanish"},
           "prompt": ("Always reply in Spanish. Address the user as 'Señor' — never 'Sir'. "
                      "Use 'usted', never 'tú'."),
           "say_code": "Bloque de código omitido.",
           "sample": "Buenas noches. ¿En qué puedo ayudarle?"},
}
BUILTIN = {
    "en": ["alba", "anna", "azelma", "bill_boerst", "caro_davy", "charles", "cosette", "eponine", "eve",
           "fantine", "george", "jane", "jean", "javert", "marius", "mary", "michael", "paul",
           "peter_yearsley", "stuart_bell", "vera"],
    "fr": ["estelle"],
    "es": ["lola"],
}
# Whisper sometimes invents these on silence or noise
JUNK = {"", "you", "thank you", "thanks for watching", "bye", "merci", "merci davoir regarde cette video",
        "gracias"}

stt = None
engines = {}          # language -> TTSModel (loaded on first use)
voice_states = {}     # (language, voice id) -> voice state
tts_pool = ThreadPoolExecutor(max_workers=1)
stt_pool = ThreadPoolExecutor(max_workers=1)
sessions = set()
settings = {"lang_mode": "auto", "voices": {}, "search_limit": DEFAULT_LIMIT}


# ------------------------------------------------------------------ voices and languages
def list_voices():
    """Voices offered for each language. Ids: builtin:<name>, wav:<file>, st:<file>, default."""
    out = {l: [] for l in LANGS}
    for l in LANGS:
        for b in BUILTIN.get(l, []):
            out[l].append({"id": f"builtin:{b}", "label": b})
    for p in sorted(VOICES_DIR.glob("*.wav")):               # recordings: any language can clone these
        for l in LANGS:
            out[l].append({"id": f"wav:{p.name}", "label": f"{p.stem} (clone from wav)"})
    for p in sorted(VOICES_DIR.glob("*.safetensors")):       # saved voices belong to one language
        stem, lang = p.stem, "en"
        m = re.match(r"(.+)\.(en|fr|es)$", stem)
        if m:
            stem, lang = m.group(1), m.group(2)
        out[lang].append({"id": f"st:{p.name}", "label": f"{stem} (saved voice)"})
    return out


def default_voice(lang):
    if lang == "en":
        return "st:jarvis.safetensors" if (VOICES_DIR / "jarvis.safetensors").exists() else "builtin:alba"
    return {"fr": "builtin:estelle", "es": "builtin:lola"}.get(lang, "builtin:alba")


def load_settings():
    try:
        saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        saved = {}
    mode = saved.get("lang_mode", "auto")
    settings["lang_mode"] = mode if mode in ("auto", *LANGS) else "auto"
    try:
        settings["search_limit"] = max(1, min(MAX_LIMIT, int(saved.get("search_limit", DEFAULT_LIMIT))))
    except (TypeError, ValueError):
        settings["search_limit"] = DEFAULT_LIMIT
    ids = {l: {v["id"] for v in vs} for l, vs in list_voices().items()}
    for l in LANGS:
        v = saved.get("voices", {}).get(l)
        settings["voices"][l] = v if v in ids[l] else default_voice(l)


def save_settings():
    try:
        SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    except Exception as e:
        print("[settings not saved]", e)


def get_engine(lang):
    if lang not in engines:
        print(f"Loading {LANGS[lang]['name']} voice model (first use may download it)...")
        try:
            engines[lang] = TTSModel.load_model(**LANGS[lang]["tts"])
        except TypeError:
            if lang != "en":
                raise
            engines[lang] = TTSModel.load_model()
    return engines[lang]


def get_voice(lang, vid):
    key = (lang, vid)
    if key in voice_states:
        return voice_states[key]
    eng = get_engine(lang)
    kind, _, name = vid.partition(":")
    if kind == "builtin":
        state = eng.get_state_for_audio_prompt(name)
    elif kind == "st":
        state = eng.get_state_for_audio_prompt(str(VOICES_DIR / name))
    elif kind == "wav":
        wav = VOICES_DIR / name
        cache_dir = VOICES_DIR / "cache"
        cache_dir.mkdir(exist_ok=True)
        cache = cache_dir / f"{wav.stem}.{lang}.safetensors"
        if cache.exists() and cache.stat().st_mtime >= wav.stat().st_mtime:
            state = eng.get_state_for_audio_prompt(str(cache))
        else:
            state = eng.get_state_for_audio_prompt(str(wav))
            try:
                export_model_state(state, str(cache))     # next start loads this instantly
            except Exception as e:
                print("[voice cache not saved]", e)
    else:
        state = eng.get_state_for_audio_prompt(DEFAULT_VOICE_AUDIO)
    voice_states[key] = state
    return state


def synth(lang, vid, text):
    """Runs in the TTS thread. Returns (sample_rate, int16 pcm bytes)."""
    eng = get_engine(lang)
    if not text:
        return eng.sample_rate, b""
    audio = eng.generate_audio(get_voice(lang, vid), text).numpy()
    return eng.sample_rate, (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()


# ------------------------------------------------------------------ speech recognition
def transcribe(pcm: bytes, mode: str):
    """Returns (text, language). mode is 'auto' or one of the LANGS keys."""
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

    def run(language):
        segs, info = stt.transcribe(a, language=language, beam_size=1, vad_filter=True)
        return " ".join(s.text.strip() for s in segs).strip(), info

    if mode != "auto":
        return run(mode)[0], mode
    text, info = run(None)
    lang = info.language
    if lang not in LANGS:    # a detected language we have no voice for: fall back to the closest supported one
        probs = dict(getattr(info, "all_language_probs", None) or [])
        scores = {l: probs.get(l, 0.0) for l in LANGS}
        lang = max(scores, key=scores.get) if any(scores.values()) else "en"
        text = run(lang)[0]
    return text, lang


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


# ------------------------------------------------------------------ helpers
def list_models():
    out = []
    for p in sorted(MODELS_DIR.glob("*.gguf")):
        n = p.name.lower()
        if "mmproj" in n or re.search(r"-0000[2-9]-of-", n):
            continue
        gb = round(p.stat().st_size / 1e9, 1)
        out.append({"name": p.name, "gb": gb, "big": gb > VRAM_WARN_GB})
    return out


def default_model():
    names = [m["name"] for m in list_models()]
    if DEFAULT_MODEL in names:
        return DEFAULT_MODEL
    return names[0] if names else None


def reasoning_flags(level):
    if level == "off":
        return ["--reasoning", "off"]
    if level in ("low", "medium", "high"):
        return ["--reasoning", "on", "--reasoning-effort", level]
    return []


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


def voices_msg():
    return {"type": "voices", "voices": list_voices(), "selected": settings["voices"],
            "lang_mode": settings["lang_mode"]}


async def broadcast(obj):
    for s in list(sessions):
        try:
            await s.send(obj)
        except Exception:
            pass


# ------------------------------------------------------------------ llama-server manager
class Llama:
    def __init__(self):
        self.proc = None
        self.owned = False
        self.model = None
        self.reasoning = DEFAULT_REASONING
        self.vision = False        # mmproj was loaded -> image messages are accepted
        self.status = "stopped"
        self.msg = ""
        self.lock = asyncio.Lock()

    def status_msg(self):
        return {"type": "llm_status", "status": self.status, "msg": self.msg,
                "model": self.model, "reasoning": self.reasoning, "owned": self.owned,
                "vision": self.vision}

    async def _set(self, status, msg=""):
        self.status, self.msg = status, msg
        await broadcast(self.status_msg())

    async def health(self):
        try:
            async with httpx.AsyncClient(timeout=2) as c:
                return (await c.get(LLM_BASE + "/health")).status_code
        except Exception:
            return None

    async def stop_proc(self):
        p = self.proc
        if p and p.poll() is None:
            p.terminate()
            try:
                await asyncio.to_thread(p.wait, 20)
            except subprocess.TimeoutExpired:
                p.kill()
        self.proc = None

    async def start(self, model, reasoning):
        async with self.lock:
            path = MODELS_DIR / model
            if not path.is_file():
                await self._set("error", f"Model file not found: {model}")
                return
            await self.stop_proc()
            self.model, self.reasoning = model, reasoning
            self.vision = MMPROJ.is_file()
            await self._set("loading", f"Loading {model}…")
            cmd = [str(LLAMA_EXE), "-m", str(path), "-ngl", "99", "-c", str(CTX),
                   "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--jinja", "-np", "1",
                   *reasoning_flags(reasoning), "--host", "127.0.0.1", "--port", "8080"]
            if self.vision:
                cmd += ["--mmproj", str(MMPROJ)]     # multimodal: image messages in chat
                print(f"[vision] mmproj loaded: {MMPROJ.name}")
            else:
                print(f"[vision] mmproj not found at {MMPROJ} -> images disabled")
            try:
                self.proc = subprocess.Popen(cmd, cwd=str(LLAMA_EXE.parent))
            except OSError as e:
                await self._set("error", f"Could not start llama-server: {e}")
                return
            self.owned = True
            t0 = time.time()
            while time.time() - t0 < 300:
                if self.proc.poll() is not None:
                    await self._set("error", f"llama-server stopped (exit code {self.proc.returncode}). "
                                             "Check the console for the reason.")
                    return
                if await self.health() == 200:
                    await self._set("ready", model)
                    return
                await asyncio.sleep(1)
            await self._set("error", "Timed out waiting for llama-server to load the model.")


llama = Llama()


async def startup_llama():
    if await llama.health() is not None:
        llama.owned = False
        await llama._set("ready", "Using the llama-server that is already running on port 8080. "
                                  "Close it and restart this app to enable model switching.")
        return
    m = default_model()
    if not m:
        await llama._set("error", f"No .gguf files found in {MODELS_DIR}")
        return
    await llama.start(m, DEFAULT_REASONING)


# ------------------------------------------------------------------ acting on a link
OPEN_MARK = "OPEN:"


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


class OpenDirective:
    """Strips `OPEN: <url>` out of the answer before it reaches the screen or the voice.

    The model is told to put that line on its own line, but text arrives in small pieces, so this
    holds back only the few characters that could still grow into the marker.
    """

    def __init__(self):
        self.hold = ""        # characters that may still become the marker
        self.at_start = True  # we are at the beginning of a line
        self.in_dir = False   # inside a directive line
        self.confirmed = False
        self.dir = ""
        self.urls = []

    def feed(self, piece):
        out = []
        for ch in piece:
            if self.in_dir:
                if not self.confirmed:            # "OPEN:" alone proves nothing: wait for the address
                    if ch in " \t":
                        continue
                    if ch in "hH\"'<":            # http://  "http…  <http…
                        self.confirmed = True
                        self.dir += ch
                        continue
                    out.append("OPEN: " + ch)     # it was prose after all: give the line back
                    self.dir, self.in_dir, self.at_start = "", False, ch == "\n"
                    continue
                if ch == "\n":
                    self._close()
                    out.append("\n")          # keep the sentence after the directive from gluing to the one before
                else:
                    self.dir += ch
                continue
            if self.at_start:
                self.hold += ch
                probe = self.hold.lstrip(" \t->*•").upper()
                if OPEN_MARK.startswith(probe):
                    if probe == OPEN_MARK:      # the marker is complete: swallow the rest of the line
                        self.in_dir, self.dir, self.hold, self.at_start = True, "", "", False
                        self.confirmed = False
                    continue
                out.append(self.hold)           # ordinary sentence after all: release what we held
                self.hold = ""
                self.at_start = ch == "\n"
                continue
            out.append(ch)
            self.at_start = ch == "\n"
        return "".join(out), self._take()

    def _close(self):
        m = re.search(r"https?://[^\s<>\"'`]+", self.dir, re.I)
        if m:
            self.urls.append(m.group(0).rstrip(").,;:!?"))
        self.dir = ""
        self.in_dir = False
        self.confirmed = False
        self.at_start = True

    def _take(self):
        got = self.urls
        self.urls = []
        return got

    def finish(self):
        """End of stream: close a dangling directive and release anything still held."""
        if self.in_dir and self.confirmed:
            self._close()
        elif self.in_dir:                     # a bare "OPEN:" with nothing after it: drop it
            self.dir, self.in_dir, self.confirmed = "", False, False
        rest = self.hold
        self.hold = ""
        return rest, self._take()


# ------------------------------------------------------------------ speaking one turn
class Speaker:
    """Cuts the streamed answer into sentences, skips code blocks, synthesizes and streams audio."""

    def __init__(self, sess, epoch, lang):
        self.sess, self.epoch, self.lang = sess, epoch, lang
        self.vid = settings["voices"][lang]
        self.q = asyncio.Queue()
        self.pending = ""
        self.buf = ""
        self.in_code = False
        self.failed = False

    def _say(self, s, flush=False):
        s = clean(s)
        if s:
            self.pending = (self.pending + " " + s).strip()
        if self.pending and (flush or len(self.pending) >= MIN_CHARS):
            self.q.put_nowait(self.pending)
            self.pending = ""

    def text(self, piece):
        self.buf += piece
        while True:
            if self.in_code:
                i = self.buf.find("```")
                if i == -1:
                    self.buf = self.buf[-2:]
                    break
                self.buf = self.buf[i + 3:]
                self.in_code = False
            else:
                i = self.buf.find("```")
                if i != -1:
                    self._say(self.buf[:i])
                    self._say(LANGS[self.lang]["say_code"], flush=True)
                    self.buf = self.buf[i + 3:]
                    self.in_code = True
                    continue
                sents, self.buf = pop_sentences(self.buf)
                for s in sents:
                    self._say(s)
                break

    def finish(self):
        if not self.in_code:
            self._say(self.buf)
        self._say("", flush=True)
        self.q.put_nowait(None)

    async def run(self):
        loop = asyncio.get_running_loop()
        first = True
        while True:
            s = await self.q.get()
            if s is None:
                return
            if self.failed:
                continue
            try:
                if first and (self.lang not in engines or (self.lang, self.vid) not in voice_states):
                    await self.sess.send({"type": "note",
                                          "text": f"Loading the {LANGS[self.lang]['name']} voice (first use is slow)…"})
                first = False
                sr, data = await loop.run_in_executor(tts_pool, synth, self.lang, self.vid, s)
            except Exception as e:
                self.failed = True
                print("[tts error]", repr(e))
                await self.sess.send({"type": "error",
                                      "text": f"Voice failed for {LANGS[self.lang]['name']}: {e}. "
                                              "The text is still shown; pick another voice or check the console."})
                continue
            if data:
                await self.sess.send_bytes(struct.pack("<II", self.epoch, sr) + data)


# ------------------------------------------------------------------ multiple saved chats
CHATS = {}                     # id -> {"title": str, "created": ts, "messages": [rich msg...]}
# rich message: {"role": "user"|"assistant", "text": str, "images": [data URI...], "cut": bool}


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


# ------------------------------------------------------------------ one browser connection
class Session:
    def __init__(self, ws):
        self.ws = ws
        self.chat_id = None          # current saved chat
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.pending_images = []     # data URIs attached by the user, sent with the next message
        self.epoch = 0
        self.turn = None
        self.last_lang = "en"
        self.web = False
        self.deep = False          # deep research: slower but far more thorough searches
        self.search = WebSearch(self.send, LLM_URL, settings["search_limit"])
        self.send_lock = asyncio.Lock()
        self.links = None            # source cards shown during this turn (site -> group), saved with the reply
        self.links_title = ""

    async def send(self, obj):
        kind = obj.get("type")
        if kind == "links":
            self._track_links(obj)
        elif kind == "links_update":
            self._track_group(obj.get("group"))
        async with self.send_lock:
            await self.ws.send_text(json.dumps(obj))

    def _track_links(self, m):
        """Keep the source cards of this turn so they survive switching chats."""
        if self.links is None:
            self.links = {}
        for g in m.get("groups") or []:
            self.links[g.get("site") or ""] = g
        if m.get("title"):
            self.links_title = m["title"]

    def _track_group(self, g):
        """A Retry replaced one group. Keep the saved copy in step, inside a turn or after it."""
        if not isinstance(g, dict):
            return
        if self.links is not None:
            self.links[g.get("site") or ""] = g
            return
        chat = CHATS.get(self.chat_id) or {}          # retry while idle: patch the last reply showing it
        for msg in reversed(chat.get("messages") or []):
            if msg.get("role") != "assistant" or not msg.get("links"):
                continue
            groups = msg["links"].setdefault("groups", [])
            for i, old in enumerate(groups):
                if old.get("site") == g.get("site"):
                    groups[i] = g
                    break
            else:
                groups.append(g)
            save_chats()
            return

    async def send_bytes(self, b):
        async with self.send_lock:
            await self.ws.send_bytes(b)

    # ---- multiple chats -------------------------------------------------
    async def chats_update(self):
        await self.send(chats_msg())

    async def open_chat(self, cid, create=False):
        """Switch to a saved chat (or start an unsaved draft) and send its transcript to the UI.
        A draft is written to chats.json only when its first message arrives, so deleting chats and
        clicking 'New chat' never leaves empty 'Chat' entries behind in the sidebar."""
        await self.stop_turn()
        self.search.reset()
        if create or cid not in CHATS:
            self.chat_id = None                  # draft: nothing is stored until the first message
            self.history = rich_to_llm([])
            await self.chats_update()
            await self.send({"type": "chat_load", "id": None, "title": "", "messages": [],
                             "usage": {"prompt": 0, "gen": 0, "limit": CTX, "msgs": 0}})
            return
        self.chat_id = cid
        msgs = CHATS[cid]["messages"]
        self.history = rich_to_llm(msgs)
        await self.chats_update()
        # real token counts when this chat was used before; else a chars÷4 estimate
        u = CHATS[cid].get("usage")
        if not u:
            u = {"prompt": (sum(len(m.get("text") or "") for m in msgs)
                            + 800 * sum(len(m.get("images", [])) for m in msgs)   # ~vision tokens per image
                            + len(SYSTEM_PROMPT)) // 4, "gen": 0}
        await self.send({"type": "chat_load", "id": cid, "title": CHATS[cid]["title"],
                         "messages": [{"role": m["role"], "text": m["text"],
                                       "images": m.get("images", []), "cut": m.get("cut", False),
                                       "links": m.get("links")}
                                      for m in msgs],
                         "usage": {"prompt": u.get("prompt", 0), "gen": u.get("gen", 0),
                                   "limit": CTX, "msgs": len(msgs)}})

    async def delete_chat(self, cid):
        if cid in CHATS:
            del CHATS[cid]
            save_chats()
        if self.chat_id == cid:
            await self.open_chat(None, create=True)
        else:
            await self.chats_update()

    async def stop_turn(self):
        t = self.turn
        self.turn = None
        if t and not t.done():
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

    async def start_turn(self, coro):
        await self.stop_turn()
        self.turn = asyncio.create_task(coro)

    async def audio_turn(self, pcm):
        loop = asyncio.get_running_loop()
        text, lang = await loop.run_in_executor(stt_pool, transcribe, pcm, settings["lang_mode"])
        if is_junk(text):
            await self.send({"type": "ignored"})
            return
        await self.run_turn(text, lang)

    async def text_turn(self, text):
        await self.run_turn(text, guess_text_lang(text, self.last_lang))

    async def preview(self, lang):
        self.epoch += 1
        ep = self.epoch
        await self.send({"type": "turn_start", "epoch": ep, "preview": True, "lang": lang})
        sp = Speaker(self, ep, lang)
        task = asyncio.create_task(sp.run())
        sp.q.put_nowait(LANGS[lang]["sample"])
        sp.q.put_nowait(None)
        try:
            await task
            await self.send({"type": "turn_end", "epoch": ep})
        finally:
            if not task.done():
                task.cancel()

    def _writable_chat(self):
        """The saved record of this conversation, created on its first message. Draft chats are not
        stored, so deleting chats never leaves an empty 'Chat' in the sidebar."""
        if self.chat_id is None or self.chat_id not in CHATS:
            self.chat_id = new_chat_id()
            CHATS[self.chat_id] = {"id": self.chat_id, "title": "", "created": time.time(), "messages": []}
            return CHATS[self.chat_id], True
        return CHATS[self.chat_id], False

    async def run_turn(self, user_text, lang):
        self.last_lang = lang
        self.epoch += 1
        ep = self.epoch
        images = self.pending_images          # images attached since the last message
        self.pending_images = []
        await self.send({"type": "user_text", "text": user_text, "lang": lang, "images": images})
        self.history[0]["content"] = (SYSTEM_PROMPT + f" Today's date is {date.today():%A %d %B %Y}. "
                                      + LANGS[lang]["prompt"])
        uentry = {"role": "user", "content": user_text}
        if images:
            uentry["images"] = images
        self.history.append(uentry)
        chat, created = self._writable_chat()  # saved copy; the record is born with the first message
        if created:
            await self.chats_update()          # the new chat shows up in the sidebar right away
        if not chat["title"]:
            chat["title"] = user_text[:42] + ("…" if len(user_text) > 42 else "")
            await self.chats_update()          # and so does its name
        chat["messages"].append({"role": "user", "text": user_text, "images": images})
        save_chats()
        self.links = None                     # source cards of the previous turn are no longer live
        self.links_title = ""
        await self.send({"type": "turn_start", "epoch": ep, "lang": lang})
        speaker = Speaker(self, ep, lang)
        tts_task = asyncio.create_task(speaker.run())
        answer = ""
        ok = False
        usage = None
        opener = OpenDirective()
        opened = []
        try:
            if self._ensure_room():
                await self.send({"type": "note",
                                 "text": "Context nearly full — the oldest turns of this chat were "
                                         "dropped to make room. Long-lived chats slowly forget their start."})
            messages = [llm_message(e) for e in self.history]
            if self.web:
                extra = await self.search.context(user_text, self.history, lang, deep=self.deep)
                if not extra:                     # no new search: still allow "open the second one"
                    extra = self.search.address_block()
                if extra:
                    last = self.history[-1]
                    messages[-1] = llm_message({**last, "content": last["content"] + extra})
            payload = {"messages": messages, "stream": True,
                       "stream_options": {"include_usage": True},
                       "temperature": 0.6, "top_p": 0.95, "top_k": 20}
            async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=5)) as c:
                async with c.stream("POST", LLM_URL, json=payload) as r:
                    if r.status_code != 200:
                        raise RuntimeError(f"llama-server returned {r.status_code}: "
                                           f"{(await r.aread()).decode(errors='ignore')[:200]}")
                    async for line in r.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        d2 = json.loads(data)
                        u = d2.get("usage")           # final chunk carries token counts
                        if u:
                            usage = {"prompt": u.get("prompt_tokens") or 0,
                                     "gen": u.get("completion_tokens") or 0}
                        ch = d2.get("choices")
                        if not ch:
                            continue
                        delta = ch[0].get("delta", {})
                        think = delta.get("reasoning_content")
                        if think:
                            await self.send({"type": "thinking", "text": think})
                        piece = delta.get("content")
                        if piece:
                            piece, found = opener.feed(piece)
                            opened += found
                            if piece:
                                answer += piece
                                await self.send({"type": "token", "text": piece})
                                speaker.text(piece)
            tail, found = opener.finish()
            opened += found
            if tail:
                answer += tail
                await self.send({"type": "token", "text": tail})
                speaker.text(tail)
            answer = answer.rstrip()
            speaker.finish()
            await tts_task
            await self.send({"type": "turn_end", "epoch": ep})
            ok = True
            urls = self._open_urls(opened, user_text)
            if urls:                                   # the model picked a page: put it on screen
                for u in urls:
                    try:
                        await asyncio.to_thread(webbrowser.open_new_tab, u)
                    except Exception as e:
                        print("[open tab]", e)
                try:
                    await self.send({"type": "opened", "urls": urls})
                except Exception:
                    pass
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print("[turn error]", e)
            try:
                await self.send({"type": "error", "text": f"{type(e).__name__}: {e}"})
            except Exception:
                pass
        finally:
            if not tts_task.done():
                tts_task.cancel()
            if answer:
                text = answer if ok else answer + " [interrupted by the user]"
                self.history.append({"role": "assistant", "content": text})
                if chat is not None:
                    rec = {"role": "assistant", "text": answer, "cut": not ok}
                    if self.links:                    # keep the cards with the reply they belong to
                        rec["links"] = {"title": self.links_title,
                                        "groups": list(self.links.values())}
                    chat["messages"].append(rec)
                    self.links = None
                    self.links_title = ""
            else:
                self.history.pop()   # no answer: drop the unanswered question so roles keep alternating
                if chat is not None and chat["messages"]:
                    chat["messages"].pop()
            if chat is not None:
                save_chats()
            self._trim_history()
            if not usage:             # rough estimate when llama-server did not report usage
                chars = sum(len(m.get("content") or "") for m in self.history)
                usage = {"prompt": chars // 4, "gen": len(answer) // 4}
            if chat is not None:      # remember per-chat usage: shown again when the chat is reopened
                chat["usage"] = {"prompt": usage["prompt"], "gen": usage["gen"]}
                save_chats()
            try:
                await self.send({"type": "ctx", "prompt": usage["prompt"], "gen": usage["gen"],
                                 "limit": CTX, "msgs": len(self.history) - 1})
            except Exception:
                pass

    def _open_urls(self, urls, user_text):
        """Only addresses that really came past us: a search result, or something the user gave."""
        pool = [r.get("url") or "" for r in (getattr(self.search, "last_results", None) or [])]
        for m in self.history:
            pool += re.findall(r"https?://[^\s<>\"']+", m.get("content") or "")
        pool += re.findall(r"https?://[^\s<>\"']+", user_text or "")
        out, seen = [], set()
        for u in urls:
            if not re.match(r"^https?://", u, re.I) or len(u) > 2000:
                continue
            if not any(same_url(u, p) for p in pool if p):
                print("[open skipped, not a known address]", u)
                continue
            k = url_key(u)
            if k in seen:
                continue
            seen.add(k)
            out.append(u)
            if len(out) >= 2:                     # two tabs at most: this is a butler, not a pop-up farm
                break
        return out

    def _ensure_room(self):
        """Token-aware guard using the real usage of the last turn in this chat: when the context
        is close to the limit, drop the oldest completed pairs until there is comfortable room."""
        chat = CHATS.get(self.chat_id)
        u = (chat or {}).get("usage")
        if not u:
            return False
        if u.get("prompt", 0) <= int(CTX * 0.90):
            return False
        est = u["prompt"]
        dropped = 0
        while len(self.history) > 3 and est > int(CTX * 0.70):
            a = self.history.pop(1)                  # user turn
            b = self.history.pop(1)                  # assistant turn
            est -= (len(a.get("content") or "")) // 4 + 800 * len(a.get("images") or [])
            est -= (len(b.get("content") or "")) // 4
            dropped += 2
        if chat is not None and dropped:
            del chat["messages"][:dropped]           # keep the saved copy in sync
            save_chats()
        return dropped > 0

    def _trim_history(self):
        """Keep the context bounded: drop the oldest completed turns (system prompt always stays).
        Popping whole user+assistant pairs keeps the roles alternating."""
        while len(self.history) > MAX_HISTORY:
            del self.history[1:3]
        chat = CHATS.get(self.chat_id)
        if chat is not None and len(chat["messages"]) > MAX_HISTORY - 1:
            del chat["messages"][:2]      # keep the saved copy in sync with the trimmed history
            save_chats()


# ------------------------------------------------------------------ web app
@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(startup_llama())
    yield
    task.cancel()
    if llama.owned:
        await llama.stop_proc()


app = FastAPI(lifespan=lifespan)


# ------------------------------------------------------------------ access key (protects / and /ws when exposed to the internet)
def _load_token():
    """Access key: JARVIS_TOKEN env var if set, else token.txt next to this file
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


if __name__ == "__main__":
    VOICES_DIR.mkdir(exist_ok=True)
    load_settings()
    load_chats()
    print(f"Loading speech recognition ({STT_MODEL})...")
    stt = WhisperModel(STT_MODEL, device="cpu", compute_type="int8")
    try:
        get_voice("en", settings["voices"]["en"])     # warm up English so the first answer is quick
    except Exception as e:
        print("[could not load the English voice]", repr(e))
    print(f"Ready. On this PC:   http://127.0.0.1:{WEB_PORT}   (Cloudflare tunnel for other devices)")
    if TOKEN:
        print(f"Access key ({TOKEN_FROM}): {TOKEN}")
    uvicorn.run(app, host="127.0.0.1", port=WEB_PORT, log_level="warning")