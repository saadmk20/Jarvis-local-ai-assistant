"""JARVIS voice assistant (multilingual): web UI + Whisper (CPU) + llama-server + Pocket TTS.

Run:  python voice_app.py   then open http://127.0.0.1:7860
Keep index.html next to this file. Put your voices in the "voices" folder next to it.
"""
import asyncio
import json
import re
import struct
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel
from pocket_tts import TTSModel, export_model_state

# ------------------------------------------------------------------ settings
HERE = Path(__file__).parent
VOICES_DIR = HERE / "voices"
SETTINGS_FILE = HERE / "app_settings.json"
LLAMA_EXE = Path(r"F:\AI\Prism\llama-server.exe")
MODELS_DIR = Path(r"F:\AI\Models")
DEFAULT_MODEL = "Ternary-Bonsai-2-27B-Uncensored-Heretic-v2-PTQ1_0.gguf"  # falls back to the first .gguf found
DEFAULT_REASONING = "medium"        # auto | off | low | medium | high
CTX = 49152
LLM_BASE = "http://127.0.0.1:8080"
LLM_URL = LLM_BASE + "/v1/chat/completions"
WEB_PORT = 7860
MIN_CHARS = 30                      # merge very short sentences before speaking
STT_MODEL = "small"                 # multilingual. "medium" is better for Arabic but slower on CPU
VRAM_WARN_GB = 7.3                  # models bigger than this are flagged in the UI
DEFAULT_VOICE_AUDIO = "hf://kyutai/tts-voices/alba-mackenna/casual.wav"   # voice used by models without a catalog
SYSTEM_PROMPT = (
    "You are JARVIS, a calm, witty, formal British AI assistant. "
    "Address the user as 'Master'. Keep spoken answers short and precise. "
    "You are also an expert coding assistant: when you write code, put it in "
    "code blocks and explain it briefly in plain sentences."
)

# One entry per language. "tts" is passed to TTSModel.load_model(...).
# Add "_24l" to a language name (e.g. "french_24l") for the bigger, higher-quality but slower variant.
LANGS = {
    "en": {"name": "English", "tts": {"language": "english"},
           "prompt": "Always reply in English.",
           "say_code": "Code block skipped.",
           "sample": "Good evening. How can I help you today?"},
    "fr": {"name": "Français", "tts": {"language": "french"},
           "prompt": "Always reply in French.",
           "say_code": "Bloc de code ignoré.",
           "sample": "Bonsoir. Comment puis-je vous aider ?"},
    "es": {"name": "Español", "tts": {"language": "spanish"},
           "prompt": "Always reply in Spanish.",
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
settings = {"lang_mode": "auto", "voices": {}}


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
        segs, info = stt.transcribe(
            a, language=language, beam_size=1, vad_filter=True,
            initial_prompt="Python, PyCharm, llama.cpp, GGUF, CUDA, JSON" if language == "en" else None)
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
        self.status = "stopped"
        self.msg = ""
        self.lock = asyncio.Lock()

    def status_msg(self):
        return {"type": "llm_status", "status": self.status, "msg": self.msg,
                "model": self.model, "reasoning": self.reasoning, "owned": self.owned}

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
            await self._set("loading", f"Loading {model}…")
            cmd = [str(LLAMA_EXE), "-m", str(path), "-ngl", "99", "-c", str(CTX),
                   "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--jinja", "-np", "1",
                   *reasoning_flags(reasoning), "--host", "127.0.0.1", "--port", "8080"]
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


# ------------------------------------------------------------------ one browser connection
class Session:
    def __init__(self, ws):
        self.ws = ws
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.epoch = 0
        self.turn = None
        self.last_lang = "en"
        self.send_lock = asyncio.Lock()

    async def send(self, obj):
        async with self.send_lock:
            await self.ws.send_text(json.dumps(obj))

    async def send_bytes(self, b):
        async with self.send_lock:
            await self.ws.send_bytes(b)

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

    async def run_turn(self, user_text, lang):
        self.last_lang = lang
        self.epoch += 1
        ep = self.epoch
        await self.send({"type": "user_text", "text": user_text, "lang": lang})
        self.history[0]["content"] = SYSTEM_PROMPT + " " + LANGS[lang]["prompt"]
        self.history.append({"role": "user", "content": user_text})
        await self.send({"type": "turn_start", "epoch": ep, "lang": lang})
        speaker = Speaker(self, ep, lang)
        tts_task = asyncio.create_task(speaker.run())
        answer = ""
        ok = False
        try:
            payload = {"messages": self.history, "stream": True,
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
                        ch = json.loads(data).get("choices")
                        if not ch:
                            continue
                        delta = ch[0].get("delta", {})
                        think = delta.get("reasoning_content")
                        if think:
                            await self.send({"type": "thinking", "text": think})
                        piece = delta.get("content")
                        if piece:
                            answer += piece
                            await self.send({"type": "token", "text": piece})
                            speaker.text(piece)
            speaker.finish()
            await tts_task
            await self.send({"type": "turn_end", "epoch": ep})
            ok = True
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
            if ok and answer:
                self.history.append({"role": "assistant", "content": answer})
            elif answer:
                self.history.append({"role": "assistant", "content": answer + " [interrupted by the user]"})
            else:
                self.history.pop()   # no answer: drop the unanswered question so roles keep alternating


# ------------------------------------------------------------------ web app
@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(startup_llama())
    yield
    task.cancel()
    if llama.owned:
        await llama.stop_proc()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(HERE / "index.html")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    sess = Session(ws)
    sessions.add(sess)
    await sess.send({"type": "hello", "models": list_models(),
                     "reasoning_options": ["auto", "off", "low", "medium", "high"],
                     "langs": [{"code": c, "name": v["name"]} for c, v in LANGS.items()]})
    await sess.send(voices_msg())
    await sess.send(llama.status_msg())
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
                    sess.history = [{"role": "system", "content": SYSTEM_PROMPT}]
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
    print(f"Loading speech recognition ({STT_MODEL})...")
    stt = WhisperModel(STT_MODEL, device="cpu", compute_type="int8")
    try:
        get_voice("en", settings["voices"]["en"])     # warm up English so the first answer is quick
    except Exception as e:
        print("[could not load the English voice]", repr(e))
    print(f"Ready. Open http://127.0.0.1:{WEB_PORT}")
    uvicorn.run(app, host="127.0.0.1", port=WEB_PORT, log_level="warning")