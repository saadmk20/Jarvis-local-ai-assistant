"""Voices: catalogs, TTS workers, Whisper recognition and the per-turn sentence speaker."""
import asyncio
import re
import struct
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from faster_whisper import WhisperModel
from pocket_tts import TTSModel, export_model_state

from .config import (BUILTIN, DEFAULT_VOICE_AUDIO, LANGS, MIN_CHARS, STT_MODEL, VOICES_DIR,
                     settings)
from .textutil import clean, pop_sentences


stt = None


engines = {}          # language -> TTSModel (loaded on first use)


voice_states = {}     # (language, voice id) -> voice state


tts_pool = ThreadPoolExecutor(max_workers=1)


stt_pool = ThreadPoolExecutor(max_workers=1)


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


def voices_msg():
    return {"type": "voices", "voices": list_voices(), "selected": settings["voices"],
            "lang_mode": settings["lang_mode"]}
