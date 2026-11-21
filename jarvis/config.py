"""Paths, tunables, prompts, the language table and the persisted user settings."""
import json
from pathlib import Path

from .search.config import DEFAULT_LIMIT, MAX_LIMIT

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parent      # data files (chats, voices, Index.html) live next to the launcher
HERE = ROOT


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


JUNK = {"", "you", "thank you", "thanks for watching", "bye", "merci", "merci davoir regarde cette video",
        "gracias"}


settings = {"lang_mode": "auto", "voices": {}, "search_limit": DEFAULT_LIMIT}


def load_settings():
    from .speech import default_voice, list_voices   # speech imports config

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
