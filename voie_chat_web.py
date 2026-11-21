"""JARVIS voice assistant — entry point.

Run:  python voie_chat_web.py     then open http://127.0.0.1:7860

The implementation lives in the `jarvis` package:
    jarvis/config.py      paths, tunables, prompts, language table, saved settings
    jarvis/server.py      FastAPI app, token gate, websocket endpoint, launcher
    jarvis/session.py     one browser connection: chat lifecycle and a streamed turn
    jarvis/speech.py      Whisper STT, Pocket TTS voices, the per-turn speaker
    jarvis/llm.py         llama-server lifecycle (launch, switch, stop)
    jarvis/chats.py       saved conversations on disk
    jarvis/search/        the web search engine (plan, search, judge, read)
"""
from jarvis.server import run

if __name__ == "__main__":
    run()
