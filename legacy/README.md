# legacy

`voice_chat.py` — the original single-file version of the assistant (web UI, Whisper,
llama-server, Pocket TTS in 640 lines). It is kept for reference only and is **not
maintained**: features added since the split (web search pipeline, `OPEN:` tabs, saved
link cards, the modular `jarvis` package) are not back-ported to it.

The current app is `voie_chat_web.py` + the `jarvis` package at the project root.

## Running it

From the project root, so imports and data files resolve:

```
python legacy/voice_chat.py
```

`HERE` points one level up, so it shares the root data files rather than owning copies:
`index.html`, `app_settings.json` and `voices/` are all at the project root.
