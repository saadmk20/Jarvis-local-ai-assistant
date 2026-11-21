# JARVIS — local AI voice assistant with a research-grade search engine

A fully local, self-hosted voice assistant: you speak, it plans searches, reads pages,
judges what it found, and answers out loud - in English, French or Spanish. No cloud LLM,
no telemetry: the brain, the ears, the voice and the search index all run on one PC.

Built as a general "clever assistant" (research, news, fact questions, job hunting…),
not a demo: every search is planned, fused, filtered and *learned from*, and the whole
stack survives real-world conditions — dead search providers, anti-bot sites, overflowed
context windows.

```
you (voice or text) ──> Whisper STT ──> local LLM (27B, llama.cpp) ──> planned answer
                                              │                            │
                                    web search engine                  Pocket TTS
                                    (websearch.py)                     (cloned voices)
                                              │
              ┌───────────────┬───────────────┴──────────────┐
              ddgs            self-hosted SearXNG           Tavily (paid,
              (free)          (Docker, Bing+Yahoo)          last resort only)
```

## What it does

| | |
|---|---|
| **Voice in/out** | faster-whisper STT (CPU), Pocket TTS with custom voice clones (EN/FR/ES); speech plays in real time |
| **Local LLM brain** | a 27B GGUF model via llama-server on a single consumer GPU (~52 tok/s generation, ~800 tok/s prompt processing) |
| **Vision** | attach, paste or drag up to 4 images per message (auto-downscaled); a vision projector runs alongside the LLM |
| **Web search engine** | one pipeline for every question: plan → search → fuse → judge → read → answer |
| **Deep research mode** | same pipeline, 4× rounds, more query variants, up to 8 full pages read — for demanding questions (e.g. a complete job hunt) |
| **Multi-chat UI** | sidebar with persistent conversations, per-chat context-usage dashboard, live link cards |
| **Acts, not just answers** | when it judges that a page *is* what you asked for, it opens it in your browser — the model ends its reply with `OPEN: <url>`, the marker is stripped before speaking, and the address is validated against the results it actually received |
| **Remote access** | token-gated UI (401 page + websocket 4401), 90-day cookie, exposed through a Cloudflare Tunnel |

## The search pipeline (the interesting part)

```
question
  │
  ▼
PLANNER (schema-constrained JSON)
  decides intent, 2–5-word keyword queries (one skill/topic each, in the
  target country's language), and which sites to try — inferring from the
  question who actually publishes that kind of content
  │
  ▼
SEARCH  providers in order: ddgs → searxng → tavily (paid credits guarded)
        results fused with reciprocal-rank fusion, deduped, homepage/junk filtered
        provider-restricted searches that return nothing get a free open-web retry
        before any paid call is spent
  │
  ▼
JUDGE (LLM-as-judge, structured output)
  labels every result good / partial / irrelevant — topic, page type,
  country and language all checked
  │
  ▼
READ  a three-rung fetch ladder (Scrapling) reads the good pages:
      plain httpx → Scrapling AsyncFetcher (browser-like TLS via curl_cffi)
      → Scrapling StealthyFetcher, a real stealth browser (deep mode, capped)
      so JavaScript shells and anti-bot sites still yield text
  │
  ▼
ANSWER + links card on screen; for job searches, a schema-constrained
extraction pass pulls structured offers (title, company, city, skills…)
  │
  ▼
LEARN  domains of "good" results are counted per intent in source_memory.json
       and auto-registered as resolvable sites: the next similar search starts
       from sources already proven to work
```

Nothing is hardcoded per topic or country — the assistant infers sources itself and
improves from its own runs. A typical deep run judges 40+ candidates, keeps ~60% as
`good`, and falls back to the open web to rescue suggested boards that returned nothing
(one live run rescued 12 and 17 candidates this way).

## Resilience details that matter in production

- **Provider health**: a search provider that fails (dead endpoint, timeouts) is retried
  around, never silently swallowed; deep mode fuses two providers when one is down and
  the paid provider only opens when the free ones hard-fail.
- **Context guard**: if the last turn's token usage exceeds 90% of the 49k context,
  the oldest turns are dropped automatically until it's back near 70% — the user is
  told, never truncated silently. (History is capped at 61 messages per chat.)
- **Per-page timeouts** on the fetch ladder: one slow site can't stall the pipeline.
- **Paid-API budgeting**: Tavily is capped per search, gated behind hard failures, and
  never used while free providers still return results.

## Setup

### 1. Requirements
- Python 3.12+, and for the voice server: `pip install fastapi uvicorn httpx numpy faster-whisper pocket-tts ddgs trafilatura scrapling[fetchers]`
- A llama.cpp `llama-server` with a local GGUF model (and an `mmproj` GGUF for vision)

### 2. Configuration
Copy `.env.example` to `.env` and set what you use:

```ini
TAVILY_API_KEY=...      # optional, paid last-resort search provider
SEARXNG_URL=http://127.0.0.1:8888
# JARVIS_TOKEN=...      # optional UI access key; a random token.txt is generated otherwise
```

### 3. SearXNG (recommended, free meta-search)
```bash
docker run -d --name searxng -p 8888:8080 searxng/searxng
```
Then enable the JSON API in its `settings.yml`: `search: formats: [html, json]`.
Only engines that actually work from your network need to stay enabled — a startup
probe checks the instance and reports blocked engines.

### 4. Run
```bash
python voie_chat_web.py        # serves http://127.0.0.1:7860
```
The server reuses a running llama-server or starts one; Whisper and TTS load lazily.
For remote access, point a Cloudflare Tunnel at port 7860 — the token gate applies
to both the web page and the websocket.

## Design decisions

- **Everything local**: the LLM, STT, TTS, judging and planning run on-device. Only the
  actual web queries leave the machine (to search providers). This keeps latency
  predictable and private by default.
- **One generic pipeline** instead of one scraper per website: the planner, judge and
  fetch ladder are topic-agnostic, so "find me internships in Casablanca" and "what
  changed in the news today" take the same path.
- **LLM-as-judge** for relevance: keyword filtering is a fallback only; the model checks
  intent, page type, country and language, in 40-result chunks (~3 s per chunk locally).
- **Experience memory**: sources are learned, not curated — the system records which
  domains produced `good` results per intent and prefers them next time. No human
  ever writes a site list.
- **Paid calls as a safety net**: Tavily exists because single-network search coverage
  sometimes fails entirely; its budget and ordering make it the exception, not the plan.

## Limitations (known, by design)

- Sites that fully block non-browser clients only yield to Scrapling's stealth
  browser, capped at 3 fetches per deep search (slow + rate-limit safety).
- Provider-restricted fallback cards may contain results found *via* the suggested
  site's name, not hosted on it.
- STT uses whisper-small on CPU; medium is more accurate but roughly doubles latency.
- The whole stack assumes one user (a single GPU, one TTS voice at a time).

## Project structure

```
voie_chat_web.py            launcher: `python voie_chat_web.py`
jarvis/
  server.py                 FastAPI app: token gate, index page, websocket endpoint, startup
  session.py                one browser connection: chat lifecycle and a streamed turn
  speech.py                 Whisper STT, Pocket TTS voices, the sentence speaker
  llm.py                    llama-server lifecycle (launch, switch, stop)
  chats.py                  saved conversations, chat summaries, rich → LLM conversion
  config.py                 paths, tunables, prompts, language table, saved settings
  textutil.py               display cleaning, sentence splitting, language guessing
  directives.py             in-band directives (e.g. `OPEN: <url>`) stripped before speaking
  hub.py                    the live-session registry used for broadcasts
  search/
    pipeline.py             WebSearch: plan → search → fuse → judge → read → answer
    planning.py             the model turns a sentence into sites, queries and limits
    engines.py              providers (ddgs, searxng, tavily), health, pacing, cache
    ranking.py              fusion, relevance scoring, per-source collection
    judging.py              the model judges results and asks for better queries
    reading.py              page fetch ladder and reading (HTTP, trafilatura, stealth browser)
    sites.py                named-site detection and resolution
    memory.py               learned sources per intent
    utils.py                text normalisation, URL identity, numbering of results
    config.py               search tunables, site catalog, .env loading
websearch.py                compatibility shim for the old import path
Index.html                  the UI (sidebar chats, dashboard, link cards, image attach)
.env.example                template for API keys and endpoints
source_memory.json          learned sources per intent (generated at runtime)
known_sites.json            name → domain map, auto-extended by the learning loop (generated)
legacy/voice_chat.py        the original single-file version, kept for reference (unmaintained)
```

The `jarvis.search` package has no imports from the rest of the app: it takes a `send`
callback and an LLM URL, so it can be tested or reused on its own. The app package never
imports `websearch.py`; that file only keeps old scripts working.

