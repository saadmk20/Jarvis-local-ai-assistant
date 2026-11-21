"""One browser connection: chat lifecycle, streaming a turn, search and speech hooks."""
import asyncio
import json
import re
import time
import webbrowser
from datetime import date

import httpx

from .chats import CHATS, chats_msg, llm_message, new_chat_id, rich_to_llm, save_chats
from .config import (CTX, LANGS, LLM_URL, MAX_HISTORY, SYSTEM_PROMPT, settings)
from .directives import OpenDirective
from .search import WebSearch
from .speech import Speaker, stt_pool, transcribe
from .textutil import guess_text_lang, is_junk, same_url, url_key


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
