"""llama-server lifecycle: launching, switching and stopping the local brain."""
import asyncio
import json
import os
import re
import subprocess
import time
from pathlib import Path

import httpx

from .config import (CTX, DEFAULT_MODEL, DEFAULT_REASONING, LLAMA_EXE, LLM_BASE, MMPROJ,
                     MODELS_DIR, VRAM_WARN_GB)
from .hub import broadcast


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
