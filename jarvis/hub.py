"""The live browser sessions, so any module can broadcast to all of them."""
sessions = set()


sessions = set()


async def broadcast(obj):
    for s in list(sessions):
        try:
            await s.send(obj)
        except Exception:
            pass
