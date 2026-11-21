"""The per-session search pipeline: plan -> search -> fuse -> judge -> read -> answer."""
import asyncio
from datetime import date

from .config import (DEEP_MAX_Q, DEEP_MAX_ROUNDS, DEEP_PAGES, DEEP_PAGE_CHARS, DEEP_TAVILY,
                     DEFAULT_LIMIT, FALLBACK_JOB_BOARDS, JUDGE_ENABLED, MAX_LIMIT, MAX_JUDGED,
                     MAX_ROUNDS, PAGES_TO_READ, RELEVANCE_FILTER, SITES, TAVILY_MAX_PER_SEARCH,
                     WEAK_BELOW)
from .engines import active_providers, diagnose, run_search
from .judging import judge, refine_queries
from .memory import remember_sources
from .planning import make_plan
from .ranking import collect, fuse, relevant, topic_tokens
from .reading import describe, extract_job_offers, read_pages
from .sites import resolve_sites
from .utils import (OPEN_HINT, address_rows, fallback_job_query, interleave, norm_url, number,
                    public, strip_sites, trim_total)


class WebSearch:
    def __init__(self, send, llm_url, limit=DEFAULT_LIMIT):
        self.send, self.llm_url = send, llm_url
        self.limit = max(1, min(MAX_LIMIT, int(limit)))
        self.reset()

    def reset(self):
        self.last_plan, self.last_results, self.last_groups, self.last_ctx = None, [], [], None

    def prefs(self):
        return {"type": "search_prefs", "limit": self.limit, "providers": active_providers()}

    async def _judge_groups(self, groups, request, queries, n, pools):
        """Lets the model pick the results that match the request. pools: id(group) -> candidates to judge.
        Picked results are merged with what the group already has (good ones first), up to n per group."""
        flat = [(g, r) for g in groups for r in pools.get(id(g), [])][:MAX_JUDGED]
        if not flat:
            return
        toks = topic_tokens(queries, [])
        verdict = None
        if JUDGE_ENABLED:
            await self.send({"type": "note", "text": f"Checking {len(flat)} results with the local model…"})
            verdict = await judge(self.llm_url, request, queries, [(g["site"], r) for g, r in flat])
        picked = {id(g): {"good": [], "partial": [], "unchecked": []} for g in groups}
        for i, (g, r) in enumerate(flat):
            if verdict is None or i in verdict["unknown"]:
                kind = "unchecked" if (relevant(r, toks) or not RELEVANCE_FILTER) else None   # model unavailable
            else:
                kind = "good" if i in verdict["good"] else "partial" if i in verdict["partial"] else None
            if kind:
                picked[id(g)][kind].append(dict(r, match=kind))
        if verdict is not None:
            print(f"[judge] {len(flat)} results -> good={len(verdict['good'])} partial={len(verdict['partial'])} "
                  f"unchecked={len(verdict['unknown'])}")
        for g in groups:
            p = picked[id(g)]
            old = g["items"]
            ranked = ([x for x in old if x["match"] == "good"] + p["good"] +
                      [x for x in old if x["match"] != "good"] + p["partial"] + p["unchecked"])
            seen, out = set(), []
            for x in ranked:
                k = norm_url(x["url"])
                if k not in seen:
                    seen.add(k)
                    out.append(x)
            g["items"] = out[:n]

    def _weak(self, groups, n):
        want = min(n, WEAK_BELOW)
        return [g for g in groups if sum(1 for x in g["items"] if x["match"] == "good") < want]

    async def context(self, user_text, history, lang, deep=False):
        """Runs the whole pipeline. Returns text to append to the user's message ('' = nothing to add)."""
        await diagnose()
        plan = await make_plan(self.llm_url, user_text, history, self.last_plan, self.last_results)
        if plan["refs"] and self.last_results:
            return await self._followup(plan)
        sites, unresolved = await resolve_sites(plan["sites"], user_text, lang)
        named = bool(sites or unresolved)
        print(f"[plan] intent={plan['intent']} search={plan['search']} sites={[s['name'] for s in sites]} "
              f"unresolved={unresolved} suggest={plan['suggest']} queries={plan['queries']} limit={plan['limit']}")
        if not plan["search"] and not named:
            return ""
        jobs, news = plan["intent"] == "jobs", plan["intent"] == "news"
        auto = False
        if not named and (jobs or plan["suggest"]):
            # nobody named a site: the model suggests the boards for this country and field, and the names are
            # turned into domains by a search (so a made-up name is simply dropped)
            sites, _ = await resolve_sites(plan["suggest"], "", lang)
            if not sites and jobs and FALLBACK_JOB_BOARDS:
                sites = [s for s in SITES.values() if s.get("jobs")]
            auto = True
        if plan["limit"]:
            self.limit = plan["limit"]                           # sticky for the rest of the session
            await self.send(self.prefs())
        n = self.limit
        recency = plan["recency"] or ("week" if news else None)
        queries = strip_sites(plan["queries"], sites)
        if not queries:
            queries = (self.last_plan or {}).get("queries") or [fallback_job_query(user_text) if jobs else user_text]
        if unresolved:
            await self.send({"type": "note", "text": "Could not find the website for: " + ", ".join(unresolved) +
                             ". Using the name as a keyword in a general web search instead."})
            queries = [q + " " + " ".join(unresolved) for q in queries]
        self.last_plan = dict(plan, queries=queries)
        label = "News" if news else ("Other sources" if sites else "Web")
        targets = [(s, auto) for s in sites] + ([(None, False)] if (auto or not sites) else [])
        max_q = DEEP_MAX_Q if deep else (3 if len(targets) <= 2 else 2)
        cap = max(n * 3, 12) if deep else max(n * 2, 8)   # candidates shown to the judge per source
        await self.send({"type": "note", "text": ("Deep research — searching: " if deep else "Searching: ") +
                         " | ".join(queries[:max_q]) +
                         (" on " + ", ".join(s["name"] for s in sites) if sites else "")})
        budget = {"tavily": DEEP_TAVILY if deep else TAVILY_MAX_PER_SEARCH}
        sem = asyncio.Semaphore(3 if deep else 4)   # fewer parallel sites: searxng's upstream engines rate limit fast

        async def go(s, a):
            async with sem:
                return await collect(s, queries[:max_q], n, recency, news, lang, label, budget=budget, auto=a,
                                     deep=deep)

        groups = list(await asyncio.gather(*(go(s, a) for s, a in targets)))
        await self._judge_groups(groups, user_text, queries, n, {id(g): g["cands"][:cap] for g in groups})
        for _ in range((DEEP_MAX_ROUNDS if deep else MAX_ROUNDS) - 1):   # weak results: the model writes better queries
            weak = self._weak(groups, n)
            if not weak:
                break
            samples = [r["title"] for g in weak for r in (g["items"] + g["cands"])]
            new_q = [q for q in strip_sites(await refine_queries(self.llm_url, user_text, queries, samples), sites)
                     if q.lower() not in {x.lower() for x in queries}]
            if not new_q:
                break
            await self.send({"type": "note", "text": "Refining the search: " + " | ".join(new_q[:2])})
            pools = {}
            for g in weak:
                site = next((s for s, _a in targets if (s["name"] if s else label) == g["site"]), None)
                more = await collect(site, new_q[:max_q] if deep else new_q[:2], n, recency, news, lang, label,
                                     budget=budget, deep=deep)
                fresh_rows = [r for r in more["cands"] if norm_url(r["url"]) not in g["seen"]]
                g["seen"].update(norm_url(r["url"]) for r in fresh_rows)
                g["cands"] += fresh_rows
                if fresh_rows and g["status"] != "ok":
                    g["status"], g["provider"], g["error"] = "ok", more["provider"], None
                pools[id(g)] = fresh_rows[:cap]
            queries = queries + new_q
            await self._judge_groups(weak, user_text, queries, n, pools)
        if not any(g["items"] for g in groups):                   # never blank: show the best raw candidates
            for g in groups:
                g["items"] = [dict(r, match="unchecked") for r in g["cands"][:n]]
        if plan["limit"] and plan["limit_scope"] == "total" and len(groups) > 1:
            trim_total(groups, n)
        shown = [g for g in groups if g["items"] or g["status"] == "failed" or not g["auto"]]
        self.last_results = number(shown)
        remember_sources(plan["intent"], shown)      # learn which domains delivered good results
        self.last_groups = shown
        self.last_ctx = {"queries": queries[:max_q], "recency": recency, "news": news, "n": n, "lang": lang,
                         "request": user_text, "deep": deep,
                         "sites": {(s["name"] if s else label): s for s, _a in targets}}
        title = (queries[0] + (f" · {n} per site" if len(sites) > 1 else "")) if plan["display"] == "links" \
            else "Sources"
        await self.send({"type": "links", "title": title, "groups": [public(g) for g in shown]})
        flat = interleave(shown)
        failed = [g["site"] for g in shown if g["status"] == "failed" and not g["items"]]
        if not flat:
            return ("\n\n[The search returned no results" +
                    (" because the search sources failed or were rate limited" if failed else "") +
                    ". Say so in one sentence: the screen shows each site's status and a Retry button.]")
        read_n = DEEP_PAGES if deep else PAGES_TO_READ
        page_of, job_of = {}, {}                 # page text keyed by URL, so read order is free
        if deep or plan["display"] != "links":    # deep research reads pages even for plain link lists
            order = {"good": 0, "partial": 1, "unchecked": 2}
            best = sorted(flat, key=lambda r: order.get(r.get("match", "unchecked"), 2))
            top, seen_r = [], set()
            for r in best:                        # judged-best pages first, one copy of each URL
                k = norm_url(r["url"])
                if k not in seen_r:
                    seen_r.add(k)
                    top.append(r)
                    if len(top) >= read_n:
                        break
            if deep:
                await self.send({"type": "note", "text": f"Deep research: reading {len(top)} pages…"})
            pages = await read_pages(top, total=40 if deep else None, chars=DEEP_PAGE_CHARS if deep else None,
                                     deep=deep)
            page_of = {norm_url(r["url"]): t for r, t in zip(top, pages)}
            if deep and plan.get("intent") == "jobs" and any(pages):
                await self.send({"type": "note", "text": "Deep research: extracting the offers from the pages…"})
                try:
                    job_notes = await extract_job_offers(self.llm_url, top, pages)
                    job_of = {norm_url(r["url"]): t for r, t in zip(top, job_notes)}
                except Exception as e:
                    print("[job extraction failed]", repr(e))

        def blk(i, r):
            """One result for the LLM: header + page text (+ structured offers when extracted)."""
            k = norm_url(r["url"])
            s = describe(r, page_of.get(k, ""))
            return s + job_of.get(k, "")
        from datetime import date
        if plan["display"] == "links":
            found = "\n".join(f"{g['site']}: " + "; ".join(i["title"] for i in g["items"][:4])
                              for g in shown if g["items"])
            body = f"\n\n[{'DEEP RESEARCH' if deep else 'SEARCH'} RESULTS for \"{queries[0]}\"]\n{found}\n"
            if deep:
                body += "\n\n".join(blk(i, r) for i, r in enumerate(top)) + "\n"
            rows = address_rows(flat)
            if rows:
                body += ("\n[EXACT ADDRESSES, in the order shown on screen]\n" + rows +
                         "\n[END ADDRESSES]\n" + OPEN_HINT)
            how = ("The pages of the best results were opened and read: point out the strongest ones and say what "
                   "each really contains (requirements, dates, salary, what the page is about), in a few short "
                   "sentences. " if deep else
                   "In two or three short sentences, say how many results were found and on which sites, mention two "
                   "or three notable ones by title or company, and ")
            return (body + "\n[END]\n\nThe full list with clickable links is already displayed on screen. " + how +
                    "say the links are on screen." +
                    (f" These sites failed and have a Retry button: {', '.join(failed)}." if failed else "") +
                    " Never read URLs aloud.")
        blocks = [blk(i, r) for i, r in enumerate(flat)]
        rows = address_rows(flat, 5)
        addrs = ("\n[EXACT ADDRESSES]\n" + rows + "\n[END ADDRESSES]\n" + OPEN_HINT) if rows else ""
        return (f"\n\n[{'DEEP RESEARCH' if deep else 'WEB SEARCH'} RESULTS, fetched {date.today():%d %B %Y}]\n" +
                "\n\n".join(blocks) + "\n[END OF RESULTS]" + addrs + "\n\nAnswer the message above using these "
                "results. Give the "
                "actual facts, not a list of websites. For news, summarize the 3 or 4 most important stories in one "
                "sentence each, naming the outlet naturally. Never read URLs or result numbers aloud. Keep it "
                "suitable for speech. Only if the results truly lack the answer, say what is missing.")

    async def _followup(self, plan):
        chosen = [r for r in self.last_results if r.get("n") in plan["refs"]][:3]
        if not chosen:
            return ""
        await self.send({"type": "note", "text": "Reading result " + ", ".join(f"#{r['n']}" for r in chosen)})
        pages = await read_pages(chosen)
        blocks = [describe(r, pages[i] if i < len(pages) else "") for i, r in enumerate(chosen)]
        rows = address_rows(chosen)
        addrs = ("\n[EXACT ADDRESSES]\n" + rows + "\n[END ADDRESSES]\n" + OPEN_HINT) if rows else ""
        return ("\n\n[SELECTED RESULTS]\n" + "\n\n".join(blocks) + "\n[END]" + addrs +
                "\n\nAnswer the user's request about these "
                "selected results (summarize, list requirements, compare...). Never read URLs aloud. If a page text "
                "is missing, work from the snippet and say so.")

    def address_block(self, n=5):
        """Addresses from the previous search, so 'open the second one' still works without searching again."""
        rows = address_rows(self.last_results, n)
        if not rows:
            return ""
        return ("\n\n[ADDRESSES FROM THE LAST SEARCH — the results numbered on screen]\n" + rows +
                "\n[END ADDRESSES]" + OPEN_HINT)

    async def retry(self, site_name, lang=None):
        """Re-runs one failed site with the same queries, bypassing the cache, and updates its card."""
        try:
            ctx = self.last_ctx
            old = next((g for g in self.last_groups if g["site"] == site_name), None)
            if not ctx or not old:
                return
            deep = ctx.get("deep", False)
            budget = {"tavily": DEEP_TAVILY if deep else TAVILY_MAX_PER_SEARCH}
            new = await collect(ctx["sites"].get(site_name), ctx["queries"], ctx["n"], ctx["recency"], ctx["news"],
                                lang or ctx["lang"], site_name, fresh=True, budget=budget, deep=deep)
            cap = max(ctx["n"] * 3, 12) if deep else max(ctx["n"] * 2, 8)
            await self._judge_groups([new], ctx["request"], ctx["queries"], ctx["n"], {id(new): new["cands"][:cap]})
            self.last_results = [r for r in self.last_results if r not in old["items"]]
            number([new], max([r.get("n", 0) for r in self.last_results] + [0]) + 1)
            self.last_results += new["items"]
            self.last_groups[self.last_groups.index(old)] = new
            await self.send({"type": "links_update", "group": public(new)})
        except Exception as e:
            print("[retry error]", repr(e))
            await self.send({"type": "error", "text": f"Retry failed: {e}"})
