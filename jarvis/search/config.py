"""Search tunables, the site catalog, provider credentials and .env loading."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]     # project folder: .env, Index.html, chats.json


def _load_env():
    """Reads KEY=VALUE lines from the .env in the project folder (real environment variables win)."""
    f = ROOT / ".env"
    try:
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass
    except Exception as e:
        print("[.env not read]", e)


_load_env()


SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://127.0.0.1:8888")


TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")


PROVIDER_ORDER = ["ddgs", "searxng", "tavily"]


RELEVANCE_FILTER = True    # keyword check, used only when the local model cannot judge the results


JUDGE_ENABLED = True       # the local model checks every result against what you asked (country, field, kind of page)


MAX_ROUNDS = 2             # 1 = no second attempt; 2 = weak results make the model write new queries and search again


WEAK_BELOW = 3             # a source with fewer good results than this (or your limit, if smaller) is searched again


JUDGE_CHUNK = 40           # results shown to the model per judging call (about 3 seconds each on a fast GPU)


MAX_JUDGED = 120           # never judge more than this many results per round


TAVILY_MAX_PER_SEARCH = 4  # paid credits: at most this many Tavily calls per question, even for empty sources


FALLBACK_JOB_BOARDS = True # job search, no site named and none suggested by the model: use the boards marked jobs in SITES


COOLDOWN = 60               # seconds a provider is skipped after it failed, so a blocked one is not hammered


TAVILY_ONLY_ON_FAILURE = True   # paid credits: used only when the other providers errored, not on a genuine "no results"


DEFAULT_LIMIT = 5          # results per site (or in total for an open-web search); changeable by voice or in the UI


MAX_LIMIT = 20


PAGES_TO_READ = 5          # pages downloaded and read when the answer is spoken (not for plain link lists)


PAGE_CHARS = 3000


CACHE_TTL = 600            # seconds: the same search gives the same results for 10 minutes


GAPS = {"searxng": 2.0, "tavily": 0.15, "ddgs": 0.8}   # minimum seconds between requests per provider


DEEP_MAX_ROUNDS = 4        # refine rounds instead of MAX_ROUNDS


DEEP_MAX_Q = 5             # query variants per source instead of 2-3


DEEP_FETCH = 14            # candidates fetched per query instead of max(n*2, 8)


DEEP_PAGES = 8             # pages actually read instead of PAGES_TO_READ


DEEP_PAGE_CHARS = 3500     # characters kept per page in deep mode


DEEP_GAPS = {"searxng": 3.0, "ddgs": 1.5, "tavily": 0.15}   # gentler pace: fewer captchas and suspensions


DEEP_TAVILY = 8            # tavily credits for the whole question instead of TAVILY_MAX_PER_SEARCH


UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"}


DDG_REGION = {"en": "us-en", "fr": "fr-fr", "es": "es-es"}


SITES = {
    "indeed": {"name": "Indeed", "domain": "indeed.com", "post": r"viewjob|/rc/clk|jk=", "jobs": True,
               "search": "https://ma.indeed.com/jobs?q={q}&l=Maroc", "only_if": r"maroc|morocco", "alias": ["indeed"]},
    "linkedin": {"name": "LinkedIn", "domain": "linkedin.com/jobs", "post": r"/jobs/view/", "jobs": True,
                 "search": "https://www.linkedin.com/jobs/search/?keywords={q}&location=Morocco",
                 "only_if": r"maroc|morocco", "alias": ["linkedin", "linkdin", "linkedn", "linked in", "لينكدان", "لينكد ان"]},
    "rekrute": {"name": "ReKrute", "domain": "rekrute.com", "post": r"offre-emploi", "jobs": True,
                "search": None, "alias": ["rekrute"]},
    "emploi": {"name": "Emploi.ma", "domain": "emploi.ma", "post": r"offre", "jobs": True,
               "search": None, "alias": ["emploi.ma"]},
    "marocannonces": {"name": "MarocAnnonces", "domain": "marocannonces.com", "post": "", "jobs": True,
                      "search": None, "alias": ["marocannonces", "maroc annonces"]},
    "jobzyn": {"name": "Jobzyn", "domain": "jobzyn.com", "post": "", "jobs": True, "search": None,
               "alias": ["jobzyn"]},
    "bayt": {"name": "Bayt", "domain": "bayt.com", "post": r"/jobs?/", "jobs": True, "search": None,
             "alias": ["bayt"]},
    "joblia": {"name": "Joblia", "domain": "joblia.ma", "post": "", "jobs": True, "search": None,
               "alias": ["joblia"]},
    "reddit": {"name": "Reddit", "domain": "reddit.com", "post": "", "search": None, "alias": ["reddit"]},
    "github": {"name": "GitHub", "domain": "github.com", "post": "", "search": None, "alias": ["github"]},
    "stackoverflow": {"name": "Stack Overflow", "domain": "stackoverflow.com", "post": "", "search": None,
                      "alias": ["stackoverflow", "stack overflow"]},
    "wikipedia": {"name": "Wikipedia", "domain": "wikipedia.org", "post": "", "search": None,
                  "alias": ["wikipedia"]},
    "youtube": {"name": "YouTube", "domain": "youtube.com", "post": "", "search": None, "alias": ["youtube"]},
}


ALIAS_MAP = {a: k for k, s in SITES.items() for a in s["alias"]}


BLOCKED = ("pinterest.", "facebook.com", "instagram.com", "tiktok.com")
