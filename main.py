import logging
import time
import asyncio
import sys
import os
import re
import json
import ollama
from ddgs import DDGS
from newspaper import Article, Config, ArticleException
from sentence_transformers import SentenceTransformer, util
import torch
from urllib.parse import urlparse
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth

# ─── Logging Configuration ────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agent.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Silence third-party HTTP libraries that spam HEAD/GET redirect logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

MAX_SEARCH_QUERIES = 5    # how many search queries the planner generates
MAX_RESULTS_PER_QUERY = 5 # URLs per search
PAYWALL_CHAR_THRESHOLD = 500
MIN_CONTENT_CHARS = 300   # below this, don't even check relevance
SCRAPE_TIMEOUT_MS = 30000 # 30 seconds

BLACKLIST_FILE = "blacklist.json"
AUTO_BLACKLIST_THRESHOLD = 2  # failures before domain is auto-blacklisted

# Domains always blocked (social media / login walls)
_STATIC_BLACKLIST = {
    "twitter.com", "facebook.com", "linkedin.com",
    "instagram.com", "youtube.com", "reddit.com",
    "x.com",
    "msn.com",        # consistently returns 0 chars
}

class DomainBlacklist:
    """Persistent blacklist that self-learns from scraping failures."""

    def __init__(self):
        self._blocked: set[str] = set(_STATIC_BLACKLIST)
        self._failures: dict[str, int] = {}
        self._load()

    def _load(self):
        if os.path.exists(BLACKLIST_FILE):
            try:
                with open(BLACKLIST_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._blocked.update(data.get("blocked", []))
                self._failures = data.get("failures", {})
                logger.info(f"Loaded blacklist: {len(self._blocked)} blocked domains, "
                            f"{len(self._failures)} tracked domains")
            except Exception as e:
                logger.warning(f"Could not load blacklist file: {e}")

    def _save(self):
        try:
            data = {
                "blocked": sorted(self._blocked - _STATIC_BLACKLIST),
                "failures": self._failures
            }
            with open(BLACKLIST_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save blacklist file: {e}")

    def is_blocked(self, domain: str) -> bool:
        """Check if domain or any parent domain is blacklisted (handles subdomains)."""
        return any(
            domain == b or domain.endswith("." + b)
            for b in self._blocked
        )

    def record_failure(self, domain: str, zero_chars: bool):
        """Track domains that return empty content and auto-blacklist them."""
        if domain in self._blocked:
            return
        if zero_chars:
            self._failures[domain] = self._failures.get(domain, 0) + 1
            if self._failures[domain] >= AUTO_BLACKLIST_THRESHOLD:
                self._blocked.add(domain)
                logger.warning(f"[BLACKLIST] Auto-blacklisted '{domain}' after "
                               f"{self._failures[domain]} zero-content failures.")
                self._save()

BLACKLIST = DomainBlacklist()

# Semantic Search Model (Global Placeholder)
_SEMANTIC_MODEL = None

def get_semantic_model():
    """Lazy load the semantic model with fallback logic."""
    global _SEMANTIC_MODEL
    if _SEMANTIC_MODEL is not None:
        return _SEMANTIC_MODEL

    models_to_try = [
        ('all-MiniLM-L6-v2', "cuda" if torch.cuda.is_available() else "cpu"),
        ('bert-base-uncased', "cpu"),
        ('paraphrase-multilingual-MiniLM-L12-v2', "cpu")
    ]

    for model_name, device in models_to_try:
        try:
            logger.info(f"Attempting to load model '{model_name}' on {device}...")
            _SEMANTIC_MODEL = SentenceTransformer(model_name, device=device)
            logger.info(f"Model '{model_name}' loaded successfully on {device}.")
            return _SEMANTIC_MODEL
        except Exception as e:
            logger.warning(f"Failed to load model '{model_name}': {e}")

    logger.error("All semantic models failed to load.")
    return None

# ─── Tools ────────────────────────────────────────────────────────────────────

class SafeScraper:
    def __init__(self):
        self.pw = None
        self.browser = None

    def start(self):
        if self.pw:
            return
        logger.info("Initializing browser...")
        try:
            self.pw = sync_playwright().start()
            self.browser = self.pw.chromium.launch(headless=True)
        except Exception as e:
            logger.error(f"Failed to start browser: {e}")
            self.stop()
            raise

    def stop(self):
        if self.browser:
            self.browser.close()
        if self.pw:
            self.pw.stop()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def fetch(self, url: str) -> str:
        page = self.browser.new_page()
        Stealth().apply_stealth_sync(page)
        try:
            # Set a generic user agent
            page.set_extra_http_headers({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            })
            page.goto(url, wait_until="domcontentloaded", timeout=SCRAPE_TIMEOUT_MS)
            content = page.content()
            return content
        finally:
            page.close()

# Initialize global scraper instance
scraper_instance = SafeScraper()

def validate_url(url: str) -> bool:
    """Check if the URL is valid and reachable."""
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False
        return True
    except Exception:
        return False

def search_news(query: str, max_results: int = MAX_RESULTS_PER_QUERY) -> list:
    try:
        # DDGS version 5+ sometimes requires specific headers or proxy to bypass impersonation blocks
        # We use a broad try-except to catch the 'edge_127' impersonation error specifically if it arises
        with DDGS(timeout=30) as ddgs:
            logger.info(f"Searching news for: '{query}'")
            # Try to return results. If it fails with the specific impersonation error, 
            # we might need to handle it or fallback.
            results = list(ddgs.news(query, max_results=max_results))
            return results
    except Exception as e:
        logger.error(f"Search error for '{query}': {e}")
        if "edge_127" in str(e):
            logger.warning("Detected DuckDuckGo impersonation error. Check library version or network.")
        return []


def scrape_article(url: str) -> dict:
    """
    Returns a result dict using Playwright for fetching and newspaper for parsing.
    """
    domain = urlparse(url).netloc

    if BLACKLIST.is_blocked(domain):
        return {"url": url, "content": None, "error": "blacklisted"}

    try:
        # Use Playwright to fetch the HTML
        html_content = scraper_instance.fetch(url)
        
        # Use newspaper to parse the HTML
        # Wrap parse separately to handle encoding quirks on some sites
        article = Article(url)
        article.set_html(html_content)
        try:
            article.parse()
        except Exception as parse_err:
            logger.warning(f"Parse error for {url}: {parse_err} — attempting raw text extraction")
            article.text = article.text or ""

        char_count = len(article.text or "")
        zero_chars = char_count == 0
        BLACKLIST.record_failure(domain, zero_chars)

        if not article.text or char_count < MIN_CONTENT_CHARS:
            return {"url": url, "content": None, "error": f"too_short ({char_count} chars)"}

        return {"url": url, "content": article.text, "error": None}

    except Exception as e:
        logger.warning(f"Scraping failed for {url}: {e}")
        return {"url": url, "content": None, "error": str(e)[:100]}


# ─── LLM Helpers ──────────────────────────────────────────────────────────────

def parse_json(content: str) -> dict | None:
    """Robustly extract JSON from LLM response."""
    start = content.find('{')
    end = content.rfind('}') + 1
    if start == -1 or end == 0:
        return None

    raw = content[start:end]

    for attempt in [
        lambda s: json.loads(s),
        lambda s: json.loads(s.replace('\n', '\\n').replace('\r', '\\r')),
        lambda s: json.loads(re.sub(r'\*\*(.*?)\*\*', r'\1', s).replace('\n', '\\n')),
    ]:
        try:
            return attempt(raw)
        except json.JSONDecodeError:
            continue

    print(f"  [parse_json] Failed to parse JSON from response")
    logger.debug(f"Raw content that failed parsing: {content}")
    return None


def call_llm(messages: list, model: str = 'qwen3:8b') -> str:
    """Single LLM call, returns raw content string."""
    response = ollama.chat(model=model, messages=messages)
    return response['message']['content']


# ─── Pipeline Stages ──────────────────────────────────────────────────────────

def stage_extract_count(query: str) -> int:
    """Stage 0: Extract how many summaries the user wants."""

    # Fast path — plain digit
    digit_match = re.search(r'\b(\d+)\b', query)
    if digit_match:
        count = int(digit_match.group(1))
        # No header here as it's almost instant, but let's be consistent
        print(f"\n--- [STAGE 0: INTENT ANALYSIS] ---")
        logger.info(f"[stage_0] Requested count: {count} (digit)")
        return count

    # Slow path — LLM interprets
    try:
        content = call_llm([{"role": "user", "content": (
            f"How many articles/summaries is the user requesting?\n"
            f"Query: \"{query}\"\n"
            "Handle any language, typos, word numbers (e.g. 'thre'=3, 'tiga'=3, 'trois'=3).\n"
            "If no number mentioned, return 1.\n"
            "Respond with JSON only: {\"count\": <integer>}"
        )}])
        result = parse_json(content)
        if result and "count" in result:
            count = int(result["count"])
            logger.info(f"[stage_0] Requested count: {count} (LLM)")
            return count
    except Exception as e:
        logger.error(f"[stage_0] Error: {e}")

    return 1


def stage_plan(query: str, requested_count: int) -> list[str]:
    """
    Stage 1: LLM generates a set of search queries upfront.
    """
    print(f"\n--- [STAGE 1: PLANNING QUERIES] ---")
    logger.info(f"Planning search queries for target count: {requested_count}")

    try:
        content = call_llm([{"role": "user", "content": (
            f"You are a search query planner for a news research task.\n\n"
            f"User query: \"{query}\"\n"
            f"Target: find {requested_count} relevant news articles.\n\n"
            f"Generate up to {MAX_SEARCH_QUERIES} diverse search queries to find relevant articles.\n"
            "Rules:\n"
            "- Make queries diverse — vary keywords, angles, and specificity\n"
            "- If query mentions multiple entities, include queries for each separately AND together\n"
            "- Do NOT repeat similar queries\n"
            "- Order from most to least specific\n"
            f"Respond with JSON only: {{\"queries\": [\"query1\", \"query2\", ...]}}"
        )}])

        result = parse_json(content)
        if result and "queries" in result:
            queries = result["queries"][:MAX_SEARCH_QUERIES]
            logger.info(f"[stage_1] Generated {len(queries)} queries.")
            return queries
    except Exception as e:
        logger.error(f"[stage_1] Error: {e}")

    return [query]


def stage_search(queries: list[str]) -> list[str]:
    """
    Stage 2: Run all searches and collect unique non-blacklisted URLs.
    Dynamically expands result requests to compensate for blacklisted URLs.
    """
    print(f"\n--- [STAGE 2: SEARCHING NEWS] ---")
    logger.info(f"Searching with {len(queries)} queries...")

    seen_urls = set()
    all_urls = []
    total_blacklisted = 0

    for query in queries:
        # Request extra results to compensate for blacklisted domains found so far
        compensation = min(total_blacklisted, 10)  # cap compensation at 10 extra
        fetch_count = MAX_RESULTS_PER_QUERY + compensation

        results = search_news(query, max_results=fetch_count)
        blacklisted_this_query = 0

        for item in results:
            url = item.get('url', '')
            if not url or url in seen_urls:
                continue

            domain = urlparse(url).netloc
            if BLACKLIST.is_blocked(domain):
                blacklisted_this_query += 1
                logger.info(f"  [SKIP-BLACKLIST] {domain}")
                continue

            if validate_url(url):
                seen_urls.add(url)
                all_urls.append(url)

        if blacklisted_this_query:
            logger.info(f"  Skipped {blacklisted_this_query} blacklisted URLs "
                        f"(fetched {fetch_count} results to compensate)")
        total_blacklisted += blacklisted_this_query

    logger.info(f"[stage_2] Collected {len(all_urls)} unique valid URLs "
                f"({total_blacklisted} blacklisted skipped total)")
    return all_urls


def stage_scrape(urls: list[str]) -> list[dict]:
    """
    Stage 3: Scrape all URLs sequentially using Playwright.
    Sequential is used to avoid Playwright sync-API threading issues.
    """
    print(f"\n--- [STAGE 3: SCRAPING ARTICLES] ---")
    logger.info(f"Scraping {len(urls)} URLs sequentially...")

    successful = []

    for i, url in enumerate(urls, 1):
        logger.info(f"[{i}/{len(urls)}] Processing: {url[:60]}...")
        result = scrape_article(url)
        if result["content"]:
            char_count = len(result["content"])
            is_paywall = char_count < PAYWALL_CHAR_THRESHOLD
            logger.info(f"  [OK] {result['url'][:60]}... ({char_count} chars{'- possible paywall' if is_paywall else ''})")
            successful.append(result)
        else:
            logger.info(f"  [FAIL] {result['url'][:60]}... ({result['error']})")


    logger.info(f"[stage_3] Successfully scraped {len(successful)}/{len(urls)} articles")
    return successful


def stage_filter(scraped: list[dict], query: str, requested_count: int) -> list[dict]:
    """
    Stage 4: Filter and rank scraped articles using semantic similarity.
    Also deduplicates articles that cover the same story.
    """
    print(f"\n--- [STAGE 4: RANKING RELEVANCE] ---")
    logger.info(f"Filtering {len(scraped)} articles with semantic search...")

    if not scraped:
        return []

    model = get_semantic_model()
    if model is None:
        logger.error("Skipping semantic filter as no model could be loaded.")
        return scraped[:requested_count]

    article_contents = [article['content'] for article in scraped]

    # Encode query and all articles
    query_embedding = model.encode(query, convert_to_tensor=True)
    doc_embeddings = model.encode(article_contents, convert_to_tensor=True)

    # Rank all articles by relevance to the query
    scores = util.cos_sim(query_embedding, doc_embeddings).flatten()
    ranked_indices = scores.argsort(descending=True).tolist()

    # Deduplicate: pick articles in rank order, skip if too similar to already-selected
    DUPLICATE_THRESHOLD = 0.85  # cosine similarity above this = same story
    selected_articles = []
    selected_embeddings = []

    for idx in ranked_indices:
        if len(selected_articles) >= requested_count:
            break

        article = scraped[idx]
        score = scores[idx]
        article_emb = doc_embeddings[idx]

        # Check if this article is too similar to any already selected
        is_duplicate = False
        for selected_emb in selected_embeddings:
            similarity = util.cos_sim(article_emb, selected_emb).item()
            if similarity >= DUPLICATE_THRESHOLD:
                is_duplicate = True
                logger.info(f"  [SKIP-DUPLICATE] {article['url'][:70]}... (similarity={similarity:.2f})")
                break

        if not is_duplicate:
            logger.info(f"  - Score: {score:.4f} | {article['url'][:70]}...")
            selected_articles.append(article)
            selected_embeddings.append(article_emb)

    # Flag paywall articles
    for article in selected_articles:
        if len(article["content"]) < PAYWALL_CHAR_THRESHOLD:
            article["content"] += "\n[WARNING: possible paywall — content may be incomplete]"

    logger.info(f"[stage_4] {len(selected_articles)} unique relevant articles selected "
                f"(from {len(scraped)} scraped, duplicates removed)")
    return selected_articles



def stage_summarize(articles: list[dict], query: str, requested_count: int) -> str:
    """
    Stage 5: Final summarization.
    """
    print(f"\n--- [STAGE 5: GENERATING SUMMARY] ---")
    logger.info(f"Summarizing {len(articles)} articles...")

    if not articles:
        return "No relevant articles could be retrieved for your query."

    articles_text = "\n\n".join(
        f"[Article {i+1} — {a['url']}]:\n{a['content'][:4000]}"
        for i, a in enumerate(articles)
    )

    try:
        content = call_llm([{"role": "user", "content": (
            f"The user asked: \"{query}\"\n\n"
            f"You have {len(articles)} relevant article(s) below. "
            f"The user requested {requested_count} summaries.\n\n"
            f"{articles_text}\n\n"
            "Rules:\n"
            "- Summarize ONLY from the article content above\n"
            "- Never invent or infer information not present in the articles\n"
            "- If fewer articles than requested, summarize only what exists — do not pad\n"
            "- Keep each summary concise and factual\n"
            "Respond with JSON only: {\"final_answer\": \"your numbered summaries here\"}"
        )}])

        result = parse_json(content)
        if result and "final_answer" in result:
            return result["final_answer"]
    except Exception as e:
        logger.error(f"[stage_5] Error: {e}")

    return "Agent could not generate a summary."


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_agent(query: str) -> str:
    logger.info(f"Starting agent for query: {query}")
    
    try:
        # Initialize scraper
        scraper_instance.start()

        # Stage 0: Extract count
        requested_count = stage_extract_count(query)

        # Stage 1: Plan search queries
        queries = stage_plan(query, requested_count)

        # Stage 2: Search all queries
        urls = stage_search(queries)

        if not urls:
            return "No search results found. Please try a different query."

        # Stage 3: Scrape all URLs
        scraped = stage_scrape(urls)

        if not scraped:
            return "Could not retrieve any articles. Sites may be blocking scraping."

        # Stage 4: Filter
        relevant = stage_filter(scraped, query, requested_count)

        # Stage 5: Summarize
        return stage_summarize(relevant, query, requested_count)

    except Exception as e:
        logger.exception("An unexpected error occurred during agent execution")
        return f"An error occurred: {str(e)}"
    finally:
        # Ensure scraper is stopped
        scraper_instance.stop()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        print("\nEnter news query (e.g., 'summarize 3 news about tesla'): ", end="", flush=True)
        query = sys.stdin.readline().strip()
    
    if query:
        result = run_agent(query)
        print(f"\n{'='*50}\nFINAL ANSWER:\n{result}\n{'='*50}\n")
    else:
        logger.error("No query provided. Exiting.")
