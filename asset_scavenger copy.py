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
from urllib.parse import urlparse, urljoin
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth
from bs4 import BeautifulSoup
import requests
from PIL import Image
from io import BytesIO

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

MAX_SEARCH_QUERIES = 10    # how many search queries the planner generates
MAX_RESULTS_PER_QUERY = 10 # URLs per search (asset sites are sparser than news)
PAYWALL_CHAR_THRESHOLD = 20 
MIN_CONTENT_CHARS = 20    # Asset pages often have VERY little text (just a download button and specs)
SCRAPE_TIMEOUT_MS = 30000 # 30 seconds

# Deduplication Threshold (Cosine Similarity)
DUPLICATE_THRESHOLD = 0.85
# Minimum Relevance Threshold (Cosine Similarity) to keep an asset
MIN_RELEVANCE_SCORE = 0.225

BLACKLIST_FILE = "blacklist.json"
AUTO_BLACKLIST_THRESHOLD = 5  # failures before domain is auto-blacklisted (increased for sparse asset pages)

# Domains always blocked (social media / generic news)
_STATIC_BLACKLIST = {
    "twitter.com", "facebook.com", "linkedin.com",
    "instagram.com", "youtube.com", "reddit.com",
    "x.com", "tiktok.com", "pinterest.com", "pinterest.co.uk",
    "news.yahoo.com", "finance.yahoo.com", "msn.com", "cnn.com", "foxnews.com", "bbc.com", "nytimes.com"
}

# Domains NEVER to be auto-blacklisted (vital 3D hubs)
_STATIC_WHITELIST = {
    "sketchfab.com", "turbosquid.com", "polyhaven.com", "ambientcg.com", 
    "cgtrader.com", "artstation.com", "blendswap.com", "free3d.com"
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
            
        # NEVER auto-blacklist our critical 3D asset hubs
        if any(domain == w or domain.endswith("." + w) for w in _STATIC_WHITELIST):
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
        ('clip-ViT-B-32', "cuda" if torch.cuda.is_available() else "cpu"),
        # Fallbacks in case CLIP fails to download
        ('all-MiniLM-L6-v2', "cpu"),
        ('bert-base-uncased', "cpu")
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

    # Popup selectors to dismiss before scraping. Ordered from most to least specific.
    POPUP_SELECTORS = [
        # Quixel / Epic Games announcements
        "[data-testid='announcement-close']",
        "button[aria-label='Close']",
        "button.announcement-close",
        ".modal-close",
        ".cookie-close",
        # Generic close/dismiss patterns
        "button.close",
        "[class*='dismiss']",
        "[class*='modal'] [class*='close']",
        "[id*='cookie'] button",
        "[aria-label*='close' i]",
        "[aria-label*='dismiss' i]",
        # Target the "Stay here" button for quixel specific popup
        "//button[text()='Stay here']"
    ]

    def _dismiss_popups(self, page) -> None:
        """Attempt to click any popup / modal close buttons that block the page content."""
        for selector in self.POPUP_SELECTORS:
            try:
                el = page.query_selector(selector)
                if el and el.is_visible():
                    el.click(timeout=1500)
                    logger.info(f"  [POPUP-DISMISSED] Clicked: {selector}")
                    page.wait_for_timeout(500)  # wait for animation
            except Exception:
                pass  # silence — most selectors won't exist on most pages

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
            # Dismiss any overlays / popups that could block content extraction
            self._dismiss_popups(page)
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
            results = list(ddgs.text(query, max_results=max_results))
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
            # Fallback: manually strip tags, preserving text inside attributes like 'alt' if possible
            body_match = re.search(r'<body[^>]*>(.*?)</body>', html_content, re.IGNORECASE | re.DOTALL)
            raw_text = body_match.group(1) if body_match else html_content
            
            # Extract basic text
            clean_text = re.sub(r'<[^>]+>', ' ', raw_text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            article.text = clean_text

        char_count = len(article.text or "")
        zero_chars = char_count == 0
        BLACKLIST.record_failure(domain, zero_chars)

        # New Feature: Extract the Main Preview Image using BeautifulSoup
        preview_image_url = None
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Priority 1: OpenGraph Image (usually the highest quality thumbnail)
        og_image = soup.find('meta', property='og:image')
        if og_image and og_image.get('content'):
            preview_image_url = og_image['content']
        else:
            # Priority 2: Twitter Image
            twitter_image = soup.find('meta', attrs={'name': 'twitter:image'})
            if twitter_image and twitter_image.get('content'):
                preview_image_url = twitter_image['content']
            else:
                # Priority 3: The first substantial image on the page
                # Avoid tiny tracking pixels or ui icons by looking for descriptive src/alt or size
                for img in soup.find_all('img'):
                    src = img.get('src')
                    if src and not src.startswith('data:') and '.svg' not in src.lower():
                        # Resolve relative URLs
                        preview_image_url = urljoin(url, src)
                        break

        # If zero chars and no preview image, fail it. If it has a preview image, we MIGHT save it depending on visual score
        if char_count < MIN_CONTENT_CHARS and not preview_image_url:
            return {"url": url, "content": None, "preview_image_url": None, "error": f"too_short ({char_count} chars) and no image"}

        return {"url": url, "content": article.text, "preview_image_url": preview_image_url, "error": None}

    except Exception as e:
        logger.warning(f"Scraping failed for {url}: {e}")
        return {"url": url, "content": None, "preview_image_url": None, "error": str(e)[:100]}


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

def stage_plan(query: str) -> list[str]:
    """
    Stage 1: LLM generates a set of search queries upfront.
    """
    print(f"\n--- [STAGE 1: PLANNING QUERIES] ---")
    logger.info(f"Planning search queries for: {query}")

    try:
        content = call_llm([{"role": "user", "content": (
            f"You are a search query planner for a 3D Asset Scavenger agent.\n\n"
            f"User query: \"{query}\"\n"
            f"Target: find as many relevant 3D assets, materials, or textures online as possible.\n\n"
            f"Generate up to {MAX_SEARCH_QUERIES} diverse search queries to find these assets.\n"
            "Rules:\n"
            "- Use keywords like: \"free download\", \"CC0\", \"PBR\", \"texture\", \"seamless\", \"4k\", \"obj\", \"fbx\", \"blend\".\n"
            "- You may choose to mix broad web searches with targeted site searches (e.g. site:polyhaven.com, site:ambientcg.com, site:sketchfab.com, site:fab.com, site:quixel.com).\n"
            "- Do NOT generate repetitive queries. Explore different synonyms.\n"
            "- Order from most to least specific.\n"
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
        
        # Sleep to avoid DDGS HTTP 429 Too Many Requests rate limits
        time.sleep(8.0)

        for item in results:
            url = item.get('href', '')
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


def stage_filter(scraped: list[dict], query: str) -> list[dict]:
    """
    Stage 4: Filter and rank scraped articles using Visual Similarity (CLIP).
    Deduplicates articles that cover the same story and drops articles below MIN_RELEVANCE_SCORE.
    """
    print(f"\n--- [STAGE 4: RANKING RELEVANCE] ---")
    logger.info(f"Filtering {len(scraped)} articles with MULTIMODAL visual search (CLIP)...")

    if not scraped:
        return []

    model = get_semantic_model()
    if model is None:
        logger.error("Skipping semantic filter as no model could be loaded.")
        return scraped

    # Encode query text once
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Download and score images
    final_scores = []
    doc_embeddings = [] # We'll store embeddings for deduplication
    
    # We may need a slightly higher or different baseline for CLIP (since it's a different vector space)
    # Let's recalibrate the minimum relevance visually to 0.22 (CLIP scores often hover around 0.20-0.28 for okay matches)
    VISUAL_THRESHOLD = 0.22

    for article in scraped:
        img_url = article.get("preview_image_url")
        content = article.get("content", "")
        img_embedding = None
        score = 0.0

        if img_url:
            try:
                # Download the image into memory
                response = requests.get(img_url, timeout=5)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content)).convert("RGB")
                    # Optionally resize to speed up encoding (CLIP expects 224x224 anyway)
                    img.thumbnail((400, 400)) 
                    # Encode the Image using the Vision Model
                    img_embedding = model.encode(img, convert_to_tensor=True)
                    # Score Image vs Text Query
                    score = util.cos_sim(query_embedding, img_embedding).item()
                    logger.info(f"  [VISUAL-SCORE] {score:.4f} | {article['url']}")
                else:
                    logger.warning(f"  [IMG-FAIL] HTTP {response.status_code} for {img_url}")
            except Exception as e:
                logger.warning(f"  [IMG-FAIL] Could not process image {img_url}: {e}")

        if img_embedding is None:
            # Fallback: encode the text if image extraction fails
            # Note: CLIP *can* encode text vs text, but it's not strictly built for long-form document retrieval.
            # We'll encode a truncated version of the text to give it a fighting chance.
            text_snippet = content[:500] if content else urlparse(article['url']).netloc
            img_embedding = model.encode(text_snippet, convert_to_tensor=True)
            score = util.cos_sim(query_embedding, img_embedding).item()
            logger.info(f"  [TEXT-FALLBACK-SCORE] {score:.4f} | {article['url']}")

        final_scores.append(score)
        doc_embeddings.append(img_embedding)

    # Sort indices by score descending
    ranked_indices = sorted(range(len(final_scores)), key=lambda i: final_scores[i], reverse=True)

    # Deduplicate: pick articles in rank order, skip if too similar to already-selected
    selected_articles = []
    selected_embeddings = []

    for idx in ranked_indices:
        article = scraped[idx]
        score = final_scores[idx]
        article_emb = doc_embeddings[idx]

        # Stop looking if we've fallen below the relevance threshold
        if score < VISUAL_THRESHOLD:
            logger.info(f"  [SKIP-LOW-SCORE] {article['url'][:70]}... (score={score:.4f})")
            continue

        # Deduplication is DISABLED — show all articles above the relevance threshold
        # (uncomment the block below to re-enable deduplication)
        # is_duplicate = False
        # for selected_emb in selected_embeddings:
        #     similarity = util.cos_sim(article_emb, selected_emb).item()
        #     if similarity >= DUPLICATE_THRESHOLD:
        #         is_duplicate = True
        #         logger.info(f"  [SKIP-DUPLICATE] {article['url'][:70]}... (similarity={similarity:.2f})")
        #         break

        # if not is_duplicate:
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



def stage_summarize(articles: list[dict], query: str) -> str:
    """
    Stage 5: Final summarization.
    """
    print(f"\n--- [STAGE 5: GENERATING SUMMARY] ---")
    logger.info(f"Summarizing {len(articles)} articles...")

    if not articles:
        return "No relevant assets could be found for your query."

    articles_text = "\n\n".join(
        f"--- ASSET FILE {i+1} ---\nURL: {a['url']}\nCONTENT:\n{a['content'][:2500]}"
        for i, a in enumerate(articles)
    )

    try:
        content = call_llm([{"role": "user", "content": (
            f"You are an expert 3D Asset Curator. The user requested: \"{query}\"\n\n"
            f"I have provided the scraped content of {len(articles)} highly-relevant webpages that likely contain these assets.\n\n"
            f"{articles_text}\n\n"
            "Your task:\n"
            "1. Read EVERY provided article and list ALL assets you can identify from them that are relevant to the user's prompt.\n"
            "   **DO NOT filter them down** — if 10 articles each have a matching asset, list all 10.\n"
            "   **HOWEVER**: If the user asks for a complex combination (e.g., 'scratched glass') and you cannot find an exact combined match, you MAY extract separate component assets that could be combined to achieve the effect. Clearly label them as component pieces.\n"
            "2. Identify the **License Type** for each asset (e.g., CC0, CC-BY, Royalty-Free, Paid, or Unknown).\n"
            "3. Output a curated Markdown list categorizing the assets by their license type.\n"
            "4. For each asset, provide a brief description and the exact SOURCE URL. DO NOT invent URLs. ONLY use the provided URLs.\n\n"
            "Format Example:\n"
            "### 🟢 CC0 / Free Commercial Use\n"
            "* **[Exact Asset Name]** - Brief description. [Source Link](URL)\n\n"
            "If NO truly relevant assets exist in the text, just reply: 'No relevant assets were found.'\n\n"
            "Do NOT include JSON wrappers. Output the raw markdown text."
        )}])

        # For the final summary, since we removed JSON constraints, we just return the raw text if it's not JSON
        result = parse_json(content)
        if result and "final_answer" in result:
            return result["final_answer"]
        return content  # Fallback to returning raw Markdown if parser fails
    except Exception as e:
        logger.error(f"[stage_5] Error: {e}")

    return "Agent could not generate a summary."


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_agent(query: str) -> str:
    logger.info(f"Starting agent for query: {query}")
    
    try:
        # Initialize scraper
        scraper_instance.start()

        # Stage 1: Plan search queries
        queries = stage_plan(query)

        # Stage 2: Search all queries
        urls = stage_search(queries)

        if not urls:
            return "No search results found. Please try a different query."

        # Stage 3: Scrape all URLs
        scraped = stage_scrape(urls)

        if not scraped:
            return "Could not retrieve any articles. Sites may be blocking scraping."

        # Stage 4: Filter
        relevant = stage_filter(scraped, query)
        return relevant
        # # Stage 5: Summarize
        # return stage_summarize(relevant, query)

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
        print("\nEnter asset query (e.g., 'find me a rusty metal pipe texture seamless cc0'): ", end="", flush=True)
        query = sys.stdin.readline().strip()
    
    if query:
        result = run_agent(query)
        # result is now a list[dict] (raw articles after filtering)
        logger.info(f"\n{'='*50}")
        logger.info(f"FINAL RESULTS ({len(result)} assets found):")
        logger.info('='*50)
        if isinstance(result, list):
            for i, article in enumerate(result, 1):
                url = article.get('url', 'N/A')
                score = article.get('visual_score', 'N/A')
                has_img = bool(article.get('preview_image_url'))
                logger.info(f"[{i:02d}] {'📷' if has_img else '📝'} {url}")
        else:
            logger.info(result)
        logger.info('='*50)
    else:
        logger.error("No query provided. Exiting.")
