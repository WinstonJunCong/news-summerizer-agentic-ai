from ddgs import DDGS
from newspaper import Article, Config, ArticleException
import ollama
from urllib.parse import urlparse

blacklist = ["twitter.com", "facebook.com", "linkedin.com", "instagram.com", "youtube.com"]
max_results = 5

def search_news(query, max_results):
    try:
        with DDGS(timeout=20) as ddgs:
            print("Searching for news...")
            return list(ddgs.news(query, max_results=max_results))
    except (Exception, KeyboardInterrupt) as e:
        print(f"An error occurred during the news search: {e}")
        return []

def scrape_article(url):
    domain = urlparse(url).netloc
    if domain in blacklist:
        print(f"Skipping blacklisted domain: {domain}")
        return None

    try:
        print(f"Reading: {url}")
        config = Config()
        # Make requests look more like they are coming from a real browser
        config.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'DNT': '1', # Do Not Track
        }
        article = Article(url, config=config)
        article.download()
        article.parse()
        return article.text
    except ArticleException as e:
        error_string = str(e).lower()
        if '403' in error_string:
            print(f"Error 403: Access to {url} is forbidden. This site likely has anti-scraping measures.")
        elif '404' in error_string:
            print(f"Error 404: Article at {url} not found.")
        else:
            print(f"Failed to download or parse the article: {e}")
        blacklist.append(domain)  # Add to blacklist to avoid future attempts
        return None
    except Exception as e:
        print(f"An unexpected error occurred while processing {url}: {e}")
        blacklist.append(domain)  # Add to blacklist to avoid future attempts
        return None

def free_search_and_summarize(query):
    results = search_news(query, max_results)
    if not results:
        return "No news found."

    for result in results:
        text = scrape_article(result['url'])
        if text:
            return summarize(text)
            
    return "Could not scrape any articles from the search results."

def summarize(text):
    prompt = f"Summarize this news: {text[:5000]}" # Limiting text for speed
    print("Generating summary...")
    response = ollama.generate(model='qwen3:8b', prompt=prompt)
    return response['response']

print(free_search_and_summarize("tell me news about the OpenClaw"))