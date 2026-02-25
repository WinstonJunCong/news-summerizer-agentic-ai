"""
quixel_debug.py
═══════════════════════════════════════════════════════════════════════════════
Isolated diagnostic for Quixel.com scraping.

Runs a HEADED Playwright browser so you can SEE what happens, then:
  1. Navigates to a real Quixel asset page
  2. Waits for the page to fully settle (network + animations)
  3. Screenshots after initial load
  4. Iterates through every popup dismiss selector and logs matches
  5. Screenshots after each click attempt
  6. Dumps ALL img tags found (src, width, height, alt)
  7. Logs all <meta> og:image / twitter:image values
  8. Prints the text content extracted from the page

Run:   python quixel_debug.py
"""

import time
import sys
import re
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth
from bs4 import BeautifulSoup

# ─── Configuration ────────────────────────────────────────────────────────────

TEST_URLS = [
    # A real individual asset page
    "https://quixel.com/megascans/home?assetId=vcvodh0",
    # The main megascans landing
    "https://quixel.com/megascans/home",
    # Collections page
    "https://quixel.com/megascans/collections",
]

SCREENSHOT_DIR = Path(__file__).parent / "quixel_debug_screenshots"
SCREENSHOT_DIR.mkdir(exist_ok=True)

# Every popup/overlay selector we try, in priority order
POPUP_SELECTORS_CSS = [
    # Epic / Quixel specific -- try a broad set
    "button[aria-label='Close']",
    "button[aria-label='close']",
    "[data-testid='announcement-close']",
    "button.announcement-close",
    ".modal-close",
    ".cookie-close",
    # Generic close patterns
    "button.close",
    "[class*='dismiss']",
    "[class*='modal'] [class*='close']",
    "[class*='overlay'] [class*='close']",
    "[id*='cookie'] button",
    "[aria-label*='close' i]",
    "[aria-label*='dismiss' i]",
    # Quixel specific modal patterns seen in network traces
    "[class*='MuiDialog'] button",
    "[role='dialog'] button",
    "[role='alertdialog'] button",
    "[class*='Banner'] button",
    "[class*='Modal'] button",
    "[class*='Popup'] button",
]

# XPath selectors for text-based button matching
POPUP_SELECTORS_XPATH = [
    "//button[contains(text(),'Stay here')]",
    "//button[contains(text(),'Accept')]",
    "//button[contains(text(),'Got it')]",
    "//button[contains(text(),'OK')]",
    "//button[contains(text(),'Close')]",
    "//button[contains(text(),'Continue')]",
    "//button[contains(text(),'Dismiss')]",
    "//span[contains(text(),'Accept') and ancestor::button]/..",
    "//span[contains(text(),'Got it') and ancestor::button]/..",
    "//span[contains(text(),'Close') and ancestor::button]/..",
]

WAIT_MS_AFTER_LOAD = 4000   # 4s after domcontentloaded for JS to finish
WAIT_MS_AFTER_CLICK = 800   # small wait after each popup close


def take_screenshot(page, label: str, index: int) -> str:
    safe_label = re.sub(r"[^a-zA-Z0-9_-]", "_", label)[:50]
    filename = SCREENSHOT_DIR / f"{index:03d}_{safe_label}.png"
    page.screenshot(path=str(filename), full_page=False)
    print(f"  📸 Screenshot saved: {filename.name}")
    return str(filename)


def try_dismiss_popups(page, shot_counter: list) -> None:
    """Iterate every selector, log whether it found an element, click if visible."""
    print("\n─── Trying CSS selectors ─────────────────────────────────────────")
    for sel in POPUP_SELECTORS_CSS:
        try:
            elements = page.query_selector_all(sel)
            if elements:
                print(f"  [FOUND] {len(elements)} element(s) for: {sel}")
                for el in elements:
                    try:
                        visible = el.is_visible()
                        text = (el.inner_text() or "").strip()[:60]
                        print(f"    → visible={visible}  text='{text}'")
                        if visible:
                            el.click(timeout=1500)
                            print(f"    ✅ CLICKED!")
                            shot_counter[0] += 1
                            take_screenshot(page, f"after_click_{sel[:30]}", shot_counter[0])
                            page.wait_for_timeout(WAIT_MS_AFTER_CLICK)
                    except Exception as click_err:
                        print(f"    ⚠  Click failed: {click_err}")
            else:
                print(f"  [       ] 0 matches: {sel}")
        except Exception as sel_err:
            print(f"  [ERROR] Selector '{sel}': {sel_err}")

    print("\n─── Trying XPath selectors ───────────────────────────────────────")
    for xpath in POPUP_SELECTORS_XPATH:
        try:
            elements = page.query_selector_all(xpath)
            if elements:
                print(f"  [FOUND] {len(elements)} element(s) for xpath: {xpath}")
                for el in elements:
                    try:
                        visible = el.is_visible()
                        text = (el.inner_text() or "").strip()[:60]
                        print(f"    → visible={visible}  text='{text}'")
                        if visible:
                            el.click(timeout=1500)
                            print(f"    ✅ CLICKED!")
                            shot_counter[0] += 1
                            take_screenshot(page, f"after_xpath_click", shot_counter[0])
                            page.wait_for_timeout(WAIT_MS_AFTER_CLICK)
                    except Exception as click_err:
                        print(f"    ⚠  Click failed: {click_err}")
            else:
                print(f"  [       ] 0 matches: {xpath}")
        except Exception as sel_err:
            print(f"  [ERROR] XPath '{xpath}': {sel_err}")


def analyse_page(page, url: str) -> None:
    html = page.content()
    soup = BeautifulSoup(html, "html.parser")

    print("\n─── Meta tags (og:image / twitter:image) ─────────────────────────")
    og = soup.find("meta", property="og:image")
    tw = soup.find("meta", attrs={"name": "twitter:image"})
    print(f"  og:image       = {og['content'] if og and og.get('content') else 'NOT FOUND'}")
    print(f"  twitter:image  = {tw['content'] if tw and tw.get('content') else 'NOT FOUND'}")

    print("\n─── All <img> tags on page ───────────────────────────────────────")
    imgs = soup.find_all("img")
    if imgs:
        for i, img in enumerate(imgs[:30]):   # cap at 30 to avoid spam
            src = img.get("src", "")
            alt = img.get("alt", "")[:40]
            w = img.get("width", "?")
            h = img.get("height", "?")
            cls = " ".join(img.get("class", []))[:40]
            print(f"  [{i+1:02d}] {w}×{h}  alt='{alt}'  class='{cls}'")
            print(f"       src={src[:100]}")
    else:
        print("  ⚠  No <img> tags found at all!")

    print("\n─── Page text (first 800 chars) ──────────────────────────────────")
    body = soup.find("body")
    if body:
        raw = re.sub(r"\s+", " ", body.get_text(separator=" ")).strip()
        print(f"  {raw[:800]}")
    else:
        print("  ⚠  No <body> tag found!")

    print(f"\n─── Rendered HTML size: {len(html):,} bytes ─────────────────────────")

    # Also dump all rendered text from page.inner_text() — this captures
    # JS-rendered content that BeautifulSoup misses
    try:
        body_text = page.inner_text("body") or ""
        body_text = re.sub(r"\s+", " ", body_text).strip()
        print(f"\n─── JS-rendered body text (first 800 chars) ─────────────────────")
        print(f"  {body_text[:800]}")
        print(f"\n  Total JS-rendered text length: {len(body_text):,} chars")
    except Exception as e:
        print(f"  ⚠  Could not get inner_text: {e}")


def debug_url(url: str, shot_counter: list, pw) -> None:
    print(f"\n{'═'*70}")
    print(f"  🔍  Testing URL: {url}")
    print(f"{'═'*70}")

    browser = pw.chromium.launch(headless=False, slow_mo=200)  # HEADED + slow-mo for visibility
    context = browser.new_context(
        viewport={"width": 1400, "height": 900},
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    )
    page = context.new_page()
    Stealth().apply_stealth_sync(page)

    try:
        print(f"\n[1] Navigating (domcontentloaded)...")
        page.goto(url, wait_until="domcontentloaded", timeout=30000)

        shot_counter[0] += 1
        take_screenshot(page, "01_after_domcontentloaded", shot_counter[0])

        print(f"[2] Waiting {WAIT_MS_AFTER_LOAD}ms for JS/animations to settle...")
        page.wait_for_timeout(WAIT_MS_AFTER_LOAD)

        shot_counter[0] += 1
        take_screenshot(page, "02_after_wait", shot_counter[0])

        print("[3] Attempting popup dismissal...")
        try_dismiss_popups(page, shot_counter)

        shot_counter[0] += 1
        take_screenshot(page, "03_after_popup_dismissal", shot_counter[0])

        print("[4] Waiting an extra 2s then analysing...")
        page.wait_for_timeout(2000)
        analyse_page(page, url)

        shot_counter[0] += 1
        take_screenshot(page, "04_final_state", shot_counter[0])

        print("\n⏸  Browser is still open — inspect it, then press ENTER to continue.")
        input()

    except Exception as e:
        print(f"  ❌ Error: {e}")
        shot_counter[0] += 1
        take_screenshot(page, "ERROR", shot_counter[0])
    finally:
        context.close()
        browser.close()


def main():
    # Allow passing a custom URL as CLI arg
    urls = sys.argv[1:] if len(sys.argv) > 1 else TEST_URLS

    print("=" * 70)
    print("  Quixel.com Scraping Diagnostic")
    print(f"  Screenshots will be saved to: {SCREENSHOT_DIR}")
    print("=" * 70)

    shot_counter = [0]   # mutable list so nested functions can increment it

    with sync_playwright() as pw:
        for url in urls:
            debug_url(url, shot_counter, pw)

    print(f"\n✅ Done. {shot_counter[0]} screenshots saved in: {SCREENSHOT_DIR}")


if __name__ == "__main__":
    main()
