"""
Riyasewana vehicle crawler: paginates search pages, opens each ad, extracts fields,
and can save results as JSON under the project `data/` folder.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Set
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse, urljoin

from bs4 import BeautifulSoup
from loguru import logger
from playwright.async_api import async_playwright

# Riyasewana serves a minimal shell to the default Playwright UA; use a desktop Chrome string.
DESKTOP_CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Default search URLs (same HTML pattern: ul.v-list). Spare-parts page is different.
SEARCH_URLS = [
    "https://riyasewana.com/search/cars",
    "https://riyasewana.com/search/suvs",
    "https://riyasewana.com/search/vans",
    "https://riyasewana.com/search/motorcycles",
    "https://riyasewana.com/search/lorries",
    "https://riyasewana.com/search/three-wheels",
    "https://riyasewana.com/search/pickups",
    "https://riyasewana.com/search/heavy-duties",
]


def project_root() -> Path:
    """Riyasewana repo root (parent of `src/`)."""
    return Path(__file__).resolve().parents[3]


def data_dir() -> Path:
    d = project_root() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _abs_url(href: str, base: str = "https://riyasewana.com") -> str:
    if not href or href.startswith("#"):
        return ""
    href = href.strip()
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("http://") or href.startswith("https://"):
        return href
    return urljoin(base if base.endswith("/") else base + "/", href)


def _search_url_for_page(search_url: str, page: int) -> str:
    parsed = urlparse(search_url)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc or "riyasewana.com"
    path = parsed.path or "/"
    base_q = {k: v[0] for k, v in parse_qs(parsed.query).items() if k != "page"}
    if page <= 1:
        query = urlencode(base_q) if base_q else ""
        return urlunparse((scheme, netloc, path, "", query, ""))
    base_q["page"] = str(page)
    return urlunparse((scheme, netloc, path, "", urlencode(base_q), ""))


def _json_slug_from_search_url(search_url: str) -> str:
    path = (urlparse(search_url).path or "listings").strip("/").replace("/", "_")
    return path or "listings"


class RiyasewanaVehicleCrawler:
    """Crawl one Riyasewana search URL and parse each `/buy/...` listing page."""

    BUY_PREFIX = "/buy/"

    def __init__(
        self,
        max_listings: int = 500,
        request_delay: float = 4.0,
        request_delay_jitter: float = 1.5,
        max_retries: int = 5,
        ban_backoff_base_s: float = 30.0,
        page_load_timeout_ms: int = 60_000,
    ):
        self.max_listings = max_listings
        self.request_delay = request_delay
        self.request_delay_jitter = request_delay_jitter
        self.max_retries = max_retries
        self.ban_backoff_base_s = ban_backoff_base_s
        self.page_load_timeout_ms = page_load_timeout_ms
    
    def _sleep_s(self) -> float:
        return max(0.0, self.request_delay + random.uniform(0, self.request_delay_jitter))

    def _looks_like_rate_limited(self, html: str) -> bool:
        txt = (html or "").lower()
        return (
            "rate limit exceeded" in txt
            or "temporary ban" in txt
            or "too many requests" in txt
            or "access denied" in txt
        )

    async def _goto_with_backoff(self, page, url: str) -> str:
        """
        Navigate with backoff when rate limited / temporarily banned.
        Returns page HTML on success, raises after max retries.
        """
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=self.page_load_timeout_ms)
                await page.wait_for_timeout(1200)
                html = await page.content()
                if self._looks_like_rate_limited(html):
                    raise RuntimeError("Rate limit exceeded / temporary ban page detected")
                return html
            except Exception as e:
                last_err = e
                if attempt >= self.max_retries:
                    break
                backoff = self.ban_backoff_base_s * (2 ** attempt) + random.uniform(0, 5.0)
                logger.warning(f"Rate limited when fetching {url}. Backing off for {backoff:.1f}s (attempt {attempt+1}/{self.max_retries}).")
                await asyncio.sleep(backoff)
        raise last_err if last_err else RuntimeError("Navigation failed")

    def _max_page(self, soup: BeautifulSoup) -> int:
        nav = soup.select_one("div.pagination")
        if not nav:
            return 1
        best = 1
        for a in nav.find_all("a", href=True):
            m = re.search(r"[?&]page=(\d+)", a["href"])
            if m:
                best = max(best, int(m.group(1)))
        return max(best, 1)

    def _listing_urls(self, html: str) -> List[str]:
        soup = BeautifulSoup(html, "html.parser")
        out: List[str] = []
        seen: Set[str] = set()
        for li in soup.select("ul.v-list li.v-card"):
            for sel in ("div.v-card-title a[href]", "div.v-card-img a[href]"):
                a = li.select_one(sel)
                if not a or not a.get("href"):
                    continue
                u = _abs_url(a["href"])
                if not u or self.BUY_PREFIX not in urlparse(u).path:
                    continue
                if u not in seen:
                    seen.add(u)
                    out.append(u)
        return out

    def _parse_detail(self, soup: BeautifulSoup, ad_url: str) -> Dict[str, Any]:
        title_el = soup.select_one(".vmore-title h1")
        title = title_el.get_text(strip=True) if title_el else ""

        price_el = soup.select_one(".price-amount")
        price = price_el.get_text(strip=True) if price_el else ""

        contact = ""
        call = soup.select_one("a.ph-call[href^='tel:']")
        if call:
            num = call.select_one(".ph-num")
            contact = num.get_text(strip=True) if num else ""
            if not contact:
                contact = call.get("href", "").replace("tel:", "").strip()

        details: Dict[str, str] = {}
        for row in soup.select(".detail-card .detail-row"):
            lab = row.select_one(".detail-label")
            val = row.select_one(".detail-value")
            if not lab or not val:
                continue
            details[lab.get_text(strip=True)] = val.get_text(strip=True)

        more_text = ""
        for block in soup.select("div.more-card"):
            t = block.select_one(".more-card-title")
            if t and t.get_text(strip=True).lower() == "more details":
                b = block.select_one(".more-card-body")
                if b:
                    more_text = b.get_text("\n", strip=True)
                break

        # Keep only common fields across vehicle ads.
        return {
            "source_link": ad_url,
            "title": title,
            "price": price,
            "contact": contact,
            "location": details.get("Location", ""),
            "year": details.get("Year", ""),
            "make": details.get("Make", ""),
            "model": details.get("Model", ""),
            "fuel_type": details.get("Fuel Type", ""),
            "gear": details.get("Gear", ""),
            "condition": details.get("Condition", ""),
            "mileage": details.get("Mileage", ""),
            "more_details": more_text,
        }

    async def crawl_search_async(self, search_url: str) -> List[Dict[str, Any]]:
        search_url = _abs_url(search_url, "https://riyasewana.com")
        collected: List[str] = []
        results: List[Dict[str, Any]] = []

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(user_agent=DESKTOP_CHROME_UA)
            page.set_default_timeout(self.page_load_timeout_ms)

            html = await self._goto_with_backoff(page, search_url)
            soup = BeautifulSoup(html, "html.parser")
            last_page = self._max_page(soup)

            page_num = 1
            while page_num <= last_page and len(collected) < self.max_listings:
                url = _search_url_for_page(search_url, page_num)
                if page_num > 1:
                    html = await self._goto_with_backoff(page, url)

                batch = self._listing_urls(html)
                if not batch:
                    break
                for u in batch:
                    if u not in collected:
                        collected.append(u)
                    if len(collected) >= self.max_listings:
                        break

                if len(collected) >= self.max_listings:
                    break

                page_num += 1
                if page_num <= last_page:
                    await asyncio.sleep(self._sleep_s())

            for i, ad_url in enumerate(collected[: self.max_listings]):
                try:
                    logger.info(f"Listing {i + 1}/{len(collected[: self.max_listings])} {ad_url}")
                    detail_html = await self._goto_with_backoff(page, ad_url)
                    dsoup = BeautifulSoup(detail_html, "html.parser")
                    results.append(self._parse_detail(dsoup, ad_url))
                except Exception as e:
                    logger.error(f"Detail page error: {str(e)[:120]}")
                    results.append({"source_link": ad_url, "error": str(e)})
                await asyncio.sleep(self._sleep_s())

            await browser.close()

        return results

    def crawl_search(self, search_url: str) -> List[Dict[str, Any]]:
        return asyncio.run(self.crawl_search_async(search_url))


def save_listings_json(
    search_url: str,
    listings: List[Dict[str, Any]],
    output_path: str | Path | None = None,
) -> Path:
    """
    Write listings to `data/riyasewana_<slug>.json` unless `output_path` is set.
    Returns the path written.
    """
    if output_path is None:
        output_path = data_dir() / f"riyasewana_{_json_slug_from_search_url(search_url)}.json"
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(listings, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Saved {len(listings)} listings to {path}")
    return path


def crawl_search_to_json(
    search_url: str,
    max_listings: int = 500,
    request_delay: float = 1.5,
    output_path: str | Path | None = None,
) -> Path:
    """Crawl one search URL and save JSON under `data/`. Returns output file path."""
    crawler = RiyasewanaVehicleCrawler(
        max_listings=max_listings, request_delay=request_delay
    )
    rows = crawler.crawl_search(search_url)
    return save_listings_json(search_url, rows, output_path=output_path)

if __name__ == "__main__":
    crawl_search_to_json("https://riyasewana.com/search/suvs", max_listings=10, request_delay=1.5)
