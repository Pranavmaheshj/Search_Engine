import os
import aiohttp
import asyncio
import logging
from bs4 import BeautifulSoup
from ddgs import DDGS
from typing import List, Dict, Optional
from googleapiclient.discovery import build

logger = logging.getLogger(__name__)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

IGNORED_EXTENSIONS = [
    '.pdf', '.xlsx', '.docx', '.zip', '.rar', '.exe', '.mp3', '.mp4', '.jpg', '.png'
]

class WebFetcher:
    async def search_google_api(self, query: str, max_results: int = 3) -> List[Dict]:
        api_key = os.getenv("GOOGLE_API_KEY")
        search_engine_id = os.getenv("SEARCH_ENGINE_ID")
        if not api_key or not search_engine_id:
            logger.warning("Google API Key or Search Engine ID not found.")
            return []
        def _sync_search():
            try:
                service = build("customsearch", "v1", developerKey=api_key)
                res = service.cse().list(q=query, cx=search_engine_id, num=max_results).execute()
                return [{'href': item['link'], 'title': item['title']} for item in res.get('items', [])]
            except Exception as e:
                logger.error(f"Google API search failed: {e}"); return []
        results = await asyncio.to_thread(_sync_search)
        logger.info(f"Found {len(results)} links via Google API for query: '{query}'")
        return results

    async def search_ddg(self, query: str, max_results: int = 3) -> List[Dict]:
        try:
            def _sync_search():
                with DDGS(timeout=10) as ddgs:
                    return [{'href': r['href'], 'title': r['title']} for r in ddgs.text(query, max_results=max_results)]
            results = await asyncio.to_thread(_sync_search)
            logger.info(f"Found {len(results)} links via DDG for query: '{query}'")
            return results
        except Exception as e:
            logger.error(f"DDG search error: {e}"); return []

    async def _fetch_html(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10), headers=HEADERS, ssl=False) as response:
                if response.status == 200:
                    raw_content = await response.read()
                    return raw_content.decode('utf-8', errors='ignore')
                else:
                    return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}"); return None

    def _parse_content(self, html: str) -> str:
        if not html: return ""
        soup = BeautifulSoup(html, 'html.parser')
        for el in soup(["script", "style", "nav", "header", "footer", "aside"]): el.decompose()
        return ' '.join(soup.get_text(separator=' ', strip=True).split())[:4000]

    async def fetch_and_parse_best_result(self, query: str) -> Optional[Dict]:
        links = await self.search_google_api(query)
        if not links:
            logger.warning("Google API failed or returned no results. Falling back to DDG.")
            links = await self.search_ddg(query)
        if not links:
            logger.warning(f"No web links found for '{query}'."); return None
        async with aiohttp.ClientSession() as session:
            for link in links:
                url = link['href']
                if any(url.lower().endswith(ext) for ext in IGNORED_EXTENSIONS):
                    logger.warning(f"Skipping non-HTML link: {url}")
                    continue
                html = await self._fetch_html(session, url)
                if html:
                    content = self._parse_content(html)
                    if len(content) > 200:
                        logger.info(f"Successfully extracted content from {url}")
                        return {"text": content, "metadata": {"source": url, "title": link.get('title', query)}}
        return None