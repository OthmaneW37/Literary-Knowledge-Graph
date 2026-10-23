"""French (and other) public-domain texts from Wikisource, through the MediaWiki API.

Project Gutenberg mostly holds English editions; Wikisource holds many French
works and historical French translations (e.g. "Les Frères Karamazov",
translated by Halpérine-Kaminsky and Morice). A work is assembled from its
main page and its ordered sub-pages ("Livre I", "Livre II"...), converted to
Markdown with one heading per part, then indexed like any other book.
"""
from __future__ import annotations

import hashlib
import re
import time
from urllib.parse import quote

import httpx
from bs4 import BeautifulSoup

from .gutendex import CatalogError

# Wikimedia asks automated clients to identify themselves with a contact URL.
USER_AGENT = "NarrativeLens/2.0 (https://github.com/OthmaneW37/Literary-Knowledge-Graph; educational reading assistant)"
REMOVED = (".headertemplate, .footertemplate, .ws-noexport, .pagenum, .ws-pagenum, sup.reference, .references, "
           ".mw-references-wrap, .mw-editsection, .Z3988, .ws-data, style, script, table.navbox")
MAX_PARTS = 150
MIN_WORDS = 1500


def work_id_for(title: str, language: str = "fr") -> str:
    return f"wikisource_{language}_{hashlib.sha1(title.encode('utf-8')).hexdigest()[:12]}"


class WikisourceCatalog:
    def __init__(self, client: httpx.Client | None = None, pause_seconds: float = 0.2) -> None:
        self.client = client or httpx.Client(timeout=httpx.Timeout(30.0, connect=8.0), follow_redirects=True,
                                             headers={"User-Agent": USER_AGENT})
        self.pause_seconds = pause_seconds

    def _api(self, language: str, **params) -> dict:
        if not re.fullmatch(r"[a-z]{2,3}", language):
            raise ValueError("Langue Wikisource invalide.")
        for attempt in range(3):
            try:
                response = self.client.get(f"https://{language}.wikisource.org/w/api.php",
                                           params={"format": "json", "formatversion": 2, **params})
                response.raise_for_status()
                payload = response.json()
                break
            except httpx.TransportError as exc:  # Slow TLS handshakes happen; retry briefly.
                if attempt == 2:
                    raise CatalogError("Wikisource est momentanément inaccessible.") from exc
                time.sleep(1 + attempt)
            except (httpx.HTTPError, ValueError) as exc:
                raise CatalogError("Wikisource est momentanément inaccessible.") from exc
        if "error" in payload:
            raise CatalogError(f"Wikisource : {payload['error'].get('info', 'page introuvable')}.")
        return payload

    def search(self, query: str, language: str = "fr", limit: int = 8) -> list[dict]:
        payload = self._api(language, action="query", list="search", srsearch=query, srnamespace=0, srlimit=30,
                            srprop="snippet")
        books, seen = [], set()
        for row in payload.get("query", {}).get("search", []):
            root = row["title"].split("/", 1)[0]
            if root in seen:
                continue
            seen.add(root)
            books.append(self._describe(root, language))
            if len(books) >= limit:
                break
        return books

    @staticmethod
    def _describe(title: str, language: str) -> dict:
        translation = re.search(r"(?:traduction|trad\.)\s*([^()]+)", title, re.IGNORECASE)
        return {"provider": "wikisource", "provider_id": f"{language}:{title}", "title": title,
                "author": f"Traduction {translation.group(1).strip()}" if translation else "Wikisource",
                "languages": [language], "cover_url": "", "summary": "", "subjects": [], "people": [],
                "first_publish_year": None, "source_name": "Wikisource",
                "source_url": f"https://{language}.wikisource.org/wiki/{quote(title.replace(' ', '_'))}",
                "can_download": True, "format": "texte intégral",
                "work_id": work_id_for(title, language), "anna_url": ""}

    def _page(self, language: str, title: str) -> BeautifulSoup:
        payload = self._api(language, action="parse", page=title, prop="text", redirects=1)
        return BeautifulSoup(payload["parse"]["text"], "html.parser")

    @staticmethod
    def _text(soup: BeautifulSoup) -> str:
        for element in soup.select(REMOVED):
            element.decompose()
        lines = []
        for element in soup.find_all(["h2", "h3", "h4", "p", "dd", "li", "pre"]):
            if element.find_parent(["p", "li", "dd"]) is not None:
                continue
            text = re.sub(r"\s+", " ", element.get_text(" ", strip=True))
            text = re.sub(r"\s+([,.)\]])", r"\1", text).replace("( ", "(")
            if not text:
                continue
            # Section headings ("I", "II"...) become chapter headings for the chunker.
            lines.append(f"## {text}" if element.name in {"h2", "h3", "h4"} else text)
        return "\n\n".join(lines).strip()

    def fetch(self, provider_id: str) -> dict:
        """Return {title, author, language, text (Markdown), source_url} for a Wikisource work."""
        language, _, title = provider_id.partition(":")
        if not title:
            raise ValueError("Œuvre Wikisource invalide.")
        main = self._page(language, title)
        found = main.select_one(".headertemplate-author") or main.select_one(".ws-author")
        author = found.get_text(" ", strip=True) if found else ""
        parts = []
        for link in main.select("a[title]"):
            target = link["title"]
            if target.startswith(title + "/") and target not in parts and "action=edit" not in link.get("href", ""):
                parts.append(target)
        sections = [] if parts else [(title, self._text(main))]
        for part in parts[:MAX_PARTS]:
            time.sleep(self.pause_seconds)
            soup = self._page(language, part)
            if not author:
                found = soup.select_one(".headertemplate-author")
                author = found.get_text(" ", strip=True) if found else ""
            sections.append((part.split("/", 1)[1], self._text(soup)))
        sections = [(name, text) for name, text in sections if text]
        body = "\n\n".join(f"# Chapitre {number} — {name}\n\n{text}" for number, (name, text) in enumerate(sections, start=1))
        if len(body.split()) < MIN_WORDS:
            raise CatalogError("Cette page Wikisource ne contient pas le texte d’une œuvre (liste d’éditions ou "
                               "sommaire). Choisis une édition précise dans les résultats.")
        return {"title": title, "author": author or "Auteur inconnu", "language": language, "text": body,
                "source_url": f"https://{language}.wikisource.org/wiki/{quote(title.replace(' ', '_'))}"}
