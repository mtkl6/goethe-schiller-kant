#!/usr/bin/env python3
"""
Lädt alle Werke von Goethe, Schiller und Kant von Projekt Gutenberg-DE
(projekt-gutenberg.org) und kombiniert sie in einer einzigen TXT-Datei.

Im Gegensatz zur US-Variante (gutenberg.org) sind hier alle Texte deutsche
Originale — keine englischen Übersetzungen.

Verwendung: python3 data.py                  (alle Autoren, input.txt neu schreiben)
            python3 data.py --only ECKERMANN (nur ein Autor, an input.txt anhängen)
Ausgabe:    input.txt
"""

import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup

# ---- konfiguration ---------------------------------------------------------

BASE = "https://projekt-gutenberg.org"
OUTPUT_FILE = "input.txt"
USER_AGENT = "Mozilla/5.0 (compatible; dichter-denker/1.0)"
REQUEST_DELAY_SEC = 1.0  # höflich zur Quelle

# Autoren: (anzeigename, autorenseiten-slug, buch-pfad-slug)
# Slugs ggf. anpassen, falls eine Autorenseite nicht gefunden wird (0 Werke).
AUTHORS = [
    ("GOETHE",     "goethe",     "johann-wolfgang-von-goethe"),
    ("SCHILLER",   "schiller",   "friedrich-schiller"),
    ("KANT",       "kant",       "immanuel-kant"),
    ("HOELDERLIN", "hoelderl",   "friedrich-hoelderlin"),
    ("KLEIST",     "kleist",     "heinrich-von-kleist"),
    ("LESSING",    "lessing",    "gotthold-ephraim-lessing"),
    ("NOVALIS",    "novalis",    "novalis"),
    ("HERDER",     "herder",     "johann-gottfried-herder"),
    ("ECKERMANN",  "eckerman",   "johann-peter-eckermann"),
]

# ---- http helpers ----------------------------------------------------------

def fetch(url: str) -> str | None:
    """GET als UTF-8 String. None bei Fehler / 404."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        print(f"    HTTP {e.code} bei {url}")
        return None
    except Exception as e:
        print(f"    Fehler bei {url}: {e}")
        return None


# ---- discovery -------------------------------------------------------------

def list_book_urls(author_page_slug: str, book_path_slug: str) -> list[str]:
    """Findet alle Buch-URLs eines Autors über dessen Autorenseite."""
    url = f"{BASE}/autoren/namen/{author_page_slug}.html"
    html = fetch(url)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    needle = f"/authors/{book_path_slug}/books/"
    seen: set[str] = set()
    urls: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Nur Buch-Startseiten, keine /chapter/N Unterseiten
        if needle in href and "/chapter/" not in href:
            href = href.split("?")[0].split("#")[0].rstrip("/")
            if href not in seen:
                seen.add(href)
                urls.append(href)
    return urls


def list_chapter_urls(book_url: str) -> tuple[str, list[str]]:
    """Liefert (titel, [chapter_urls])."""
    html = fetch(book_url)
    if not html:
        return ("", [])
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string if soup.title else "").strip()
    title = re.sub(r"\s*[–—-]\s*Projekt Gutenberg.*$", "", title, flags=re.I)
    title = re.sub(r"\s*[–—-]\s*(Johann Wolfgang von Goethe|Friedrich Schiller|Immanuel Kant)\s*$", "", title, flags=re.I)

    seen: set[str] = set()
    chapters: list[tuple[int, str]] = []
    pat = re.compile(rf"^{re.escape(book_url)}/chapter/(\d+)/?$")
    for a in soup.find_all("a", href=True):
        href = a["href"].split("?")[0].split("#")[0]
        m = pat.match(href)
        if m and href not in seen:
            seen.add(href)
            chapters.append((int(m.group(1)), href))
    chapters.sort()
    return (title, [u for _, u in chapters])


# ---- text extraction -------------------------------------------------------

def extract_chapter_text(chapter_url: str) -> str | None:
    html = fetch(chapter_url)
    if not html:
        return None
    soup = BeautifulSoup(html, "html.parser")
    # erste Instanz = sichtbarer Inhalt; zweite = Lesemodus-Duplikat
    node = soup.find(class_="book-reader__chapter-text")
    if not node:
        return None
    text = node.get_text(separator="\n", strip=False)
    # whitespace normalisieren, aber Absätze erhalten
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_german(text: str) -> tuple[bool, int, int]:
    sample = text[:200_000].lower()
    de = sum(sample.count(m) for m in
             [" der ", " die ", " das ", " und ", " ist ", " nicht ",
              " sich ", " ein ", " mit ", " auch ", " wenn ", " nur ",
              " wird ", " werden ", " sind ", " ich ", " aber "])
    en = sum(sample.count(m) for m in
             [" the ", " and ", " of ", " to ", " is ", " that ",
              " it ", " was ", " for ", " with ", " which ", " he ",
              " as ", " be ", " have ", " but ", " are "])
    return de > en, de, en


# ---- main ------------------------------------------------------------------

def separator(author: str, title: str, source_url: str) -> str:
    line = "=" * 80
    return (f"\n\n{line}\n{line}\n"
            f"  AUTOR: {author}\n"
            f"  WERK:  {title}\n"
            f"  Quelle: {source_url}\n"
            f"{line}\n{line}\n\n")


def main() -> None:
    only_author = None
    if len(sys.argv) >= 3 and sys.argv[1] == "--only":
        only_author = sys.argv[2].upper()
        if only_author not in {a[0] for a in AUTHORS}:
            print(f"Unbekannter Autor: {only_author}")
            print(f"Verfügbar: {', '.join(a[0] for a in AUTHORS)}")
            sys.exit(1)

    authors = [a for a in AUTHORS if only_author is None or a[0] == only_author]
    mode = "a" if only_author else "w"
    print(f"Lade Werke von Projekt Gutenberg-DE → {OUTPUT_FILE}"
          + (f" (nur {only_author}, anhängen)" if only_author else "") + "\n")
    total_chars = 0
    works_ok = 0
    works_skipped_lang = 0
    works_empty = 0
    skipped: list[str] = []

    out_path = Path(OUTPUT_FILE)
    with out_path.open(mode, encoding="utf-8") as out:
        if not only_author:
            out.write("=" * 80 + "\n")
            out.write("  GESAMMELTE DEUTSCHE WERKE: GOETHE · SCHILLER · KANT\n")
            out.write("  Quelle: Projekt Gutenberg-DE (projekt-gutenberg.org)\n")
            out.write("  Alle Werke sind gemeinfrei (Public Domain).\n")
            out.write("=" * 80 + "\n")

        for author_name, author_slug, book_slug in authors:
            print(f"\n=== {author_name} ===")
            book_urls = list_book_urls(author_slug, book_slug)
            print(f"  {len(book_urls)} Werke auf Autorenseite gefunden")
            out.write(f"\n\n{'#' * 80}\n##  {author_name}\n{'#' * 80}\n")
            time.sleep(REQUEST_DELAY_SEC)

            for i, book_url in enumerate(book_urls, 1):
                title, chapter_urls = list_chapter_urls(book_url)
                short_title = (title or book_url.rsplit("/", 1)[-1])[:70]
                print(f"  [{i:>3}/{len(book_urls)}] {short_title}  ({len(chapter_urls)} Kapitel)", flush=True)
                time.sleep(REQUEST_DELAY_SEC)

                if not chapter_urls:
                    works_empty += 1
                    skipped.append(f"{author_name}: {short_title} (keine Kapitel)")
                    continue

                parts: list[str] = []
                for ch_url in chapter_urls:
                    txt = extract_chapter_text(ch_url)
                    if txt:
                        parts.append(txt)
                    time.sleep(REQUEST_DELAY_SEC)

                if not parts:
                    works_empty += 1
                    skipped.append(f"{author_name}: {short_title} (Text leer)")
                    continue

                full = "\n\n".join(parts)
                ok, de, en = is_german(full)
                if not ok:
                    works_skipped_lang += 1
                    skipped.append(f"{author_name}: {short_title} (nicht deutsch | de={de}, en={en})")
                    print(f"    ⊘  nicht deutsch (de={de}, en={en})")
                    continue

                out.write(separator(author_name, title or short_title, book_url))
                out.write(full)
                out.write("\n")
                total_chars += len(full)
                works_ok += 1
                print(f"    ✓  {len(full):,} Zeichen")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\n{'=' * 50}")
    print(f"Fertig!")
    print(f"  Werke gespeichert:      {works_ok}")
    print(f"  Übersprungen (Sprache): {works_skipped_lang}")
    print(f"  Übersprungen (leer):    {works_empty}")
    if skipped:
        print(f"\n  Details:")
        for s in skipped:
            print(f"    - {s}")
    print(f"\n  Gesamtzeichen:  {total_chars:,}")
    print(f"  Dateigröße:     {size_mb:.1f} MB")
    print(f"  Datei:          {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
