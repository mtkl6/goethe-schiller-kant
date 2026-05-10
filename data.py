#!/usr/bin/env python3
"""
Lädt alle verfügbaren deutschen Werke von Goethe, Schiller und Kant
von Project Gutenberg und kombiniert sie in einer einzigen TXT-Datei.

Verwendung: python3 download_klassiker_de.py
"""

import urllib.request
import time
import os

WERKE = {
    "GOETHE": [
        (2229,  "Faust - Erster Teil"),
        (2230,  "Faust - Zweiter Teil"),
        (2054,  "Die Leiden des jungen Werthers"),
        (2403,  "Wilhelm Meisters Lehrjahre"),
        (2404,  "Wilhelm Meisters Wanderjahre"),
        (14444, "Die Wahlverwandtschaften"),
        (5155,  "Götz von Berlichingen"),
        (30719, "Iphigenie auf Tauris"),
        (19956, "Egmont"),
        (3232,  "Torquato Tasso"),
        (16796, "Reineke Fuchs"),
        (2407,  "Hermann und Dorothea"),
        (1396,  "Die Leiden des jungen Werthers - Band 2"),
        (22367, "Gedichte (Auswahl)"),
        (2406,  "Dichtung und Wahrheit"),
    ],
    "SCHILLER": [
        (6788,  "Die Räuber"),
        (6791,  "Kabale und Liebe"),
        (6787,  "Don Karlos"),
        (6789,  "Maria Stuart"),
        (6792,  "Die Jungfrau von Orleans"),
        (6793,  "Wilhelm Tell"),
        (6794,  "Wallensteins Lager"),
        (6795,  "Die Piccolomini"),
        (6796,  "Wallensteins Tod"),
        (6797,  "Die Braut von Messina"),
        (47983, "Über die ästhetische Erziehung des Menschen"),
        (6800,  "Über naive und sentimentalische Dichtung"),
        (7984,  "Geschichte des Dreißigjährigen Krieges"),
        (6801,  "Gedichte (Auswahl)"),
    ],
    "KANT": [
        (6342,  "Kritik der reinen Vernunft"),
        (6343,  "Kritik der praktischen Vernunft"),
        (6344,  "Kritik der Urteilskraft"),
        (6345,  "Grundlegung zur Metaphysik der Sitten"),
        (46679, "Zum ewigen Frieden"),
        (36048, "Prolegomena zu einer jeden künftigen Metaphysik"),
        (19404, "Die Metaphysik der Sitten"),
        (15269, "Anthropologie in pragmatischer Hinsicht"),
    ],
}

OUTPUT_FILE = "alle_klassiker_deutsch.txt"
URL_TEMPLATES = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
]


def download_text(gutenberg_id: int) -> str | None:
    for url_template in URL_TEMPLATES:
        url = url_template.format(id=gutenberg_id)
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; klassiker-downloader/1.0)"}
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                raw = response.read()
                for enc in ("utf-8", "latin-1", "windows-1252"):
                    try:
                        return raw.decode(enc)
                    except UnicodeDecodeError:
                        continue
        except Exception:
            continue
    return None


def strip_gutenberg_header_footer(text: str) -> str:
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "***START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
        "*** ANFANG DIESES PROJECT GUTENBERG",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "***END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
        "*** ENDE DIESES PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ]

    start_pos = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            newline = text.find("\n", idx)
            if newline != -1:
                start_pos = newline + 1
            break

    end_pos = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_pos = idx
            break

    return text[start_pos:end_pos].strip()


def separator(author: str, title: str, gutenberg_id: int) -> str:
    line = "=" * 80
    return f"""

{line}
{line}
  AUTOR: {author}
  WERK:  {title}
  Quelle: Project Gutenberg (ID: {gutenberg_id}) | gutenberg.org
{line}
{line}

"""


def main():
    print(f"Starte Download → wird gespeichert als: {OUTPUT_FILE}\n")
    total_chars = 0
    success_count = 0
    fail_count = 0
    failed_titles = []

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write("=" * 80 + "\n")
        out.write("  GESAMMELTE DEUTSCHE WERKE: GOETHE · SCHILLER · KANT\n")
        out.write("  Quelle: Project Gutenberg (gutenberg.org)\n")
        out.write("  Alle Werke sind gemeinfrei (Public Domain)\n")
        out.write("=" * 80 + "\n\n")

        for author, works in WERKE.items():
            out.write(f"\n\n{'#' * 80}\n")
            out.write(f"##  {author}\n")
            out.write(f"{'#' * 80}\n\n")

            for gid, title in works:
                print(f"  [{author}] {title} (ID {gid}) ...", end=" ", flush=True)
                text = download_text(gid)

                if text:
                    cleaned = strip_gutenberg_header_footer(text)
                    out.write(separator(author, title, gid))
                    out.write(cleaned)
                    out.write("\n")
                    chars = len(cleaned)
                    total_chars += chars
                    success_count += 1
                    print(f"✓  ({chars:,} Zeichen)")
                else:
                    print(f"✗  (nicht verfügbar)")
                    fail_count += 1
                    failed_titles.append(f"{author}: {title} (ID {gid})")

                time.sleep(0.5)

    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"\n{'=' * 50}")
    print(f"Fertig!")
    print(f"  Werke geladen:     {success_count}")
    print(f"  Fehlgeschlagen:    {fail_count}")
    if failed_titles:
        for t in failed_titles:
            print(f"    - {t}")
    print(f"  Gesamtzeichen:     {total_chars:,}")
    print(f"  Dateigröße:        {size_mb:.1f} MB")
    print(f"  Datei:             {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
