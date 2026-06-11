#!/usr/bin/env python3
"""
Baut Chat-Trainingsdaten aus input.txt:

  1. Dramen      -> Sprecherwechsel werden (user, bot)-Dialoge, Persona = Autor
  2. Eckermann   -> »Goethe-Zitate« werden bot-Antworten, Persona = Goethe
  3. Prosa       -> Absatz-Fortsetzungspaare für dialogarme Autoren (Monologstil)

Serialisierung pro Konversation:
  <|goethe|><|user|>TEXT<|bot|>TEXT<|user|>...<|endoftext|>

Verwendung: python3 prepare_chat_data.py
Ausgabe:    chat_tokens.bin (uint16), chat_samples.txt (Stichproben)
"""

import random
import re
from collections import Counter, defaultdict

import numpy as np
from tokenizers import Tokenizer

INPUT_FILE = "input.txt"
TOKENIZER_FILE = "tokenizer.json"
OUT_BIN = "chat_tokens.bin"
OUT_SAMPLES = "chat_samples.txt"

MAX_TURN_CHARS = 600
MIN_TURN_CHARS = 5
WINDOW_CHARS = 1700          # ~ block_size 512 bei ~3.5 Zeichen/Token
MIN_TURNS_FOR_DRAMA = 50
MIN_TURN_DENSITY = 0.05      # Sprecherzeilen / Nicht-Leerzeilen — filtert Historien/TOC
MIN_NAME_COUNT = 5
PROSE_AUTHORS = {"KANT", "NOVALIS", "HERDER", "HOELDERLIN"}
MAX_PROSE_PAIRS_PER_AUTHOR = 3000

PERSONA = {a: f"<|{a.lower()}|>" for a in
           ["GOETHE", "SCHILLER", "KANT", "HOELDERLIN", "KLEIST",
            "LESSING", "NOVALIS", "HERDER", "ECKERMANN"]}

HEADER_RE = re.compile(
    r"={80}\n={80}\n  AUTOR: (\w+)\n  WERK: +(.*)\n  Quelle: .*\n={80}\n={80}\n")
SCENE_RE = re.compile(
    r"^\s*(Erste|Zweite|Dritte|Vierte|Fünfte|Sechste|Siebente|Siebte|Achte|"
    r"Neunte|Zehnte|Elfte|Zwölfte|Letzte)\w* (Aufzug|Auftritt|Szene|Scene|Akt|Auftritt)|"
    r"^\s*(Aufzug|Auftritt|Szene|Scene|Akt|Prolog|Vorspiel|Personen)\b", re.I)
# Sprecherzeile an Spalte 0: "Iphigenie:", "Miller." (mit Satzzeichen) oder
# "Daja" / "Miller" (nackt — nur gültig, wenn der Folgetext mit . oder : beginnt)
NAME_CORE = (r"[A-ZÄÖÜ][\wäöüß'’\-]+(?: (?:von|der|die|des) [A-ZÄÖÜ][\wäöüß'’\-]+"
             r"| [A-ZÄÖÜ][\wäöüß'’\-]+){0,2}")
NAME_PUNCT_RE = re.compile(rf"^({NAME_CORE})\s*[.:]\s*$")
NAME_BARE_RE = re.compile(rf"^({NAME_CORE})\s*$")
# Personenverzeichnis-Zeilen wie "Iphigenie. Arkas." (>= 2 Namen) -> Szenenbruch
CAST_RE = re.compile(r"^(?:[A-ZÄÖÜ][\wäöüß'’\-]+[.,]\s*){2,}$")
PAREN_RE = re.compile(r"\([^)]{0,400}\)", re.S)


def split_works(text):
    """[(autor, titel, werk_text), ...]"""
    parts = HEADER_RE.split(text)
    works = []
    for i in range(1, len(parts) - 2, 3):
        works.append((parts[i], parts[i + 1].strip(), parts[i + 2]))
    return works


def clean_turn(lines):
    # Regieanweisungen an Spalte 0 ("lacht.", "bringt einen Spiegel.") raus
    lines = [ln for ln in lines
             if not (ln and ln[0].islower() and len(ln) <= 60 and not ln.startswith(" "))]
    body = "\n".join(lines)
    body = PAREN_RE.sub(" ", body)
    body = re.sub(r"^[.:]\s*", "", body.strip())
    body = re.sub(r"\s+", " ", body).strip()
    return body


def speaker_name(lines, i):
    """Name, falls Zeile i eine Sprecherzeile ist, sonst None."""
    ln = lines[i]
    if len(ln) > 32:
        return None
    m = NAME_PUNCT_RE.match(ln)
    if m:
        return m.group(1)
    m = NAME_BARE_RE.match(ln)
    if not m:
        return None
    # nackter Name: Folgetext muss mit . oder : beginnen (Nathan/Kabale) oder
    # eingerückt sein (Kleist/Iphigenie). Regieanweisungen ("lacht.", "(...)")
    # werden übersprungen. Historien-Zeilenumbrüche (Folge an Spalte 0) fallen raus.
    for j in range(i + 1, min(i + 5, len(lines))):
        nxt = lines[j]
        st = nxt.strip()
        if not st or st.startswith("(") or (st[0].islower() and len(st) <= 60):
            continue
        # eingerückte Folge muss großgeschrieben anfangen — kleingeschrieben =
        # mitten-im-Satz-Umbruch (Hervorhebung in Historien), kein Sprecher
        ok = st[0] in ".:" or (nxt[0] in " \t" and not st[0].islower())
        return m.group(1) if ok else None
    return None


def parse_drama(text):
    """-> Liste von Konversationen, jede = Liste von Turn-Strings."""
    lines = text.split("\n")
    # Pass 1: Kandidaten zählen
    counts = Counter()
    for i in range(len(lines)):
        name = speaker_name(lines, i)
        if name:
            counts[name] += 1
    speakers = {n for n, c in counts.items() if c >= MIN_NAME_COUNT}
    if not speakers:
        return []

    # Pass 2: Turns einsammeln
    convos, current, buf, in_turn = [], [], [], False

    def flush_turn():
        nonlocal buf, in_turn
        if in_turn:
            t = clean_turn(buf)
            if MIN_TURN_CHARS <= len(t) <= MAX_TURN_CHARS:
                current.append(t)
            else:
                flush_convo()
        buf, in_turn = [], False

    def flush_convo():
        nonlocal current
        if len(current) >= 2:
            convos.append(current)
        current = []

    for i, ln in enumerate(lines):
        if SCENE_RE.match(ln) or CAST_RE.match(ln.strip()):
            flush_turn()
            flush_convo()
            continue
        if speaker_name(lines, i) in speakers:
            flush_turn()
            in_turn = True
            continue
        if in_turn:
            buf.append(ln)
    flush_turn()
    flush_convo()
    return convos


ATTRIB_RE = re.compile(
    r"Goethe|sagte (er|Goethe)|erwiderte er|versetzte er|fuhr er fort|fügte er hinzu")


def parse_eckermann(text):
    """Goethes »Zitate« -> bot, vorhergehende Erzählung -> user.
    Goethes Rede ist oft in mehrere »...«-Stücke zerteilt (»...,« sagte er; »...«)
    — alle Stücke eines Absatzes werden zusammengefügt."""
    pairs = []
    # Zeilen sind Absätze (Scraper liefert eine Zeile pro Absatz)
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    for prev, para in zip(paragraphs, paragraphs[1:]):
        quotes = re.findall(r"»([^«]{10,800})«", para)
        if not quotes:
            continue
        narration = re.sub(r"»[^«]*«", " ", para)
        if not ATTRIB_RE.search(narration):
            continue
        bot = re.sub(r"\s+", " ", " ".join(quotes)).strip()[:MAX_TURN_CHARS]
        # user = letzte 1–2 Sätze der vorherigen Erzählung (Eckermanns Frage/Bericht)
        prev_clean = re.sub(r"»[^«]*«", " ", prev)
        prev_clean = re.sub(r"\s+", " ", prev_clean).strip()
        prev_clean = re.sub(r"^[,;.\s]+", "", prev_clean)
        sents = re.split(r"(?<=[.!?]) ", prev_clean)
        user = " ".join(sents[-2:]).strip()
        user = re.sub(r"[,;\s]*(sagte|erwiderte|versetzte|fragte) (er|Goethe|ich)[,;\s]*$",
                      "", user).strip()
        if (20 <= len(user) <= MAX_TURN_CHARS and len(bot) >= 30
                and user[0].isupper()):
            pairs.append((user, bot))
    return pairs


def parse_prose_pairs(text):
    """Erster Satz eines Absatzes -> user, Rest -> bot."""
    pairs = []
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s+", " ", para).strip()
        if not 200 <= len(para) <= 5000:
            continue
        # erster Satz endet vor Großbuchstabe (verhindert Schnitt bei "d. i.")
        m = re.match(r"(.{50,300}?[.!?])\s+(?=[A-ZÄÖÜ»])(.{100,})", para)
        if m:
            pairs.append((m.group(1), m.group(2)[:MAX_TURN_CHARS]))
    return pairs


def serialize(persona_tok, turns):
    """Turns abwechselnd user/bot; endet immer mit bot."""
    if len(turns) % 2 == 1:
        turns = turns[:-1]
    if not turns:
        return None
    s = persona_tok
    for i, t in enumerate(turns):
        s += ("<|user|>" if i % 2 == 0 else "<|bot|>") + t
    return s + "<|endoftext|>"


def pack_windows(turns):
    """Lange Konversationen in Fenster von ~WINDOW_CHARS aufteilen."""
    windows, cur, n = [], [], 0
    for t in turns:
        if n + len(t) > WINDOW_CHARS and len(cur) >= 2:
            windows.append(cur)
            cur, n = [], 0
        cur.append(t)
        n += len(t)
    if len(cur) >= 2:
        windows.append(cur)
    return windows


def main():
    random.seed(1337)
    text = open(INPUT_FILE, encoding="utf-8").read()
    works = split_works(text)
    print(f"{len(works)} Werke gefunden")

    convo_texts = []
    stats = defaultdict(lambda: [0, 0])  # persona -> [konvos, turns]
    samples = defaultdict(list)
    prose_buckets = defaultdict(list)

    for author, title, body in works:
        persona = PERSONA.get(author)
        if persona is None:
            continue

        if author == "ECKERMANN":
            pairs = parse_eckermann(body)
            for user, bot in pairs:
                s = serialize(PERSONA["GOETHE"], [user, bot])
                if s:
                    convo_texts.append(s)
                    stats["GOETHE(eckermann)"][0] += 1
                    stats["GOETHE(eckermann)"][1] += 2
                    samples["GOETHE(eckermann)"].append(s)
            if pairs:
                print(f"  [eckermann] {title[:50]}: {len(pairs)} Paare")
            continue

        convos = parse_drama(body)
        n_turns = sum(len(c) for c in convos)
        nonblank = sum(1 for ln in body.split("\n") if ln.strip())
        density = n_turns / max(nonblank, 1)
        if n_turns >= MIN_TURNS_FOR_DRAMA and density >= MIN_TURN_DENSITY:
            for convo in convos:
                for window in pack_windows(convo):
                    s = serialize(persona, window)
                    if s:
                        convo_texts.append(s)
                        stats[author][0] += 1
                        stats[author][1] += len(window)
                        samples[author].append(s)
            print(f"  [drama]     {author:<10} {title[:45]:<45} {n_turns} Turns")
        elif author in PROSE_AUTHORS:
            prose_buckets[author].extend(parse_prose_pairs(body))

    for author, pairs in prose_buckets.items():
        random.shuffle(pairs)
        for user, bot in pairs[:MAX_PROSE_PAIRS_PER_AUTHOR]:
            s = serialize(PERSONA[author], [user, bot])
            if s:
                convo_texts.append(s)
                stats[author][0] += 1
                stats[author][1] += 2
                samples[author].append(s)

    print(f"\n{'Persona':<22}{'Konvos':>8}{'Turns':>8}")
    for p, (c, t) in sorted(stats.items()):
        print(f"{p:<22}{c:>8}{t:>8}")

    random.shuffle(convo_texts)
    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
    print("\nTokenisiere ...")
    all_ids = []
    for enc in tokenizer.encode_batch(convo_texts):
        all_ids.extend(enc.ids)
    arr = np.array(all_ids, dtype=np.uint16)
    arr.tofile(OUT_BIN)
    print(f"{len(arr):,} Tokens -> {OUT_BIN}")

    with open(OUT_SAMPLES, "w", encoding="utf-8") as f:
        for p in sorted(samples):
            f.write(f"\n{'=' * 70}\n== {p}\n{'=' * 70}\n")
            for s in random.sample(samples[p], min(5, len(samples[p]))):
                f.write(s.replace("<|user|>", "\n\nUSER: ")
                         .replace("<|bot|>", "\nBOT:  ")
                         .replace("<|endoftext|>", "\n----") + "\n")
    print(f"Stichproben -> {OUT_SAMPLES}")


if __name__ == "__main__":
    main()
