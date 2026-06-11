#!/usr/bin/env python3
"""
Trainiert einen deutschen BPE-Tokenizer (8192 Vokabeln) auf input.txt.

Im Gegensatz zum gpt2-Tokenizer (50k, englisch-optimiert) sitzt das ganze
Vokabular auf unserem Korpus — deutsche Wörter bleiben ganz, die Embedding-
Tabelle schrumpft von 50k auf 8k Einträge.

Verwendung: python3 train_tokenizer.py
Ausgabe:    tokenizer.json
"""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre
from tokenizers.decoders import ByteLevel as ByteLevelDec
from tokenizers.trainers import BpeTrainer

VOCAB_SIZE = 8192
INPUT_FILE = "input.txt"
OUTPUT_FILE = "tokenizer.json"

PERSONAS = ["goethe", "schiller", "kant", "hoelderlin", "kleist",
            "lessing", "novalis", "herder", "eckermann"]
SPECIAL_TOKENS = ["<|endoftext|>", "<|user|>", "<|bot|>"] + \
    [f"<|{p}|>" for p in PERSONAS]


def main() -> None:
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevelPre(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDec()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    print(f"Trainiere BPE (vocab {VOCAB_SIZE}) auf {INPUT_FILE} ...")
    tokenizer.train([INPUT_FILE], trainer)
    tokenizer.save(OUTPUT_FILE)
    print(f"Gespeichert: {OUTPUT_FILE}")

    # ---- smoke tests --------------------------------------------------------
    probe = "Müßiggang ist aller Laster Anfang — sprach's und aß süße Äpfel.»Wohl!«"
    ids = tokenizer.encode(probe).ids
    rt = tokenizer.decode(ids)
    assert rt == probe, f"Roundtrip kaputt:\n{probe!r}\n{rt!r}"
    print(f"Roundtrip ok ({len(probe)} Zeichen -> {len(ids)} Tokens)")

    for tok in SPECIAL_TOKENS:
        enc = tokenizer.encode(tok).ids
        assert len(enc) == 1, f"{tok} ist nicht 1 Token: {enc}"
        print(f"  {tok:<16} -> id {enc[0]}")

    with open(INPUT_FILE, encoding="utf-8") as f:
        sample = f.read(2_000_000)
    n_tok = len(tokenizer.encode(sample).ids)
    print(f"Kompression: {len(sample)/n_tok:.2f} Zeichen/Token "
          f"(2 MB Stichprobe -> {n_tok:,} Tokens)")


if __name__ == "__main__":
    main()
