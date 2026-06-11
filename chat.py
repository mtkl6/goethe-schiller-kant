#!/usr/bin/env python3
"""
Chat mit einem Dichter oder Denker.

Verwendung: python3 chat.py [chat_model.pt]
Befehle:    /persona  Persona wechseln
            /reset    Gesprächsverlauf löschen
            /temp X   Temperatur setzen (z.B. /temp 0.7)
            /quit     Beenden
"""

import sys

import torch
from tokenizers import Tokenizer

from model import model_from_checkpoint, get_device

CKPT = sys.argv[1] if len(sys.argv) > 1 else "chat_model.pt"
PERSONAS = ["goethe", "schiller", "kant", "hoelderlin", "kleist",
            "lessing", "novalis", "herder"]
MAX_NEW_TOKENS = 200

device = get_device()
ckpt = torch.load(CKPT, map_location=device, weights_only=True)
tokenizer = Tokenizer.from_file(ckpt['tokenizer_file'])
model = model_from_checkpoint(ckpt, device)
model.eval()
block_size = ckpt['config']['block_size']
stop_ids = {tokenizer.encode(t).ids[0] for t in ["<|endoftext|>", "<|user|>", "<|bot|>"]}


def pick_persona() -> str:
    print("\nMit wem möchtest du sprechen?")
    for i, p in enumerate(PERSONAS, 1):
        print(f"  {i}. {p.capitalize()}")
    while True:
        choice = input("> ").strip().lower()
        if choice.isdigit() and 1 <= int(choice) <= len(PERSONAS):
            return PERSONAS[int(choice) - 1]
        if choice in PERSONAS:
            return choice
        print("Bitte Zahl oder Name.")


def reply(persona: str, history: list[tuple[str, str]], user_msg: str) -> str:
    """history = [(user, bot), ...]; baut Prompt, links auf block_size gekürzt."""
    turns = []
    for u, b in history:
        turns.append(f"<|user|>{u}<|bot|>{b}")
    turns.append(f"<|user|>{user_msg}<|bot|>")
    persona_ids = tokenizer.encode(f"<|{persona}|>").ids
    ids = tokenizer.encode("".join(turns)).ids
    budget = block_size - MAX_NEW_TOKENS - len(persona_ids)
    ids = persona_ids + ids[-budget:]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, MAX_NEW_TOKENS, temperature=temp, top_k=50,
                         stop_tokens=stop_ids, repetition_penalty=1.15)
    return tokenizer.decode(out[0][len(ids):].tolist()).strip()


print(f"Modell: {CKPT} (Stufe {ckpt.get('stage', '?')}, "
      f"val-loss {ckpt['val_loss']:.3f}, device {device})")
persona = pick_persona()
history: list[tuple[str, str]] = []
temp = 0.8
print(f"\nDu sprichst mit {persona.capitalize()}. (/persona /reset /temp /quit)\n")

while True:
    try:
        msg = input("Du: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not msg:
        continue
    if msg == "/quit":
        break
    if msg == "/reset":
        history = []
        print("(Verlauf gelöscht)")
        continue
    if msg == "/persona":
        persona = pick_persona()
        history = []
        print(f"(Du sprichst jetzt mit {persona.capitalize()})")
        continue
    if msg.startswith("/temp"):
        try:
            temp = float(msg.split()[1])
            print(f"(Temperatur = {temp})")
        except (IndexError, ValueError):
            print("Verwendung: /temp 0.7")
        continue

    answer = reply(persona, history, msg)
    history.append((msg, answer))
    print(f"{persona.capitalize()}: {answer}\n")

print("Lebe wohl!")
