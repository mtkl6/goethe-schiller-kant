#!/usr/bin/env python3
"""
Stufe 2: Chat-Feintuning auf chat_tokens.bin, gemischt mit ~20 % Prosa-Batches
(gegen Vergessen). Lädt das Pretrain-Checkpoint.

Verwendung: python3 finetune_chat.py [pretrain_final.pt]
Ausgabe:    chat_model.pt
"""

import sys
import time

import numpy as np
import torch
from tokenizers import Tokenizer

from model import model_from_checkpoint, get_device

batch_size = 16
max_iters = 3000
eval_interval = 250
eval_iters = 50
learning_rate = 5e-5
prose_fraction = 0.2
CKPT_IN = sys.argv[1] if len(sys.argv) > 1 else "pretrain_final.pt"
CKPT_OUT = "chat_model.pt"

torch.manual_seed(1337)
device = get_device()

ckpt = torch.load(CKPT_IN, map_location=device, weights_only=True)
block_size = ckpt['config']['block_size']
tokenizer = Tokenizer.from_file(ckpt['tokenizer_file'])
model = model_from_checkpoint(ckpt, device)
print(f"geladen: {CKPT_IN} (step {ckpt['step']}, val {ckpt['val_loss']:.3f})")

chat = torch.from_numpy(np.fromfile('chat_tokens.bin', dtype=np.uint16).astype(np.int64))
prose = torch.from_numpy(np.fromfile('pretrain_tokens.bin', dtype=np.uint16).astype(np.int64))
n = int(0.95 * len(chat))
chat_train, chat_val = chat[:n], chat[n:]
print(f"chat: {len(chat):,} Tokens, prosa: {len(prose):,} Tokens")


def get_batch(split):
    if split == 'train' and torch.rand(1).item() < prose_fraction:
        d = prose
    else:
        d = chat_train if split == 'train' else chat_val
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

last = None
t0 = time.time()
for it in range(max_iters):
    if it % eval_interval == 0 or it == max_iters - 1:
        last = estimate_loss()
        print(f"step {it}: train {last['train']:.4f}, chat-val {last['val']:.4f} "
              f"({time.time()-t0:,.0f}s)", flush=True)
    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save({
    'model_state_dict': model.state_dict(),
    'step': ckpt['step'] + max_iters,
    'train_loss': last['train'].item(),
    'val_loss': last['val'].item(),
    'tokenizer_file': ckpt['tokenizer_file'],
    'vocab_size': ckpt['vocab_size'],
    'stage': 'chat',
    'config': ckpt['config'],
}, CKPT_OUT)
print(f'model -> {CKPT_OUT}')

# kurzer In-Format-Test
model.eval()
prompt = "<|goethe|><|user|>Was ist Liebe?<|bot|>"
ids = tokenizer.encode(prompt).ids
idx = torch.tensor([ids], dtype=torch.long, device=device)
stop = {tokenizer.encode(t).ids[0] for t in ["<|endoftext|>", "<|user|>"]}
out = model.generate(idx, 150, temperature=0.8, top_k=50, stop_tokens=stop)
print('--- Goethe auf "Was ist Liebe?" ---')
print(tokenizer.decode(out[0][len(ids):].tolist()))
