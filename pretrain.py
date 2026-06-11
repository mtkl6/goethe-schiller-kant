#!/usr/bin/env python3
"""
Stufe 1: Pretraining auf dem vollen Prosa/Drama-Korpus (input.txt)
mit eigenem 8k-BPE-Tokenizer.

Verwendung: python3 pretrain.py [--smoke]
            --smoke = nur 200 Iterationen (Loss-Abfall prüfen, Durchsatz messen)
Ausgabe:    pretrain_step*.pt, pretrain_final.pt, pretrain_tokens.bin (Cache)
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from model import GPTLanguageModel, get_device

# hyperparameters
batch_size = 16
block_size = 512
max_iters = 40000
warmup_iters = 400
eval_interval = 500
checkpoint_interval = 5000
learning_rate = 3e-4
min_lr = 3e-5
eval_iters = 100
n_embd = 512
n_head = 8
n_layer = 12
dropout = 0.2
TOKENIZER_FILE = "tokenizer.json"
TOKEN_CACHE = "pretrain_tokens.bin"
# ------------

SMOKE = "--smoke" in sys.argv
if SMOKE:
    max_iters, eval_interval, checkpoint_interval, eval_iters = 200, 100, 10**9, 20

torch.manual_seed(1337)
device = get_device()

tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
vocab_size = tokenizer.get_vocab_size()

if Path(TOKEN_CACHE).exists():
    tokens = np.fromfile(TOKEN_CACHE, dtype=np.uint16)
    print(f"Token-Cache geladen: {len(tokens):,} Tokens")
else:
    print("Kodiere input.txt (einmalig, wird gecacht) ...")
    text = open("input.txt", encoding="utf-8").read()
    ids = []
    step = 1_000_000
    chunks = [text[i:i + step] for i in range(0, len(text), step)]
    for enc in tokenizer.encode_batch(chunks):
        ids.extend(enc.ids)
    tokens = np.array(ids, dtype=np.uint16)
    tokens.tofile(TOKEN_CACHE)
    print(f"{len(text):,} Zeichen -> {len(tokens):,} Tokens "
          f"({len(text)/len(tokens):.2f} Zeichen/Token)")

data = torch.from_numpy(tokens.astype(np.int64))
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    d = train_data if split == 'train' else val_data
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


def lr_at(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    progress = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    return min_lr + 0.5 * (learning_rate - min_lr) * (1 + math.cos(math.pi * progress))


model = GPTLanguageModel(vocab_size, n_embd, n_head, n_layer, block_size, dropout).to(device)
print(f'{sum(p.numel() for p in model.parameters())/1e6:.2f} M Parameter, device={device}')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


def save_checkpoint(path, step, train_loss, val_loss):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'tokenizer_file': TOKENIZER_FILE,
        'vocab_size': vocab_size,
        'stage': 'pretrain',
        'config': {
            'n_embd': n_embd, 'n_head': n_head, 'n_layer': n_layer,
            'block_size': block_size, 'dropout': dropout,
        },
    }, path)


last_losses = None
t0 = time.time()
for it in range(max_iters):
    if it % eval_interval == 0 or it == max_iters - 1:
        losses = estimate_loss()
        last_losses = losses
        tok_s = (it * batch_size * block_size) / max(time.time() - t0, 1e-9)
        print(f"step {it}: train {losses['train']:.4f}, val {losses['val']:.4f}, "
              f"lr {lr_at(it):.2e}, {tok_s:,.0f} tok/s", flush=True)

    if it > 0 and it % checkpoint_interval == 0:
        save_checkpoint(f'pretrain_step{it}.pt', it,
                        last_losses['train'].item(), last_losses['val'].item())
        print(f'checkpoint -> pretrain_step{it}.pt', flush=True)

    for g in optimizer.param_groups:
        g['lr'] = lr_at(it)
    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

if not SMOKE:
    save_checkpoint('pretrain_final.pt', max_iters,
                    last_losses['train'].item(), last_losses['val'].item())
    print('model -> pretrain_final.pt')

model.eval()
ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
out = model.generate(ctx, max_new_tokens=200, temperature=0.8, top_k=50)
print('--- sample ---')
print(tokenizer.decode(out[0].tolist()))
