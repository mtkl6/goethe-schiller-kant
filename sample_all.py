"""Lädt jeden v3-Checkpoint (pretrain_*/chat_model.pt) und generiert 3000 Tokens."""

import glob

import torch
from tokenizers import Tokenizer

from model import model_from_checkpoint, get_device

device = get_device()
MAX_NEW_TOKENS = 3000


def sample_from_checkpoint(path: str) -> None:
    print(f"\n{'=' * 80}\nCheckpoint: {path}\n{'=' * 80}")
    ckpt = torch.load(path, map_location=device, weights_only=True)
    tokenizer = Tokenizer.from_file(ckpt['tokenizer_file'])
    model = model_from_checkpoint(ckpt, device)
    model.eval()

    step = ckpt.get('step', '?')
    tl = ckpt.get('train_loss')
    vl = ckpt.get('val_loss')
    print(f"step={step}  train_loss={tl}  val_loss={vl}\n")

    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    out_tokens = model.generate(ctx, max_new_tokens=MAX_NEW_TOKENS,
                                temperature=0.8, top_k=50)[0].tolist()
    text = tokenizer.decode(out_tokens)

    out_file = path.replace('.pt', '_sample.txt')
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(f"# Sample von {path} (step={step}, train={tl}, val={vl})\n\n")
        f.write(text)
    print(text[:2000])
    print(f"\n-> gespeichert: {out_file}")


def main() -> None:
    paths = sorted(glob.glob('pretrain_step*.pt')) + \
        sorted(glob.glob('pretrain_final.pt')) + sorted(glob.glob('chat_model.pt'))
    if not paths:
        print("Keine Checkpoints gefunden.")
        return
    for p in paths:
        sample_from_checkpoint(p)


if __name__ == '__main__':
    main()
