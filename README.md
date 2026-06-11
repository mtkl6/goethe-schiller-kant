# Dichter & Denker

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Model-mtkl6%2Fdichter--denker-ffce1c.svg)](https://huggingface.co/mtkl6/dichter-denker)

A small GPT trained **from scratch** on the German classics, fine-tuned into a
chat model you can talk to as **Goethe, Schiller, Kant, Lessing, Kleist,
Hölderlin, Novalis or Herder**.

No pretrained weights, no API. Every part of the pipeline — the corpus, the
tokenizer, the model — is built here from raw public-domain text.

```
$ python chat.py

Mit wem möchtest du sprechen?
  1. Goethe   2. Schiller   3. Kant   4. Hoelderlin
  5. Kleist   6. Lessing    7. Novalis  8. Herder
> 3

Du:    Was ist die Pflicht des Menschen?
Kant:  Wenn wir uns nun selbst in der Welt verachtend verhalten müssen:
       was können wir tun?
```

> **What to expect.** This is a ~42M-parameter model trained on ~13M tokens on a
> laptop. It answers in fluent, period-flavoured German and picks up each
> author's register, but it is thematically loose and factually empty — think
> *style transfer*, not *question answering*. It shines on short exchanges
> phrased in classic style (`Sagt mir, Herr Geheimrat, …`). See
> [Limitations](#limitations).

## Example conversations

Unedited samples (temperature 0.6–0.8). The model stays in each author's
register — discursive Goethe, aphoristic Kant — without ever leaving the
18th/19th-century idiom it was trained on:

```text
[Kant]
Du:    Was ist die Pflicht des Menschen?
Kant:  Wenn wir uns nun selbst in der Welt verachtend verhalten müssen:
       was können wir tun?

[Goethe]
Du:    Woher kommt die wahre Kunst?
Goethe: Ich bin ein großer Mann, aber ich kenne den schönen Alten; sie sind
        von jeher mit erzogen worden. Aber wir haben nicht mehr gelesen, als
        wenn er nun auch nur auf seine Weise behandelte …
```

It free-associates rather than answers — that is the honest ceiling of a 42M
model trained on ~13M tokens (see [Limitations](#limitations)) — but the voice
is unmistakably period German.

---

## How it works

The whole thing is a five-stage pipeline, each stage a single script:

```
data.py              ── scrape Projekt Gutenberg-DE ─────►  input.txt
train_tokenizer.py   ── train 8k German BPE ──────────────►  tokenizer.json
prepare_chat_data.py ── mine dialogue from dramas + ──────►  chat_tokens.bin
                        Eckermann's "Gespräche mit Goethe"
pretrain.py          ── train GPT on the full corpus ─────►  pretrain_final.pt
finetune_chat.py     ── fine-tune on dialogue ───────────►  chat_model.pt
chat.py              ── talk to it
```

**1 · Corpus.** `data.py` scrapes German originals (not translations) of nine
authors from [projekt-gutenberg.org](https://projekt-gutenberg.org) — ~50 MB of
plays, prose, philosophy and Eckermann's conversations with Goethe.

**2 · Tokenizer.** `train_tokenizer.py` trains an 8,192-token byte-level BPE
tokenizer on the corpus itself. A German-native vocabulary keeps words whole
(~4.2 chars/token) instead of shredding them the way an English GPT-2 tokenizer
would, and shrinks the embedding table from 50k to 8k. Special tokens
`<|user|>`, `<|bot|>`, `<|endoftext|>` plus one persona token per author
(`<|goethe|>`, `<|kant|>`, …) carry the chat structure.

**3 · Chat data.** `prepare_chat_data.py` turns monologue text into dialogue:
- **Dramas** — 57 plays are parsed into speaker turns (four different Gutenberg
  layout conventions are handled); consecutive turns become `(user, bot)` pairs
  attributed to the play's author.
- **Eckermann** — his narration paired with Goethe's quoted replies (`»…«`).
- **Prose authors** (Kant, Novalis, Herder) — paragraph-completion pairs, so
  they answer in monologue style.

Conversations are serialized as
`<|goethe|><|user|>…<|bot|>…<|endoftext|>` and packed into training windows.

**4 · Pretraining.** `pretrain.py` trains a decoder-only transformer
(`model.py`) on the full corpus with a cosine LR schedule.

**5 · Fine-tuning.** `finetune_chat.py` continues training on the dialogue data,
mixed with 20% prose batches to prevent the model from forgetting its German.

## The model

A standard nanoGPT-style decoder-only transformer (`model.py`):

| | |
|---|---|
| Parameters | 42.3 M |
| Layers / heads / embd | 12 / 8 / 512 |
| Context length | 512 tokens |
| Vocabulary | 8,192 (custom German BPE) |
| Tied embeddings | yes |

**Training** (Apple Silicon / MPS, ~9k tok/s):

| Stage | Steps | LR | Final loss |
|---|---|---|---|
| Pretrain | 40,000 | 3e-4 → 3e-5 cosine | train 2.74 / val 4.21 |
| Chat fine-tune | 3,000 | 5e-5 | chat-val 3.32 |

A smaller 17.5M-param config (`n_layer 8, n_embd 384`) also works and trains in
~3 h; the 42M config roughly doubles wall-clock for noticeably more coherent
output.

## Quickstart

Requires Python ≥ 3.10 and PyTorch. On Apple Silicon, MPS is used automatically;
CUDA and CPU are supported too.

```bash
git clone https://github.com/mtkl6/goethe-schiller-kant.git
cd goethe-schiller-kant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Option A — just chat (download the trained model)

The trained weights are published on the Hugging Face Hub. Download
`chat_model.pt` and `tokenizer.json` into the project root:

```bash
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download as d; \
  [d('mtkl6/dichter-denker', f, local_dir='.') for f in ('chat_model.pt','tokenizer.json')]"
python chat.py
```

### Option B — reproduce the whole thing from scratch

```bash
python data.py                 # fetch corpus   -> input.txt   (~1 h; see Data & licensing first)
python train_tokenizer.py      # train BPE       -> tokenizer.json
python prepare_chat_data.py    # mine dialogue   -> chat_tokens.bin
python pretrain.py             # pretrain        -> pretrain_final.pt   (hours on MPS)
python finetune_chat.py        # fine-tune       -> chat_model.pt       (~40 min)
python chat.py                 # talk to it
```

Useful flags:
- `python pretrain.py --smoke` — 200 iterations, to check loss falls and measure throughput.
- `python sample_all.py` — dump free-form 3,000-token samples from every checkpoint.

In `chat.py`: `/persona` switch author · `/temp 0.7` set temperature · `/reset`
clear history · `/quit`.

> **Keep your machine awake** for long training runs — if the Mac sleeps, the
> background process pauses. Use `caffeinate -d` in a separate terminal.

## Project structure

```
data.py                scraper for Projekt Gutenberg-DE (--only AUTHOR appends one author)
train_tokenizer.py     trains the 8k German BPE tokenizer
prepare_chat_data.py   mines dialogue pairs from dramas / Eckermann / prose
model.py               the GPT definition (shared by all training & inference)
pretrain.py            stage 1 — pretrain on the full corpus
finetune_chat.py       stage 2 — fine-tune on dialogue
chat.py                interactive REPL
sample_all.py          free-form sampling from checkpoints
```

## Data & licensing

The **code** is MIT-licensed (see [LICENSE](LICENSE)).

The **texts** themselves are in the public domain — the authors died well over a
century ago, so their works are free of copyright in Germany, the EU and the US.
This repository does **not** redistribute any text: `data.py` fetches it on your
machine, and the corpus (`input.txt`) is gitignored.

> [!IMPORTANT]
> **`data.py` is provided for personal and research use. You are responsible for
> how you use it.** The default source, [Projekt Gutenberg-DE](https://projekt-gutenberg.org),
> has its own terms of use, and its `robots.txt` disallows AI-training crawlers
> (`GPTBot`, `ClaudeBot`, `CCBot`, `Google-Extended`, …). The public-domain
> *texts* are free to use, but please respect the *source site's* wishes: review
> its terms before scraping, keep the polite 1-second delay (or increase it), and
> don't run bulk/automated jobs against it at scale. If you need an unambiguously
> reuse-friendly source, point the scraper at one with explicit open terms —
> e.g. [Deutsches Textarchiv](https://www.deutschestextarchiv.de),
> [Wikisource](https://de.wikisource.org) (CC BY-SA), or
> [Project Gutenberg US](https://www.gutenberg.org).

## Limitations

- **It does not understand questions.** It continues text in the right register.
  Most fine-tuning data is drama turns, which respond to the *previous line of a
  scene*, not to a question — so it free-associates more than it answers.
- **Modern phrasing is out of distribution.** "Hallo, wie geht's?" never appears
  in 18th-century German; ask in period style for the best results.
- **It is factually empty.** At this scale the model stores grammar and style,
  not knowledge.

Cheap ways to push quality further: train longer (loss was still falling),
enlarge the model, lower the sampling temperature, or — the biggest lever for
*answering* — heavily upweight the real conversational data (Eckermann) relative
to drama turns in the fine-tune mix.

## Acknowledgements

- Model architecture follows Andrej Karpathy's
  [nanoGPT](https://github.com/karpathy/nanoGPT) / *Let's build GPT* lineage.
- Texts courtesy of [Projekt Gutenberg-DE](https://projekt-gutenberg.org).
- Tokenizer via Hugging Face [`tokenizers`](https://github.com/huggingface/tokenizers).
