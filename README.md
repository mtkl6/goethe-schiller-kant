# Dichter & Denker

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
python data.py                 # scrape corpus  -> input.txt   (~1 h, polite 1s delay)
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

The **texts** are sourced from [Projekt Gutenberg-DE](https://projekt-gutenberg.org)
and are in the public domain (the authors died well over a century ago). The
corpus is not redistributed in this repo — `data.py` fetches it directly from
the source, with a polite 1-second delay between requests. Please scrape
responsibly.

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
