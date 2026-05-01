# talksplit

> [中文版本（主要）](README.md)

Split a speech transcript into semantically coherent paragraphs using sentence
embeddings and TextTiling-style depth scoring. Works on Chinese, English, and
mixed text. No length limit.

Two ways to use it:
- **CLI** — `talksplit input.txt`
- **ComfyUI** — drop-in custom node set under `custom_nodes/`

## Install

```bash
python3 INSTALL.py
```

The installer auto-detects:
- **Active venv / conda env** (e.g. you ran it from inside ComfyUI's venv) →
  installs there, no `.venv` created.
- **Plain system Python** → creates a local `.venv` and installs there.

Default model is `BAAI/bge-m3` (~2 GB). Common variants:

```bash
python3 INSTALL.py --model intfloat/multilingual-e5-small   # ~470 MB
python3 INSTALL.py --skip-model                             # deps only, fetch on first run
python3 INSTALL.py --skip-venv                              # use current Python (e.g. ComfyUI portable)
python3 INSTALL.py --force-venv                             # create a local .venv even if one is active
```

## CLI usage

```bash
# basic — defaults to "speech" mode (sensitivity 1.0, min 2, max 15)
talksplit input.txt

# article / blog post mode (sensitivity 0.7, min 4, max 30)
talksplit input.txt --mode article

# tune split frequency (0–2; higher = more splits) — overrides mode default
talksplit input.txt --sensitivity 1.4

# force exactly 8 paragraphs
talksplit input.txt --target 8

# constrain paragraph length
talksplit input.txt --min-sentences 3 --max-sentences 12

# JSON output with sentence indices
talksplit input.txt --format json

# debug: visualize the similarity / depth curves
talksplit input.txt --plot curve.png

# different model
talksplit input.txt --model intfloat/multilingual-e5-small

# allow paragraph splits to fall at commas, not just sentence ends
talksplit input.txt --clause-level

# disable the discourse-marker boost (e.g. 'Let me switch', '接下來')
talksplit input.txt --no-markers
```

## Install (ComfyUI)

```bash
cd ComfyUI/custom_nodes
git clone <this-repo> talksplit
cd talksplit

# Option A — ComfyUI Manager will handle requirements.txt automatically.
# Option B — manual: from inside ComfyUI's venv, run our installer
#   (auto-detects the active venv; no separate .venv is created)
python INSTALL.py
# Option C — ComfyUI portable on Windows:
../../python_embeded/python.exe INSTALL.py --skip-venv
```

Restart ComfyUI. Nodes appear under the **`talksplit`** category.

> Folder name matters: clone as `talksplit/` or any name **other than `split/`**.
> The bridge handles common collisions but `split` would shadow the inner package.

## ComfyUI nodes

One-shot:
- **Talksplit · Auto** — text in, paragraphs out. Handles the whole pipeline.

Granular (for custom workflows):
- **Talksplit · Sentences** — split by punctuation, with no-punctuation fallback
- **Talksplit · Embed** — sentence embeddings via `sentence-transformers`
- **Talksplit · Score** — adjacent cosine similarity + TextTiling depth
- **Talksplit · Marker Boost** — bump depth at gaps before discourse markers
- **Talksplit · Optimize** — DP that picks split points under length constraints
- **Talksplit · Assemble** — join sentences into paragraphs at chosen splits
- **Talksplit · Plot** — render similarity/depth curve as `IMAGE`

Two example workflows ship in `workflows/`:
- `basic.json` — single Auto node
- `pipeline.json` — full granular pipeline with debug plot

## How it works

1. **Sentence segmentation**: punctuation-based primary path; long
   unpunctuated stretches get window-chopped.
2. **Embedding**: sentence-level vectors (default BGE-M3, multilingual).
3. **Boundary scoring**: cosine similarity between adjacent sentences, then a
   TextTiling-style depth score that captures how *deep* a similarity valley
   is relative to its local peaks — not just how low.
4. **Threshold**: `mean(depth) + k·std(depth)` where `k` is derived from the
   `sensitivity` knob (0–2). Sensitivity 1.0 keeps deep valleys only; 2.0
   includes anything above mean.
5. **Discourse-marker boost** (default on, `--no-markers` to disable): adds a
   depth bonus at gaps preceding sentences that begin with a transition phrase
   like *Let me switch gears* or *接下來*. The threshold is computed on the
   pre-boost distribution so a single boosted gap doesn't push other
   naturally-deep valleys below threshold.
6. **Length-constrained DP**: maximize total depth at chosen splits subject to
   every paragraph being within `[min_sentences, max_sentences]`. If the
   threshold filtered out every length-feasible split, the candidate set is
   relaxed automatically.
7. **Optional `target_paragraphs`**: ignore the threshold and pick exactly
   `target − 1` splits maximizing summed depth.
8. **Optional `--clause-level`**: also treat commas as terminators, so the
   DP can pick mid-sentence clause boundaries as paragraph breaks. Useful for
   long, comma-strung sentences common in spoken transcripts.

Punctuation is a *signal* (it controls where sentences end) but not the splitter
of paragraphs — paragraphs come from semantic cohesion.

## Tests

```bash
.venv/bin/pytest tests/ -v
```

## Project layout

```
__init__.py            ComfyUI entry (loads the renamed inner package)
INSTALL.py             one-shot installer for CLI mode
pyproject.toml         Python packaging for CLI mode
requirements.txt       ComfyUI auto-install
src/split/
  segmenter.py         sentence splitting
  embedder.py          embedding model wrapper
  boundary.py          similarity + TextTiling depth
  optimizer.py         length-constrained DP
  visualize.py         CLI plot
  comfy_nodes.py       ComfyUI node classes
  cli.py               CLI entry
workflows/
  basic.json           single Auto node
  pipeline.json        full pipeline
tests/                 unit tests
```
