# Cicero LLM

A 100M-parameter Latin language model, trained from scratch, running entirely
in your browser. No server, no API — the int8-quantized model downloads once
and runs client-side via ONNX Runtime Web (WebGPU, with a WASM fallback).

Live: https://cicerollm.com

## What it is

- Decoder-only transformer, ~111M params (12 layers x 12 heads x 768 dim)
- Trained from a random init on a ~466M-token Latin corpus (v5 maximalist mix),
  30,000 steps, dropout 0.15 (no pretrained backbone, no English/Greek base)
- 32K SentencePiece-BPE tokenizer trained on the same corpus
- Canonical cloze **0.804** (first checkpoint to clear the 0.75 stretch goal);
  literary **0.746**; held-out (blind) **0.688**

It's a research artifact: autoregressive completion with temperature + top-k
sampling. No instruction tuning, no chat behavior — give it Latin and it
continues in Latin.

## Running it

Just open the live link. First load downloads the weights (~136 MB) and
compiles them; later prompts run from browser cache.

For local development see [DEPLOY.md](DEPLOY.md).

## How the weights are hosted

The model binary is hosted on the [Hugging Face Hub](https://huggingface.co/gmmeyer/cicero)
(CDN reflects the requesting origin, so cross-origin browser fetch works). It is
not committed to this repo — at ~136 MB it exceeds GitHub's 100 MB file limit,
and GitHub Pages does not serve Git LFS objects. The static site loads it at
runtime from HF. A [GitHub Release asset](https://github.com/gmmeyer/cicero/releases/latest)
mirrors the same file for CLI/direct download (GitHub's asset CDN sends no CORS
headers, so it is not usable as an in-browser source).
