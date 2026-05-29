# Cicero LLM

A 100M-parameter Latin language model, trained from scratch, running entirely
in your browser. No server, no API — the int8-quantized model downloads once
and runs client-side via ONNX Runtime Web (WebGPU, with a WASM fallback).

Live: https://cicerollm.com

## What it is

- Decoder-only transformer, ~111M params (12 layers x 12 heads x 768 dim)
- Trained from a random init on a ~466M-token Latin corpus (no pretrained
  backbone, no English/Greek base), then continued-pretrained on a targeted
  classical-grammar curriculum mixed 30/70 with clean classical replay
- 32K SentencePiece-BPE tokenizer trained on the same corpus
- Held-out (blind) cloze **0.72**, literary **0.82**, grammar-probe set **0.82**

It's a research artifact: autoregressive completion with temperature + top-k
sampling and a repetition penalty. No instruction tuning, no chat behavior —
give it Latin and it continues in Latin. The curriculum-tuning step pushes
generation toward classical register and away from the medieval/neo-Latin
contamination of the base model.

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
