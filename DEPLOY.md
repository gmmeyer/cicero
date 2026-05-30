# Deploying the Cicero LLM site (GitHub Pages)

Static, no build step. Browser-side inference via ONNX Runtime Web +
Transformers.js (both from CDN). The deployed default backend is WASM; WebGPU is
kept opt-in with `?gpu=1` because it produced bad logits for this model path.
The only artifact that isn't in this repo is the model binary — see step 2.

## 1. Enable GitHub Pages

Repo **Settings → Pages**:
- Source: **Deploy from a branch**
- Branch: **main**, folder **/ (root)**

Default URL is `https://gmmeyer.github.io/cicero/`. The `CNAME` file in this repo
points the custom domain **cicerollm.com** at it; finish the DNS setup in
Cloudflare (see step 4).

## 2. Host the model weights externally

`model.int8.onnx` is ~136 MB — over GitHub's 100 MB repo file limit, and
GitHub Pages does **not** serve Git LFS objects (it returns the LFS pointer,
not the binary). So the weights live off-repo, in two places.

**Primary — Hugging Face Hub** (`app.js` loads from here by default):

```bash
hf upload gmmeyer/cicero /path/to/model.int8.onnx model.int8.onnx
```

Served at:
`https://huggingface.co/gmmeyer/cicero/resolve/main/model.int8.onnx`

**Mirror — GitHub Release asset** (CLI / direct download only):

```bash
gh release create v1 /path/to/model.int8.onnx \
  --repo gmmeyer/cicero \
  --title "Cicero LLM weights v1 (int8)" \
  --notes "111M-param Latin GPT, int8 ONNX."
```

Served at:
`https://github.com/gmmeyer/cicero/releases/latest/download/model.int8.onnx`

⚠️ GitHub's release-asset CDN sends **no CORS headers**, so this URL **cannot**
be fetched cross-origin by the browser app — it is a download mirror, not a
browser fallback. The browser only ever loads from Hugging Face.

To ship a newer model: re-`hf upload` (overwrites `main`) and cut a new GitHub
release with the same asset filename — both URLs always resolve to the latest.

## 3. Local testing

The model URL is overridable so you can test against a local copy without the
Release:

```bash
# put model.int8.onnx in ./model/, then:
python -m http.server 8000
# open http://localhost:8000/?model=./model/model.int8.onnx
```

## 4. Custom domain (cicerollm.com via Cloudflare)

The `CNAME` file in this repo tells GitHub Pages to serve `cicerollm.com`.
Finish in Cloudflare DNS:

- Apex `cicerollm.com` → four `A` records to GitHub Pages IPs
  (`185.199.108.153`, `.109.153`, `.110.153`, `.111.153`), or an `ALIAS`/`CNAME`
  flattening to `gmmeyer.github.io`.
- `www` → `CNAME` to `gmmeyer.github.io`.
- Set the records to **DNS only** (grey cloud) initially so GitHub can issue the
  Let's Encrypt cert; once Pages shows the cert as issued, proxying can be
  re-enabled if desired.
- In repo **Settings → Pages**, confirm the custom domain is `cicerollm.com` and
  enable **Enforce HTTPS**.

## Files

- `index.html` — UI
- `app.js` — inference (tokenizer + ORT-Web); `MODEL_URL` at top, `?model=` override
- `CNAME` — custom domain for GitHub Pages (`cicerollm.com`)
- `model/tokenizer.json`, `model/tokenizer_config.json`, `model/config.json` — small, committed
- `model/*.onnx` — gitignored; lives on HF + the GitHub Release
