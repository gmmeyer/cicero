// Cicero LLM browser inference (MAX-V5, int8 ONNX, ORT-Web + Transformers.js tokenizer)
//
// Stack:
//   - onnxruntime-web (1.20+) for the ONNX model inference (WebGPU when available)
//   - @xenova/transformers v3 for the SentencePiece-BPE tokenizer (loads tokenizer.json)
// Both libs come from a CDN so no build step is needed.

import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.webgpu.min.mjs';
import { AutoTokenizer, env as txEnv } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3';

// Transformers.js downloads tokenizers via fetch; point it at our local model dir.
txEnv.allowLocalModels = true;
txEnv.useBrowserCache = true;
// Disable remote model download; we provide tokenizer.json locally.
txEnv.allowRemoteModels = false;
txEnv.localModelPath = './model/';

// Configure ORT-Web: point at the CDN copy of the WASM/WebGPU artifacts
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

// The int8 ONNX model (~136 MB) exceeds GitHub's 100 MB repo file limit and
// GitHub Pages does not serve Git LFS objects, so the weights are hosted on the
// Hugging Face Hub — purpose-built for model delivery, with a CDN that reflects
// the requesting Origin (verified CORS from this domain). A GitHub Release asset
// mirrors the file for CLI/direct download, but GitHub's asset CDN sends no CORS
// headers, so it can't serve a cross-origin browser fetch — HF is the only
// in-browser source. Override with ?model=<url> (e.g. ./model/model.int8.onnx
// for local testing).
const HF_MODEL_URL =
  'https://huggingface.co/gmmeyer/cicero/resolve/main/model.int8.onnx';
const MODEL_URL =
  new URLSearchParams(location.search).get('model') || HF_MODEL_URL;

const statusEl = document.getElementById('status');
const outputEl = document.getElementById('output');
const promptEl = document.getElementById('prompt');
const goBtn   = document.getElementById('go');
const stopBtn = document.getElementById('stop');

let session = null;
let tokenizer = null;
let stopRequested = false;

function setStatus(msg, isErr=false) {
  statusEl.textContent = msg;
  statusEl.className = isErr ? 'err' : '';
}

// Download the model once, with a streaming progress bar and source
// fallback. Returns the weights as a Uint8Array so the WebGPU→WASM
// fallback below reuses these bytes instead of re-downloading 136 MB.
// Sources are tried in order; the browser HTTP cache makes reloads instant.
async function fetchModelBytes() {
  const sources = [MODEL_URL];  // HF Hub by default; ?model= overrides for local
  let lastErr = null;
  for (const url of sources) {
    try {
      const resp = await fetch(url, { cache: 'force-cache' });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const total = Number(resp.headers.get('content-length')) || 0;
      const reader = resp.body.getReader();
      const chunks = [];
      let received = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.length;
        const mb = (received / 1048576).toFixed(0);
        const pct = total ? ` (${Math.round(100 * received / total)}%)` : '';
        setStatus(`Downloading model: ${mb} MB${pct}, first load only…`);
      }
      const bytes = new Uint8Array(received);
      let offset = 0;
      for (const c of chunks) { bytes.set(c, offset); offset += c.length; }
      return bytes;
    } catch (e) {
      console.warn(`model fetch failed from ${url}:`, e);
      lastErr = e;
    }
  }
  throw new Error(`could not download model from any source: ${lastErr?.message || lastErr}`);
}

async function loadAll() {
  try {
    setStatus('Loading tokenizer…');
    tokenizer = await AutoTokenizer.from_pretrained('.');

    const modelBytes = await fetchModelBytes();

    setStatus('Compiling model…');
    // Create the session from the in-memory bytes; try WebGPU, fall back
    // to WASM. Both attempts reuse modelBytes (no second download).
    try {
      session = await ort.InferenceSession.create(modelBytes, {
        executionProviders: ['webgpu', 'wasm'],
        graphOptimizationLevel: 'all',
      });
    } catch (e) {
      console.warn('WebGPU init failed, falling back to WASM:', e);
      session = await ort.InferenceSession.create(modelBytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
    }
    const provName = session.handler?._backend?.name || 'unknown';
    setStatus(`Ready. Backend: ${provName}. Click Generate.`);
    goBtn.disabled = false;
  } catch (err) {
    console.error(err);
    setStatus(`Load failed: ${err.message || err}`, true);
  }
}

// Sample next token from logits using temperature + top-k.
function sampleNext(logits, temperature, topK) {
  // logits is a Float32Array of length vocab_size
  // Apply temperature
  const t = Math.max(temperature, 1e-3);
  const scaled = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) scaled[i] = logits[i] / t;
  // Top-k: get the k largest indices
  const k = Math.min(topK, scaled.length);
  // simple partial sort: find top-k
  const idxs = Array.from({length: scaled.length}, (_, i) => i);
  idxs.sort((a, b) => scaled[b] - scaled[a]);
  const topIdx = idxs.slice(0, k);
  // softmax over top-k
  let maxL = -Infinity;
  for (const i of topIdx) if (scaled[i] > maxL) maxL = scaled[i];
  let sum = 0;
  const probs = new Float32Array(k);
  for (let j = 0; j < k; j++) {
    probs[j] = Math.exp(scaled[topIdx[j]] - maxL);
    sum += probs[j];
  }
  // sample
  const r = Math.random() * sum;
  let acc = 0;
  for (let j = 0; j < k; j++) {
    acc += probs[j];
    if (r <= acc) return topIdx[j];
  }
  return topIdx[k - 1];
}

async function generate() {
  if (!session || !tokenizer) { setStatus('Model not loaded yet', true); return; }
  stopRequested = false;
  goBtn.disabled = true;
  stopBtn.disabled = false;

  const prompt = promptEl.value;
  const ntokens = parseInt(document.getElementById('ntokens').value, 10);
  const temperature = parseFloat(document.getElementById('temp').value);
  const topK = parseInt(document.getElementById('topk').value, 10);

  const blockSize = 2048;

  // Encode prompt
  const encoded = await tokenizer(prompt, { add_special_tokens: false });
  let ids = Array.from(encoded.input_ids.data, x => Number(x));
  // BOS prepend to match training (sentencepiece convention)
  if (tokenizer.bos_token_id != null && ids[0] !== tokenizer.bos_token_id) {
    ids = [tokenizer.bos_token_id, ...ids];
  }

  outputEl.textContent = prompt;
  // Append a marker so the generated tokens are visually distinguished
  const generatedSpan = document.createElement('span');
  generatedSpan.className = 'generated';
  outputEl.appendChild(generatedSpan);

  const startTime = performance.now();
  let nGenerated = 0;

  for (let step = 0; step < ntokens && !stopRequested; step++) {
    // Window to last block_size tokens
    const window = ids.length > blockSize ? ids.slice(ids.length - blockSize) : ids;
    const bigIds = BigInt64Array.from(window.map(BigInt));
    const tensor = new ort.Tensor('int64', bigIds, [1, window.length]);

    const t0 = performance.now();
    const outputs = await session.run({ input_ids: tensor });
    const logits = outputs.logits.data;  // Float32Array (V,)
    const next = sampleNext(logits, temperature, topK);

    // Stop on EOS — restart on a fresh sentence rather than streaming through.
    if (tokenizer.eos_token_id != null && next === tokenizer.eos_token_id) {
      setStatus(`Hit EOS at token ${nGenerated}. Stopping.`);
      break;
    }

    ids.push(next);
    nGenerated++;

    // Decode just the new token for incremental display.
    // skip_special_tokens strips <s>/</s>/<unk>/etc. so they don't appear in UI.
    const newPiece = await tokenizer.decode([next], { skip_special_tokens: true });
    generatedSpan.textContent += newPiece;

    const elapsedMs = performance.now() - startTime;
    const tps = (nGenerated / elapsedMs * 1000).toFixed(1);
    setStatus(`Generated ${nGenerated}/${ntokens} tokens (${tps} tok/s)`);

    // Yield to UI
    if (nGenerated % 4 === 0) await new Promise(r => setTimeout(r, 0));
  }

  const total = ((performance.now() - startTime) / 1000).toFixed(1);
  setStatus(`Done. ${nGenerated} tokens in ${total}s ` +
            `(${(nGenerated/parseFloat(total)).toFixed(1)} tok/s).`);
  goBtn.disabled = false;
  stopBtn.disabled = true;
}

goBtn.addEventListener('click', () => generate().catch(e => {
  console.error(e);
  setStatus(`Error: ${e.message || e}`, true);
  goBtn.disabled = false;
  stopBtn.disabled = true;
}));
stopBtn.addEventListener('click', () => { stopRequested = true; });

loadAll();
