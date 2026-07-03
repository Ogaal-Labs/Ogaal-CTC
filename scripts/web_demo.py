#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request
from werkzeug.utils import secure_filename

from infer_ogaal_ctc import (
    DEFAULT_MODEL_DIR,
    SUPPORTED_SUFFIXES,
    build_decoder,
    ensure_model,
    save_single_result,
    transcribe_audio,
)


APP_DIR = Path(__file__).resolve().parents[1]
DEMO_OUTPUT_DIR = APP_DIR / "web_demo_outputs"
DEMO_UPLOAD_DIR = APP_DIR / "web_demo_uploads"

HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ogaal CTC Demo</title>
  <style>
    :root {
      --bg: #f3efe5;
      --panel: #fffaf1;
      --ink: #1f2a2a;
      --accent: #0f766e;
      --accent-2: #c2410c;
      --line: #d6cfc0;
    }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(194, 65, 12, 0.14), transparent 30%),
        linear-gradient(180deg, #f8f3e8 0%, var(--bg) 100%);
      color: var(--ink);
    }
    .wrap {
      max-width: 980px;
      margin: 0 auto;
      padding: 32px 18px 60px;
    }
    .hero {
      margin-bottom: 24px;
    }
    h1 {
      margin: 0 0 10px;
      font-size: clamp(2rem, 4vw, 3.4rem);
      line-height: 0.95;
      letter-spacing: -0.03em;
    }
    .sub {
      max-width: 760px;
      font-size: 1.05rem;
      line-height: 1.5;
    }
    .grid {
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 18px;
    }
    .card {
      background: color-mix(in srgb, var(--panel) 92%, white);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 18px 40px rgba(47, 39, 24, 0.08);
    }
    label {
      display: block;
      font-size: 0.9rem;
      font-weight: 700;
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    select, input[type=file], button {
      font: inherit;
    }
    select, input[type=file] {
      width: 100%;
      margin-bottom: 14px;
      padding: 12px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: white;
    }
    .row {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin: 8px 0 16px;
    }
    button {
      border: 0;
      border-radius: 999px;
      padding: 12px 16px;
      cursor: pointer;
      font-weight: 700;
    }
    .primary { background: var(--accent); color: white; }
    .secondary { background: #e9dfca; color: var(--ink); }
    .danger { background: var(--accent-2); color: white; }
    button:disabled {
      opacity: 0.45;
      cursor: not-allowed;
    }
    .status {
      min-height: 24px;
      font-size: 0.98rem;
      margin-bottom: 14px;
    }
    .pill {
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: #efe7d3;
      margin: 4px 6px 0 0;
      font-size: 0.9rem;
    }
    textarea, pre {
      width: 100%;
      box-sizing: border-box;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: #fffdf8;
      padding: 14px;
      font: inherit;
    }
    textarea {
      min-height: 180px;
      resize: vertical;
    }
    pre {
      min-height: 220px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }
    audio {
      width: 100%;
      margin: 12px 0 8px;
    }
    .hint {
      font-size: 0.9rem;
      line-height: 1.45;
      opacity: 0.82;
    }
    @media (max-width: 860px) {
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <h1>Ogaal CTC</h1>
      <div class="sub">
        Record Somali speech in the browser or upload an audio file, then run the released Somali CTC package with the published decode path.
      </div>
    </div>

    <div class="grid">
      <section class="card">
        <label for="upload">Upload Audio Instead</label>
        <input id="upload" type="file" accept="audio/*">

        <label>Record In Browser</label>
        <div class="row">
          <button id="startBtn" class="primary">Start Recording</button>
          <button id="stopBtn" class="danger" disabled>Stop</button>
          <button id="runBtn" class="secondary" disabled>Transcribe Recording</button>
        </div>

        <audio id="player" controls></audio>
        <div id="status" class="status">Idle.</div>
        <div class="hint">
          This demo is record, stop, then transcribe. It is not a live-streaming ASR dashboard.
        </div>
      </section>

      <section class="card">
        <label>Run Details</label>
        <div id="meta"></div>
        <pre id="chunks">No transcription yet.</pre>
      </section>
    </div>

    <section class="card" style="margin-top: 18px;">
      <label for="transcript">Transcript</label>
      <textarea id="transcript" readonly placeholder="Transcript will appear here."></textarea>
    </section>
  </div>

  <script>
    const startBtn = document.getElementById("startBtn");
    const stopBtn = document.getElementById("stopBtn");
    const runBtn = document.getElementById("runBtn");
    const upload = document.getElementById("upload");
    const player = document.getElementById("player");
    const statusEl = document.getElementById("status");
    const transcriptEl = document.getElementById("transcript");
    const metaEl = document.getElementById("meta");
    const chunksEl = document.getElementById("chunks");

    let mediaRecorder = null;
    let recordedChunks = [];
    let recordedBlob = null;

    function setStatus(text) {
      statusEl.textContent = text;
    }

    function resetOutput() {
      transcriptEl.value = "";
      metaEl.innerHTML = "";
      chunksEl.textContent = "No transcription yet.";
    }

    function renderMeta(data) {
      const fields = [
        ["Decode path", "transcript_only"],
        ["Audio", data.audio_name],
        ["Duration", `${data.duration_seconds}s`],
        ["Runtime", `${data.runtime_seconds}s`],
        ["Chunks", data.chunk_count],
        ["VAD chunking", data.used_chunking ? "yes" : "no"],
      ];
      metaEl.innerHTML = fields.map(([k, v]) => `<span class="pill"><strong>${k}:</strong> ${v}</span>`).join("");
    }

    async function transcribe(blob, filename) {
      const form = new FormData();
      form.append("audio", blob, filename);
      form.append("decoder", "transcript_only");

      setStatus("Transcribing...");
      runBtn.disabled = true;
      startBtn.disabled = true;
      stopBtn.disabled = true;
      upload.disabled = true;

      try {
        const response = await fetch("/api/transcribe", { method: "POST", body: form });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Transcription failed");
        }
        transcriptEl.value = data.transcript_text || "";
        renderMeta(data);
        chunksEl.textContent = JSON.stringify(data.chunks, null, 2);
        setStatus(`Finished. Saved result to ${data.saved_json_path}`);
      } catch (error) {
        setStatus(`Error: ${error.message}`);
      } finally {
        startBtn.disabled = false;
        upload.disabled = false;
        runBtn.disabled = !recordedBlob;
      }
    }

    startBtn.addEventListener("click", async () => {
      resetOutput();
      recordedBlob = null;
      upload.value = "";
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        recordedChunks = [];
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = event => {
          if (event.data.size > 0) {
            recordedChunks.push(event.data);
          }
        };
        mediaRecorder.onstop = () => {
          recordedBlob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || "audio/webm" });
          player.src = URL.createObjectURL(recordedBlob);
          runBtn.disabled = false;
          stream.getTracks().forEach(track => track.stop());
          setStatus("Recording ready. Click transcribe.");
        };
        mediaRecorder.start();
        startBtn.disabled = true;
        stopBtn.disabled = false;
        runBtn.disabled = true;
        setStatus("Recording...");
      } catch (error) {
        setStatus(`Microphone error: ${error.message}`);
      }
    });

    stopBtn.addEventListener("click", () => {
      if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
      }
      stopBtn.disabled = true;
      startBtn.disabled = false;
    });

    runBtn.addEventListener("click", async () => {
      if (recordedBlob) {
        await transcribe(recordedBlob, "browser_recording.webm");
      }
    });

    upload.addEventListener("change", async () => {
      const file = upload.files[0];
      if (!file) {
        return;
      }
      resetOutput();
      recordedBlob = null;
      runBtn.disabled = true;
      player.src = URL.createObjectURL(file);
      await transcribe(file, file.name);
    });
  </script>
</body>
</html>
"""


app = Flask(__name__)
_runtime = {
    "processor": None,
    "model": None,
    "device": None,
    "blank_id": None,
    "decoder_cache": {},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small browser demo for the released Ogaal CTC Somali workflow.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    return parser.parse_args()


def ensure_runtime(model_dir: Path) -> None:
    if _runtime["processor"] is not None:
        return
    processor, model, device, blank_id = ensure_model(model_dir)
    _runtime["processor"] = processor
    _runtime["model"] = model
    _runtime["device"] = device
    _runtime["blank_id"] = blank_id


def get_decoder(decoder_name: str):
    if decoder_name == "greedy":
        return None
    decoder_cache = _runtime["decoder_cache"]
    if decoder_name not in decoder_cache:
        decoder_cache[decoder_name] = build_decoder(
            _runtime["processor"],
            decoder_name=decoder_name,
            model_dir=app.config["MODEL_DIR"],
        )
    return decoder_cache[decoder_name]


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_SUFFIXES


@app.get("/")
def index():
    return render_template_string(HTML)


@app.post("/api/transcribe")
def api_transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "Missing audio upload"}), 400

    audio = request.files["audio"]
    decoder_name = request.form.get("decoder", "transcript_only")
    if decoder_name not in {"transcript_only", "greedy"}:
        return jsonify({"error": f"Unsupported decoder: {decoder_name}"}), 400

    raw_name = secure_filename(audio.filename or "recording.webm")
    if not allowed_file(raw_name):
        raw_name = f"{Path(raw_name).stem or 'recording'}.webm"

    DEMO_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamped_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{raw_name}"
    persisted_path = DEMO_UPLOAD_DIR / stamped_name
    audio.save(persisted_path)

    ensure_runtime(app.config["MODEL_DIR"])
    result = transcribe_audio(
        audio_path=persisted_path,
        model_dir=app.config["MODEL_DIR"],
        processor=_runtime["processor"],
        model=_runtime["model"],
        device=_runtime["device"],
        blank_id=_runtime["blank_id"],
        decoder_name=decoder_name,
        decoder=get_decoder(decoder_name),
        beam_width=128,
        chunk_args={
            "long_audio_seconds": 18.0,
            "max_chunk_seconds": 18.0,
            "chunk_overlap_seconds": 1.0,
            "frame_ms": 30,
            "hop_ms": 10,
            "min_speech_ms": 250,
            "min_silence_ms": 350,
            "speech_pad_ms": 150,
        },
    )
    save_single_result(result, DEMO_OUTPUT_DIR)
    result["saved_json_path"] = str(DEMO_OUTPUT_DIR / f"{Path(result['audio_name']).stem}.json")
    return jsonify(result)


def main() -> None:
    args = parse_args()
    app.config["MODEL_DIR"] = args.model_dir
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
