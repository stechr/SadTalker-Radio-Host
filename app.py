"""Local web UI for SadTalker demo.

Provides a browser interface to upload image + audio, trigger SageMaker
async inference, poll for results, and play/download the output video.
"""

import json
import os
import uuid

import boto3
from flask import Flask, Response, jsonify, request, send_file, render_template_string

app = Flask(__name__)

# --- Configuration -----------------------------------------------------------

_session = boto3.Session()
REGION = os.environ.get("AWS_REGION", _session.region_name or "us-east-1")
ACCOUNT_ID = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]
BUCKET = f"sadtalker-demo-{ACCOUNT_ID}"
ENDPOINT = "sadtalker-async"

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_BASE_DIR, "..", "results")
CONFIG_FILE = os.path.join(_BASE_DIR, "..", "config.json")

os.makedirs(RESULTS_DIR, exist_ok=True)

s3 = boto3.client("s3", region_name=REGION)
sm_runtime = boto3.client("sagemaker-runtime", region_name=REGION)

DEFAULT_CONFIG = {
    "enhancer": "gfpgan",
    "still_mode": True,
    "preprocess": "crop",
    "expression_scale": 1.0,
    "pose_style": 0,
}


# --- Helpers -----------------------------------------------------------------

def load_config() -> dict:
    """Load config from disk, falling back to defaults for missing keys."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return {**DEFAULT_CONFIG, **json.load(f)}
    return DEFAULT_CONFIG.copy()


def save_config(cfg: dict) -> None:
    """Persist config to disk."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def _upload_payload(job_id: str, payload: str) -> str:
    """Upload JSON payload to S3 and return its URI for async invocation."""
    key = f"async-input/{job_id}.json"
    s3.put_object(Bucket=BUCKET, Key=key, Body=payload, ContentType="application/json")
    return f"s3://{BUCKET}/{key}"


def _get_local_path(job_id: str) -> str:
    """Return local file path for a job's result video."""
    return os.path.join(RESULTS_DIR, f"{job_id}.mp4")


# --- HTML Template -----------------------------------------------------------

HTML = """<!DOCTYPE html>
<html><head><title>SadTalker Demo</title>
<style>
body { font-family: system-ui; max-width: 900px; margin: 40px auto; padding: 0 20px; }
h1 { color: #232f3e; }
.upload-area { border: 2px dashed #ccc; padding: 20px; margin: 10px 0; border-radius: 8px; }
.settings { background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 15px 0; }
.settings h3 { margin-top: 0; }
.setting-row { display: flex; align-items: center; margin: 10px 0; gap: 15px; flex-wrap: wrap; }
.setting-row label { min-width: 160px; font-weight: 500; }
.setting-row select, .setting-row input { padding: 6px 10px; border: 1px solid #ccc; border-radius: 4px; }
.setting-row input[type=range] { width: 200px; }
.setting-row .desc { display: block; color: #666; font-size: 12px; margin-top: 4px; flex-basis: 100%; }
button { background: #ff9900; border: none; padding: 12px 24px; color: #000;
         font-weight: bold; border-radius: 4px; cursor: pointer; font-size: 16px; }
button:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-secondary { background: #232f3e; color: #fff; padding: 8px 16px; font-size: 14px; }
#status { margin: 20px 0; padding: 15px; background: #f0f0f0; border-radius: 4px; display: none; }
video { width: 100%; max-width: 640px; margin-top: 20px; border-radius: 8px; }
.spinner { display: inline-block; width: 16px; height: 16px; border: 2px solid #ccc;
           border-top-color: #ff9900; border-radius: 50%; animation: spin 1s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
#download-area { margin-top: 10px; display: none; }
</style></head><body>
<h1>&#127897;&#65039; SadTalker &mdash; Radio Host Demo</h1>

<div class="upload-area">
  <label><strong>Photo</strong> (JPG/PNG of the host):<br>
  <input type="file" id="image" accept="image/jpeg,image/png"></label>
</div>
<div class="upload-area">
  <label><strong>Audio</strong> (WAV/MP3/M4A/OGG/FLAC):<br>
  <input type="file" id="audio" accept="audio/wav,audio/mpeg,audio/mp4,audio/ogg,audio/flac"></label>
</div>

<div class="settings">
  <h3>&#9881;&#65039; Generation Settings</h3>
  <div class="setting-row">
    <label for="enhancer">Face Enhancer:</label>
    <select id="enhancer">
      <option value="gfpgan">GFPGAN (sharper face)</option>
      <option value="RestoreFormer">RestoreFormer (natural)</option>
      <option value="none">None (fastest)</option>
    </select>
    <span class="desc">Post-processing to sharpen the generated face. GFPGAN produces crisp results;
      RestoreFormer looks more natural but softer; None skips enhancement for faster output.</span>
  </div>
  <div class="setting-row">
    <label for="still_mode">Still Mode:</label>
    <select id="still_mode">
      <option value="true">On (minimal head motion)</option>
      <option value="false">Off (natural head movement)</option>
    </select>
    <span class="desc">When ON, the head stays mostly still &mdash; good for a &ldquo;news anchor&rdquo; look.
      When OFF, the model adds natural head tilts and nods driven by the audio.</span>
  </div>
  <div class="setting-row">
    <label for="preprocess">Preprocess:</label>
    <select id="preprocess">
      <option value="crop">Crop (face only)</option>
      <option value="resize">Resize (keep background)</option>
      <option value="full">Full (high-res, slower)</option>
    </select>
    <span class="desc">How the input image is handled. Crop extracts just the face (fastest, 256&times;256).
      Resize keeps the full image but scales it down. Full preserves original resolution (slowest, best quality).</span>
  </div>
  <div class="setting-row">
    <label for="expression_scale">Expression Scale:</label>
    <input type="range" id="expression_scale" min="0.5" max="2.0" step="0.1" value="1.0">
    <span id="expression_val">1.0</span>
    <span class="desc">Controls how pronounced the facial expressions are. 1.0 = normal.
      Lower (0.5) = subtle/calm. Higher (1.5&ndash;2.0) = exaggerated/animated.</span>
  </div>
  <div class="setting-row">
    <label for="pose_style">Pose Style (0-45):</label>
    <input type="number" id="pose_style" min="0" max="45" value="0" style="width:60px">
    <span class="desc">Selects a pre-defined head pose trajectory. 0 = default.
      Each number produces a different combination of head tilts and turns.</span>
  </div>
  <button class="btn-secondary" onclick="saveSettings()">Save Settings</button>
  <button class="btn-secondary" onclick="resetSettings()" style="background:#666">Reset Defaults</button>
  <span id="save-confirm" style="color:green; margin-left:10px; display:none">&#10003; Saved</span>
</div>

<button id="go" onclick="run()">Generate Video</button>
<div id="status"></div>
<video id="result" controls style="display:none"></video>
<div id="download-area">
  <a id="download-link" class="btn-secondary"
     style="text-decoration:none; display:inline-block; margin-top:10px;">&#11015; Download Video</a>
  <span id="local-path" style="margin-left:15px; color:#666;"></span>
</div>

<script>
// Load saved settings on page load
fetch('/api/config').then(r => r.json()).then(cfg => {
  document.getElementById('enhancer').value = cfg.enhancer || 'gfpgan';
  document.getElementById('still_mode').value = String(cfg.still_mode);
  document.getElementById('preprocess').value = cfg.preprocess || 'crop';
  document.getElementById('expression_scale').value = cfg.expression_scale || 1.0;
  document.getElementById('expression_val').textContent = cfg.expression_scale || 1.0;
  document.getElementById('pose_style').value = cfg.pose_style || 0;
});

document.getElementById('expression_scale').oninput = function() {
  document.getElementById('expression_val').textContent = this.value;
};

function getSettings() {
  return {
    enhancer: document.getElementById('enhancer').value,
    still_mode: document.getElementById('still_mode').value === 'true',
    preprocess: document.getElementById('preprocess').value,
    expression_scale: parseFloat(document.getElementById('expression_scale').value),
    pose_style: parseInt(document.getElementById('pose_style').value, 10),
  };
}

function saveSettings() {
  fetch('/api/config', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(getSettings()),
  });
  const c = document.getElementById('save-confirm');
  c.textContent = '\\u2713 Saved';
  c.style.display = 'inline';
  setTimeout(() => c.style.display = 'none', 2000);
}

function resetSettings() {
  fetch('/api/config/reset', {method: 'POST'}).then(r => r.json()).then(cfg => {
    document.getElementById('enhancer').value = cfg.enhancer;
    document.getElementById('still_mode').value = String(cfg.still_mode);
    document.getElementById('preprocess').value = cfg.preprocess;
    document.getElementById('expression_scale').value = cfg.expression_scale;
    document.getElementById('expression_val').textContent = cfg.expression_scale;
    document.getElementById('pose_style').value = cfg.pose_style;
    const c = document.getElementById('save-confirm');
    c.textContent = '\\u2713 Reset';
    c.style.display = 'inline';
    setTimeout(() => { c.style.display = 'none'; c.textContent = '\\u2713 Saved'; }, 2000);
  });
}

async function run() {
  const img = document.getElementById('image').files[0];
  const aud = document.getElementById('audio').files[0];
  if (!img || !aud) { alert('Please select both an image and an audio file.'); return; }

  const btn = document.getElementById('go');
  const status = document.getElementById('status');
  const video = document.getElementById('result');
  const dlArea = document.getElementById('download-area');

  btn.disabled = true;
  status.style.display = 'block';
  video.style.display = 'none';
  dlArea.style.display = 'none';
  status.innerHTML = '<span class="spinner"></span> Uploading files...';

  const settings = getSettings();
  // Persist settings before run
  fetch('/api/config', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(settings),
  });

  const form = new FormData();
  form.append('image', img);
  form.append('audio', aud);
  form.append('settings', JSON.stringify(settings));

  const resp = await fetch('/api/invoke', {method: 'POST', body: form});
  const data = await resp.json();
  if (data.error) {
    status.textContent = '\\u274c ' + data.error;
    btn.disabled = false;
    return;
  }

  status.innerHTML = '<span class="spinner"></span> Processing (1\\u20133 minutes for longer audio)...';
  const jobId = data.job_id;

  while (true) {
    await new Promise(r => setTimeout(r, 5000));
    const poll = await fetch('/api/status/' + jobId);
    const st = await poll.json();
    if (st.status === 'done') {
      status.textContent = '\\u2705 Done!';
      video.src = '/api/video/' + jobId + '?t=' + Date.now();
      video.style.display = 'block';
      dlArea.style.display = 'block';
      document.getElementById('download-link').href = '/api/download/' + jobId;
      document.getElementById('local-path').textContent = 'Saved to: results/' + jobId + '.mp4';
      break;
    } else if (st.status === 'error') {
      status.textContent = '\\u274c ' + (st.error || 'Processing failed');
      break;
    }
    status.innerHTML = '<span class="spinner"></span> ' + st.status;
  }
  btn.disabled = false;
}
</script></body></html>"""


# --- API Routes --------------------------------------------------------------

@app.route("/")
def index() -> str:
    """Serve the main UI page."""
    return render_template_string(HTML)


@app.route("/api/config", methods=["GET"])
def get_config() -> Response:
    """Return current generation settings."""
    return jsonify(load_config())


@app.route("/api/config", methods=["POST"])
def set_config() -> Response:
    """Save generation settings."""
    cfg = request.json
    if not cfg:
        return jsonify({"error": "No JSON body provided"}), 400
    save_config(cfg)
    return jsonify({"ok": True})


@app.route("/api/config/reset", methods=["POST"])
def reset_config() -> Response:
    """Reset settings to defaults."""
    save_config(DEFAULT_CONFIG)
    return jsonify(DEFAULT_CONFIG)


@app.route("/api/invoke", methods=["POST"])
def invoke() -> Response:
    """Upload files to S3 and trigger async inference."""
    image = request.files.get("image")
    audio = request.files.get("audio")
    if not image or not audio:
        return jsonify({"error": "Both image and audio files are required"}), 400

    settings = json.loads(request.form.get("settings", "{}"))
    cfg = {**load_config(), **settings}
    save_config(cfg)

    job_id = uuid.uuid4().hex[:8]
    img_key = f"input/{job_id}/image{_ext(image.filename)}"
    aud_key = f"input/{job_id}/audio{_ext(audio.filename)}"
    out_key = f"output/{job_id}/result.mp4"

    try:
        s3.upload_fileobj(image, BUCKET, img_key)
        s3.upload_fileobj(audio, BUCKET, aud_key)
    except Exception as e:
        return jsonify({"error": f"Upload failed: {e}"}), 500

    payload = json.dumps({
        "image_s3_uri": f"s3://{BUCKET}/{img_key}",
        "audio_s3_uri": f"s3://{BUCKET}/{aud_key}",
        "output_s3_uri": f"s3://{BUCKET}/{out_key}",
        "enhancer": cfg.get("enhancer", "gfpgan"),
        "still_mode": cfg.get("still_mode", True),
        "preprocess": cfg.get("preprocess", "crop"),
        "expression_scale": cfg.get("expression_scale", 1.0),
        "pose_style": cfg.get("pose_style", 0),
    })

    try:
        resp = sm_runtime.invoke_endpoint_async(
            EndpointName=ENDPOINT,
            ContentType="application/json",
            InputLocation=_upload_payload(job_id, payload),
        )
    except Exception as e:
        return jsonify({"error": f"Failed to invoke endpoint: {e}"}), 500

    return jsonify({"job_id": job_id, "inference_id": resp.get("InferenceId")})


@app.route("/api/status/<job_id>")
def status(job_id: str) -> Response:
    """Poll for job completion by checking if output exists in S3."""
    if not _is_valid_job_id(job_id):
        return jsonify({"error": "Invalid job ID"}), 400

    out_key = f"output/{job_id}/result.mp4"
    try:
        s3.head_object(Bucket=BUCKET, Key=out_key)
        # Auto-download to local results folder
        local_path = _get_local_path(job_id)
        if not os.path.exists(local_path):
            s3.download_file(BUCKET, out_key, local_path)
        return jsonify({"status": "done"})
    except s3.exceptions.ClientError:
        return jsonify({"status": "Processing (waiting for GPU inference)..."})


@app.route("/api/video/<job_id>")
def video(job_id: str) -> Response:
    """Stream the result video."""
    if not _is_valid_job_id(job_id):
        return jsonify({"error": "Invalid job ID"}), 400

    local_path = _get_local_path(job_id)
    if not os.path.exists(local_path):
        out_key = f"output/{job_id}/result.mp4"
        s3.download_file(BUCKET, out_key, local_path)
    return send_file(local_path, mimetype="video/mp4")


@app.route("/api/download/<job_id>")
def download(job_id: str) -> Response:
    """Download the result video as an attachment."""
    if not _is_valid_job_id(job_id):
        return jsonify({"error": "Invalid job ID"}), 400

    local_path = _get_local_path(job_id)
    if not os.path.exists(local_path):
        out_key = f"output/{job_id}/result.mp4"
        s3.download_file(BUCKET, out_key, local_path)
    return send_file(local_path, mimetype="video/mp4", as_attachment=True,
                     download_name=f"sadtalker-{job_id}.mp4")


# --- Utilities ---------------------------------------------------------------

def _ext(filename: str) -> str:
    """Extract file extension (with dot) from filename."""
    if filename and "." in filename:
        return "." + filename.rsplit(".", 1)[1].lower()
    return ""


def _is_valid_job_id(job_id: str) -> bool:
    """Validate job ID format to prevent path traversal."""
    return len(job_id) == 8 and job_id.isalnum()


# --- Entry point -------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n🎙️  SadTalker Demo UI")
    print(f"   Open http://localhost:5000 in your browser")
    print(f"   Results saved to: {os.path.abspath(RESULTS_DIR)}")
    print(f"   Bucket: {BUCKET} ({REGION})\n")
    app.run(host="127.0.0.1", port=5000, debug=True)
