# SadTalker Radio Host — Lip-Sync Video Generator

Generate realistic lip-synced videos from a single photo and an audio file. Upload a picture of a radio host and a voice recording, and get back a video of the host speaking the audio — with synchronized lip movements, natural expressions, and optional head motion.

Built on [SadTalker](https://github.com/OpenTalker/SadTalker) (CVPR 2023), deployed as a serverless GPU endpoint on AWS.

## Demo

1. Upload a front-facing photo (JPG/PNG)
2. Upload an audio clip (WAV/MP3)
3. Adjust generation settings (expression, head motion, quality)
4. Click **Generate Video** — result plays in-browser in 1–3 minutes

## Architecture

```
┌──────────────────┐       ┌──────────────┐       ┌──────────────────────────┐
│  Local Web UI    │──────▶│   Amazon S3  │──────▶│  SageMaker Async Endpoint │
│  (Flask, :5000)  │       │              │       │  ml.g5.xlarge (GPU)       │
│                  │◀──────│  input/      │◀──────│  SadTalker + GFPGAN       │
│  Upload → Poll   │       │  output/     │       │  CUDA 11.8                │
│  → Play video    │       └──────────────┘       └──────────────────────────┘
└──────────────────┘
```

**Components:**
- **container/** — Docker image with SadTalker, model checkpoints, and a Flask inference server
- **webapp/** — Local web UI for uploading files, triggering inference, and playing results
- **scripts/** — Deployment and cleanup automation

## Prerequisites

- **AWS Account** with permissions for: SageMaker, ECR, S3, CodeBuild, IAM
- **AWS CLI** configured with a default profile
- **Python 3.10+**
- **Service Quota:** At least 1× `ml.g5.xlarge` for SageMaker endpoint usage in your region

> **Note:** Docker is NOT required locally — the container is built in the cloud via AWS CodeBuild.

## Quick Start

### 1. Deploy Infrastructure

```bash
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

This will:
1. Create an S3 bucket for inputs/outputs
2. Create an ECR repository and build the Docker image via CodeBuild (~10 min)
3. Create IAM roles for SageMaker and CodeBuild
4. Deploy a SageMaker Async Inference endpoint (~10 min)

Total deployment time: ~20 minutes.

### 2. Run the Web UI

```bash
cd webapp
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000 in your browser.

### 3. Generate a Video

1. Select a **photo** — front-facing, well-lit, at least 256×256 pixels
2. Select an **audio file** — clear speech, WAV or MP3
3. Adjust settings if desired (see below)
4. Click **Generate Video**
5. Result auto-downloads to `results/` and plays in-browser

### 4. Cleanup

```bash
./scripts/cleanup.sh
```

Removes the SageMaker endpoint, ECR repository, and S3 bucket.

## Generation Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Face Enhancer** | GFPGAN | Post-processing to sharpen the generated face. GFPGAN = crisp; RestoreFormer = natural but softer; None = fastest |
| **Still Mode** | On | When ON, head stays mostly still (news anchor look). When OFF, adds natural head tilts and nods driven by audio |
| **Preprocess** | Crop | How the input image is handled. Crop = face only (256×256, fastest). Resize = keeps background. Full = original resolution (slowest, best quality) |
| **Expression Scale** | 1.0 | How pronounced facial expressions are. 0.5 = subtle/calm. 1.5–2.0 = exaggerated/animated |
| **Pose Style** | 0 | Pre-defined head pose trajectory (0–45). Each number produces different head tilts and turns |

Settings are persisted locally in `config.json` and reused across sessions.

## Responsible Use

This tool generates synthetic video of real people. By using it, you agree to:

- **Obtain consent** from any person whose likeness you use
- **Never use for impersonation**, fraud, misinformation, or non-consensual content
- **Label outputs as AI-generated** when sharing publicly
- **Comply with local regulations** — the EU AI Act classifies real-time deepfakes as high-risk; many jurisdictions require disclosure of synthetic media

This project is intended for legitimate use cases: media production, accessibility, entertainment, and research.

## Input Requirements

### Photo

| Property | Requirement | Recommended |
|----------|-------------|-------------|
| Format | JPG, PNG | JPG |
| Resolution | Minimum 256×256 | 512×512 or higher |
| Orientation | Front-facing, looking at camera | Slight angle OK |
| Face coverage | Face clearly visible, not occluded | >30% of frame |
| Lighting | Even, no harsh shadows | Soft diffused light |
| Expression | Neutral or slight smile | Neutral |
| Background | Any (cropped out by default) | Clean/simple |

**What doesn't work well:**
- Profile shots or extreme angles
- Sunglasses or face masks
- Multiple faces (only one will be animated)
- Very low resolution (<128×128)

### Audio

| Property | Requirement | Recommended |
|----------|-------------|-------------|
| Format | WAV, MP3, M4A, OGG, FLAC | WAV (16-bit PCM) |
| Sample rate | Any (resampled internally) | 44.1 kHz or 16 kHz |
| Channels | Mono or stereo | Mono |
| Duration | Up to ~3 minutes | 10–60 seconds |
| Content | Speech | Clear single speaker |

**What doesn't work well:**
- Music-only (no speech to sync to)
- Heavy background noise
- Multiple overlapping speakers
- Very long clips (>3 min may timeout or produce artifacts)

### Output

| Property | Value |
|----------|-------|
| Format | MP4 (H.264 + AAC) |
| Resolution | 256×256 (crop mode), or source resolution (full mode) |
| Frame rate | 25 fps |
| Duration | Matches input audio length |

## Tips for Best Results

**Photo:**
- Front-facing, looking at camera
- Good lighting, no harsh shadows
- High resolution (512×512+ ideal)
- Neutral expression works best

**Audio:**
- Clear speech, minimal background noise
- WAV format preferred (MP3 also works)
- Up to ~2–3 minutes per clip (split longer audio into segments)

**Settings for different looks:**
- *News anchor:* Still Mode ON, Expression Scale 1.0, Preprocess Crop
- *Animated host:* Still Mode OFF, Expression Scale 1.3, Pose Style 1–5
- *Quick preview:* Enhancer None, Preprocess Crop

## Cost

| Resource | Cost | Notes |
|----------|------|-------|
| SageMaker endpoint (ml.g5.xlarge) | ~$1.41/hr | **Runs continuously while deployed** |
| S3 storage | < $0.01 | Negligible for demo use |
| CodeBuild (initial build) | ~$0.10 | One-time |

⚠️ **Remember to run `./scripts/cleanup.sh` when done to stop charges.**

## Project Structure

```
SadTalker-Radio-Host/
├── container/
│   ├── Dockerfile           # Full image: CUDA + SadTalker + checkpoints
│   ├── Dockerfile.update    # Incremental update (layers on existing ECR image)
│   ├── serve.py             # Flask inference server (SageMaker-compatible)
│   └── buildspec.yml        # AWS CodeBuild spec
├── webapp/
│   ├── app.py               # Local Flask UI
│   └── requirements.txt
├── scripts/
│   ├── deploy.sh            # One-command deployment
│   └── cleanup.sh           # Tear down all resources
├── .gitignore
└── README.md
```

## Customization

**Change region or instance type:** Edit variables at the top of `scripts/deploy.sh` and `webapp/app.py`.

**Use a different model:** Replace the SadTalker setup in `container/Dockerfile` with Wav2Lip (better lip sync, no head motion) or MuseTalk (real-time, higher fidelity).

**Add an API:** The SageMaker endpoint accepts direct `invoke_endpoint_async` calls — integrate from any application without the web UI.

## How It Works

1. User uploads photo + audio via the web UI
2. Files are stored in S3 under `input/{job_id}/`
3. A JSON payload with S3 URIs and generation parameters is sent to the SageMaker async endpoint
4. The container downloads inputs, runs SadTalker inference on GPU, uploads the result MP4 to S3
5. The web UI polls S3 for the output and streams the video when ready

## Security Notes

This is a **demo project** with intentionally simplified permissions:

- **IAM roles use broad policies** (`AmazonSageMakerFullAccess`, `AmazonS3FullAccess`). For production, scope these down to the specific bucket and endpoint ARNs.
- **The web UI has no authentication.** It binds to `0.0.0.0:5000` — anyone on your local network can access it. For shared environments, bind to `127.0.0.1` only or add auth.
- **S3 bucket is private** but accessible to anyone with the IAM role. No bucket policy or encryption is configured beyond defaults.

For production hardening: use least-privilege IAM policies, enable S3 encryption, add API Gateway + Cognito in front of the endpoint, and restrict network access.

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `deploy.sh` fails at Docker build | Docker Hub rate limit (429 Too Many Requests) | Wait 1 hour and retry, or configure Docker Hub credentials in CodeBuild |
| Endpoint stuck in "Creating" | Service quota for `ml.g5.xlarge` is 0 | Request a quota increase via AWS Console → Service Quotas → SageMaker |
| Endpoint fails with "CapacityError" | No g5.xlarge capacity in region | Try a different region or instance type (g4dn.xlarge as fallback) |
| Video generation returns error | Audio is silent or corrupt | Check audio with `ffmpeg -i file.wav -af volumedetect -f null /dev/null` — mean volume should be > -50 dB |
| Output video has artifacts | Extreme head pose in photo | Use a more front-facing photo; try Preprocess = "full" |
| UI shows "Processing..." forever | Endpoint scaled down or crashed | Check endpoint status: `aws sagemaker describe-endpoint --endpoint-name sadtalker-async` |
| Build succeeds but endpoint fails to start | Container too large for instance | Ensure you're using at least `ml.g5.xlarge` (24 GB GPU memory) |
| `cleanup.sh` fails on S3 bucket | Bucket has versioning enabled | Run `aws s3 rb s3://BUCKET --force` manually |

### Checking Logs

```bash
# SageMaker container logs
aws logs get-log-events \
  --log-group-name /aws/sagemaker/Endpoints/sadtalker-async \
  --log-stream-name $(aws logs describe-log-streams --log-group-name /aws/sagemaker/Endpoints/sadtalker-async --query 'logStreams[-1].logStreamName' --output text) \
  --query 'events[].message' --output text

# CodeBuild logs
aws logs get-log-events \
  --log-group-name /aws/codebuild/sadtalker-build \
  --log-stream-name $(aws logs describe-log-streams --log-group-name /aws/codebuild/sadtalker-build --query 'logStreams[-1].logStreamName' --output text) \
  --query 'events[].message' --output text
```

## Credits

- [SadTalker](https://github.com/OpenTalker/SadTalker) — CVPR 2023, Apache 2.0 License
- [GFPGAN](https://github.com/TencentARC/GFPGAN) — Face restoration
- [AWS Media Localization Sample](https://github.com/aws-samples/media-localization-with-visual-dubbing-lip-sync) — Architecture inspiration

## License

MIT — see [LICENSE](LICENSE).

This project depends on third-party components with their own licenses (all permissive).
See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for details.

The Docker container uses NVIDIA CUDA base images subject to the
[NVIDIA Deep Learning Container License](https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license).
