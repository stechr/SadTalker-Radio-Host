"""SageMaker-compatible inference server for SadTalker.

Exposes /ping (health check) and /invocations (inference) endpoints
per the SageMaker hosting contract.
"""

import json
import logging
import os
import subprocess
import tempfile
from urllib.parse import urlparse

import boto3
from flask import Flask, Response, jsonify, request

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

s3 = boto3.client("s3")
SADTALKER_DIR = "/opt/sadtalker"
CHECKPOINT_DIR = os.path.join(SADTALKER_DIR, "checkpoints")

# Valid parameter values
VALID_ENHANCERS = {"gfpgan", "RestoreFormer", "none"}
VALID_PREPROCESS = {"crop", "resize", "full"}


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse s3://bucket/key into (bucket, key)."""
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


@app.route("/ping", methods=["GET"])
def ping() -> Response:
    """Health check — returns 200 if the container is ready."""
    return Response(status=200)


@app.route("/invocations", methods=["POST"])
def invoke() -> tuple[Response, int]:
    """Run SadTalker inference on uploaded image + audio."""
    try:
        payload = json.loads(request.data)
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Invalid JSON: {e}"}), 400

    # Validate required fields
    for field in ("image_s3_uri", "audio_s3_uri", "output_s3_uri"):
        if field not in payload:
            return jsonify({"error": f"Missing required field: {field}"}), 400

    # Extract and validate parameters
    enhancer = payload.get("enhancer", "gfpgan")
    if enhancer not in VALID_ENHANCERS:
        return jsonify({"error": f"Invalid enhancer: {enhancer}. Must be one of {VALID_ENHANCERS}"}), 400

    preprocess = payload.get("preprocess", "crop")
    if preprocess not in VALID_PREPROCESS:
        return jsonify({"error": f"Invalid preprocess: {preprocess}. Must be one of {VALID_PREPROCESS}"}), 400

    still_mode = payload.get("still_mode", True)
    expression_scale = float(payload.get("expression_scale", 1.0))
    pose_style = int(payload.get("pose_style", 0))

    if not (0.0 <= expression_scale <= 3.0):
        return jsonify({"error": "expression_scale must be between 0.0 and 3.0"}), 400
    if not (0 <= pose_style <= 45):
        return jsonify({"error": "pose_style must be between 0 and 45"}), 400

    logger.info("Starting inference: enhancer=%s, still=%s, preprocess=%s, expression=%.1f, pose=%d",
                enhancer, still_mode, preprocess, expression_scale, pose_style)

    with tempfile.TemporaryDirectory() as tmp:
        img_path = os.path.join(tmp, "input.jpg")
        aud_path = os.path.join(tmp, "input.wav")
        result_dir = os.path.join(tmp, "results")

        # Download inputs from S3
        try:
            bucket, key = parse_s3_uri(payload["image_s3_uri"])
            s3.download_file(bucket, key, img_path)
            bucket, key = parse_s3_uri(payload["audio_s3_uri"])
            s3.download_file(bucket, key, aud_path)
        except Exception as e:
            logger.error("Failed to download inputs: %s", e)
            return jsonify({"error": f"Failed to download inputs: {e}"}), 500

        # Build inference command
        cmd = [
            "python", "inference.py",
            "--driven_audio", aud_path,
            "--source_image", img_path,
            "--result_dir", result_dir,
            "--checkpoint_dir", CHECKPOINT_DIR,
            "--preprocess", preprocess,
            "--expression_scale", str(expression_scale),
            "--pose_style", str(pose_style),
        ]

        if still_mode:
            cmd.append("--still")

        if enhancer != "none":
            cmd.extend(["--enhancer", enhancer])

        # Run SadTalker
        proc = subprocess.run(cmd, cwd=SADTALKER_DIR, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            logger.error("SadTalker failed: %s", proc.stderr[-2000:])
            return jsonify({"error": f"Inference failed: {proc.stderr[-500:]}"}), 500

        # Find output video
        mp4_files = []
        for root, _, files in os.walk(result_dir):
            mp4_files.extend(os.path.join(root, f) for f in files if f.endswith(".mp4"))

        if not mp4_files:
            logger.error("No output video found. stdout: %s", proc.stdout[-1000:])
            return jsonify({"error": "No output video generated"}), 500

        # Upload result to S3
        try:
            bucket, key = parse_s3_uri(payload["output_s3_uri"])
            s3.upload_file(mp4_files[0], bucket, key, ExtraArgs={"ContentType": "video/mp4"})
        except Exception as e:
            logger.error("Failed to upload result: %s", e)
            return jsonify({"error": f"Failed to upload result: {e}"}), 500

    logger.info("Inference complete: %s", payload["output_s3_uri"])
    return jsonify({"status": "success", "output_s3_uri": payload["output_s3_uri"]}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
