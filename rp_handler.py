import os, uuid, base64, subprocess, shlex, time
import runpod
from runpod.serverless.utils import rp_download, rp_cleanup

import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

MODEL_DIR = "/app/LatentSync"
CKPT = os.path.join(MODEL_DIR, "checkpoints", "latentsync_unet.pt")
UNET_CFG = os.path.join(MODEL_DIR, "configs", "unet", "stage2_512.yaml")  # 1.6 (512x512)

# --- Cloudflare R2 (S3-compatible) ---
R2_ENDPOINT = os.getenv("BUCKET_ENDPOINT_URL")
R2_BUCKET   = os.getenv("BUCKET_NAME")
R2_KEY      = os.getenv("BUCKET_ACCESS_KEY_ID")
R2_SECRET   = os.getenv("BUCKET_SECRET_ACCESS_KEY")
EXPIRES_IN  = int(os.getenv("PRESIGN_EXPIRES", "86400"))

def _r2_client():
    if not (R2_ENDPOINT and R2_BUCKET and R2_KEY and R2_SECRET):
        return None
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_KEY,
        aws_secret_access_key=R2_SECRET,
        region_name="auto",
        config=BotoConfig(signature_version="s3v4", s3={"addressing_style": "virtual"})
    )

S3 = _r2_client()

def _normalize_path(obj):
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for k in ("file_path", "path", "download_path", "saved_to", "name"):
            v = obj.get(k)
            if isinstance(v, str) and v:
                return v
    raise TypeError(f"Unexpected download object: {type(obj)} -> {obj}")

def _upload_to_r2(file_path: str):
    if S3 is None:
        raise RuntimeError("R2 client not configured (check env vars).")
    key = f"outputs/{uuid.uuid4().hex}.mp4"
    try:
        S3.upload_file(
            Filename=file_path,
            Bucket=R2_BUCKET,
            Key=key,
            ExtraArgs={"ContentType": "video/mp4"}
        )
        url = S3.generate_presigned_url(
            "get_object",
            Params={"Bucket": R2_BUCKET, "Key": key},
            ExpiresIn=EXPIRES_IN
        )
        return {"video_url": url}
    except (BotoCoreError, ClientError) as e:
        raise RuntimeError(f"R2 upload failed: {e}")


def handler(job):
    inp = job["input"]
    video_url = inp["video_url"]
    audio_url = inp["audio_url"]

    steps    = int(inp.get("inference_steps", 20))
    guidance = float(inp.get("guidance_scale", 1.5))
    seed     = int(inp.get("seed", 1247))

    v_path = _normalize_path(rp_download.file(video_url))
    a_path = _normalize_path(rp_download.file(audio_url))

    out_path = f"/tmp/{uuid.uuid4().hex}.mp4"

    cmd = [
        "python", "-m", "scripts.inference",
        "--unet_config_path", UNET_CFG,
        "--inference_ckpt_path", CKPT,
        "--inference_steps", str(steps),
        "--guidance_scale", str(guidance),
        "--video_path", v_path,
        "--audio_path", a_path,
        "--video_out_path", out_path,
        "--seed", str(seed)
    ]

    proc = subprocess.run(cmd, cwd=MODEL_DIR, capture_output=True, text=True)

    if proc.returncode != 0:
        return {"error": "inference_failed", "stderr": proc.stderr[-2000:], "stdout": proc.stdout[-2000:]}

    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return {"error": "missing_output_file", "expected_path": out_path, "stderr": proc.stderr[-2000:]}

    try:
        result = _upload_to_r2(out_path)
    except Exception as e:
        return {"error": "upload_failed", "message": str(e)}

    rp_cleanup.clean()
    if proc.stdout:
        result["logs"] = proc.stdout[:2000]
    return result

runpod.serverless.start({"handler": handler})
