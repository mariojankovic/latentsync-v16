"""
RunPod Serverless Handler for LatentSync 1.6 (512x512)
Simple: video_url + audio_url → lip-synced video uploaded to R2
"""

import runpod
import os
import uuid
import time
import logging
import shutil
import subprocess
import tempfile
import base64
import torch
import gc
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("latentsync16")

# ── R2 / S3 Config ──────────────────────────────────────────────
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "tray")

s3_client = None
if R2_ACCOUNT_ID and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY:
    import boto3
    from botocore.config import Config
    s3_client = boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )
    logger.info(f"R2 client configured: bucket={R2_BUCKET_NAME}")
else:
    logger.warning("R2 not configured — will return base64 output")

# ── LatentSync Pipeline ─────────────────────────────────────────
PIPE = None
CONFIG = None
DTYPE = None
DEVICE = "cuda"

BASE_DIR = Path("/app")
LATENTSYNC_DIR = BASE_DIR / "LatentSync"
CONFIG_PATH = LATENTSYNC_DIR / "configs" / "unet" / "stage2_512.yaml"
SCHEDULER_DIR = LATENTSYNC_DIR / "configs"
UNET_CKPT = BASE_DIR / "checkpoints" / "latentsync_unet.pt"
WHISPER_TINY = BASE_DIR / "checkpoints" / "whisper" / "whisper" / "tiny.pt"
MASK_PATH = LATENTSYNC_DIR / "latentsync" / "utils" / "mask.png"


def load_pipe():
    global PIPE, CONFIG, DTYPE
    if PIPE is not None:
        return PIPE

    from omegaconf import OmegaConf
    from diffusers import AutoencoderKL, DDIMScheduler
    from latentsync.models.unet import UNet3DConditionModel
    from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
    from latentsync.whisper.audio2feature import Audio2Feature
    from DeepCache import DeepCacheSDHelper

    logger.info("Loading LatentSync 1.6 pipeline...")
    t0 = time.time()

    CONFIG = OmegaConf.load(str(CONFIG_PATH))
    is_fp16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    DTYPE = torch.float16 if is_fp16 else torch.float32

    scheduler = DDIMScheduler.from_pretrained(str(SCHEDULER_DIR))

    whisper_path = WHISPER_TINY
    audio_encoder = Audio2Feature(
        model_path=str(whisper_path),
        device=DEVICE,
        num_frames=CONFIG.data.num_frames,
        audio_feat_length=CONFIG.data.audio_feat_length,
    )

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", torch_dtype=DTYPE
    ).to(DEVICE)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(CONFIG.model), str(UNET_CKPT), device="cpu"
    )
    unet = unet.to(dtype=DTYPE)

    CONFIG.data.mask_image_path = str(MASK_PATH)

    PIPE = LipsyncPipeline(
        vae=vae, audio_encoder=audio_encoder, unet=unet, scheduler=scheduler
    ).to(DEVICE)

    helper = DeepCacheSDHelper(pipe=PIPE)
    helper.set_params(cache_interval=3, cache_branch_id=0)
    helper.enable()

    logger.info(f"Pipeline loaded in {time.time()-t0:.1f}s")
    return PIPE


def download_url(url, dest):
    """Download a URL to a local file."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)
    return os.path.getsize(dest)


def preprocess_video(input_path, output_path):
    """Resample to 25 FPS, h264, no audio."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-r", "25",
         "-c:v", "libx264", "-crf", "18", "-preset", "medium",
         "-pix_fmt", "yuv420p", "-an", output_path],
        capture_output=True, check=True,
    )


def preprocess_audio(input_path, output_path):
    """Resample to 16kHz mono PCM."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path,
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", output_path],
        capture_output=True, check=True,
    )


def handler(event):
    workdir = None
    try:
        inp = event["input"]
        video_url = inp["video_url"]
        audio_url = inp["audio_url"]
        seed = inp.get("seed", 1247)
        return_base64 = inp.get("return_base64", False)

        job_id = str(uuid.uuid4())[:8]
        workdir = Path(tempfile.mkdtemp(prefix=f"ls16_{job_id}_"))

        # ── Download ────────────────────────────────────────────
        logger.info(f"[{job_id}] Downloading video...")
        raw_video = workdir / "raw_video.mp4"
        vsize = download_url(video_url, str(raw_video))
        logger.info(f"[{job_id}] Video: {vsize/1024:.1f} KB")

        logger.info(f"[{job_id}] Downloading audio...")
        raw_audio = workdir / "raw_audio.wav"
        asize = download_url(audio_url, str(raw_audio))
        logger.info(f"[{job_id}] Audio: {asize/1024:.1f} KB")

        # ── Preprocess ──────────────────────────────────────────
        logger.info(f"[{job_id}] Preprocessing video (25fps)...")
        prep_video = workdir / "prep_video.mp4"
        preprocess_video(str(raw_video), str(prep_video))

        logger.info(f"[{job_id}] Preprocessing audio (16kHz mono)...")
        prep_audio = workdir / "prep_audio.wav"
        preprocess_audio(str(raw_audio), str(prep_audio))

        # ── Lip-sync ────────────────────────────────────────────
        output_path = workdir / "output.mp4"
        temp_dir = workdir / "temp"
        temp_dir.mkdir()

        pipe = load_pipe()
        if seed != -1:
            torch.manual_seed(seed)

        logger.info(f"[{job_id}] Running LatentSync 1.6 (512x512)...")
        t0 = time.time()

        pipe(
            video_path=str(prep_video),
            audio_path=str(prep_audio),
            video_out_path=str(output_path),
            num_frames=CONFIG.data.num_frames,
            num_inference_steps=20,
            guidance_scale=1.5,
            weight_dtype=DTYPE,
            width=CONFIG.data.resolution,
            height=CONFIG.data.resolution,
            mask_image_path=CONFIG.data.mask_image_path,
            temp_dir=str(temp_dir),
        )

        inference_time = time.time() - t0
        output_size = output_path.stat().st_size
        logger.info(f"[{job_id}] Done in {inference_time:.1f}s ({output_size/1024:.1f} KB)")

        # ── Return result ───────────────────────────────────────
        result = {
            "job_id": job_id,
            "inference_time": round(inference_time, 1),
            "output_size_kb": round(output_size / 1024, 1),
        }

        if s3_client and not return_base64:
            # Upload to R2 and return presigned URL
            s3_key = f"latentsync/{job_id}.mp4"
            s3_client.upload_file(
                str(output_path), R2_BUCKET_NAME, s3_key,
                ExtraArgs={"ContentType": "video/mp4"},
            )
            presigned_url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": R2_BUCKET_NAME, "Key": s3_key},
                ExpiresIn=86400,  # 24 hours
            )
            result["video_url"] = presigned_url
            logger.info(f"[{job_id}] Uploaded to R2: {s3_key}")
        else:
            # Return base64 (fallback)
            with open(output_path, "rb") as f:
                result["video_base64"] = base64.b64encode(f.read()).decode()
            logger.info(f"[{job_id}] Returning base64 ({output_size/1024:.1f} KB)")

        return result

    except Exception as e:
        logger.exception("Handler failed")
        return {"error": str(e)}

    finally:
        if workdir and workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── Preload pipeline at startup ─────────────────────────────────
logger.info("Preloading pipeline...")
try:
    load_pipe()
except Exception as e:
    logger.error(f"Failed to preload: {e}")

runpod.serverless.start({"handler": handler})
