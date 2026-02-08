# Thin layer on top of working abdedk image
# Adds LatentSync 1.6 weights + 512x512 config
FROM abdedk/latentsync-runpod:7

# Download LatentSync 1.6 weights (overwrites 1.5)
RUN pip install -q huggingface-hub hf-transfer && \
    huggingface-cli download ByteDance/LatentSync-1.6 latentsync_unet.pt \
      --local-dir /app/LatentSync/checkpoints --local-dir-use-symlinks False

# Copy handler
COPY rp_handler.py /app/rp_handler.py

CMD ["python", "-u", "/app/rp_handler.py"]
