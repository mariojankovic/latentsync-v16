# Thin layer on top of working abdedk image
# Adds LatentSync 1.6 weights + 512x512 config
FROM abdedk/latentsync-runpod:7

# Download LatentSync 1.6 weights (overwrites 1.5)
RUN pip install -q huggingface-hub hf-transfer && \
    huggingface-cli download ByteDance/LatentSync-1.6 latentsync_unet.pt \
      --local-dir /app/LatentSync/checkpoints --local-dir-use-symlinks False

# GFPGAN for optional face enhancement
RUN pip install --no-cache-dir gfpgan && \
    mkdir -p /app/models/gfpgan && \
    wget -q -O /app/models/gfpgan/GFPGANv1.4.pth \
      https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth

# Fix basicsr/torchvision compatibility (functional_tensor removed in newer torchvision)
RUN python -c "\
import os;\
for pkg in ['basicsr', 'gfpgan']:\
    try:\
        mod = __import__(pkg);\
        root = mod.__path__[0];\
    except: continue;\
    for dp, _, fns in os.walk(root):\
        for fn in fns:\
            if not fn.endswith('.py'): continue;\
            fp = os.path.join(dp, fn);\
            txt = open(fp).read();\
            if 'functional_tensor' in txt:\
                open(fp, 'w').write(txt.replace('torchvision.transforms.functional_tensor', 'torchvision.transforms.functional'));\
                print(f'Patched: {fp}');\
"

# Copy handler
COPY rp_handler.py /app/rp_handler.py

CMD ["python", "-u", "/app/rp_handler.py"]
