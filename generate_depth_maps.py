import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from unidepth.models import UniDepthV2


INPUT_DIR = Path("assets/test")
OUTPUT_DIR = Path("assets/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("models/unidepth-v2-vitl14")

USE_LOCAL_MODEL = True

if USE_LOCAL_MODEL:
    model = UniDepthV2.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )
else:
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

image_extensions = {".jpg", ".jpeg", ".png"}
image_paths = [
    image_path
    for image_path in sorted(INPUT_DIR.iterdir())
    if image_path.suffix.lower() in image_extensions
]

for image_path in tqdm(image_paths, desc="Generating depth maps", unit="image"):
    rgb = torch.from_numpy(
        np.array(Image.open(image_path).convert("RGB"))
    ).permute(2, 0, 1)

    predictions = model.infer(rgb)

    depth = predictions["depth"].squeeze().cpu().numpy()
    intrinsics = predictions["intrinsics"].squeeze().cpu().numpy()

    output_path = OUTPUT_DIR / f"{image_path.stem}.npz"

    np.savez(
        output_path,
        depth=depth,
        intrinsics=intrinsics,
    )

    tqdm.write(
        f"Saved: {output_path} | depth: {depth.shape} | "
        f"intrinsics: {intrinsics.shape}"
    )