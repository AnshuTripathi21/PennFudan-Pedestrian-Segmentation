import random
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

DATA_URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"


def download_and_prepare(data_dir="data"):
    """Download and extract Penn-Fudan only when the dataset is not present."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = data_dir / "PennFudanPed"

    if (dataset_root / "PNGImages").is_dir() and (dataset_root / "PedMasks").is_dir():
        return dataset_root

    zip_path = data_dir / "PennFudanPed.zip"
    if not zip_path.exists():
        print("Downloading Penn-Fudan dataset...")
        request = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=90) as response, open(zip_path, "wb") as out:
            total = response.headers.get("Content-Length")
            if total:
                print(f"Dataset archive size: {int(total) / (1024 * 1024):.1f} MB")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(data_dir)

    # Some archives may contain an extra top-level directory.
    candidates = list(data_dir.rglob("PNGImages"))
    for image_dir in candidates:
        root = image_dir.parent
        if (root / "PedMasks").is_dir():
            return root

    raise FileNotFoundError(
        "Could not find PNGImages and PedMasks after extraction. "
        "Check the downloaded Penn-Fudan ZIP file."
    )


def make_split(image_paths, val_fraction=0.20, seed=42, split_file="outputs/split.json"):
    """Create a deterministic image-level train/validation split and save it."""
    import json

    split_file = Path(split_file)
    split_file.parent.mkdir(parents=True, exist_ok=True)

    names = [p.name for p in image_paths]
    if split_file.exists():
        saved = json.loads(split_file.read_text(encoding="utf-8"))
        if set(saved.get("all_images", [])) == set(names):
            train_names = set(saved["train"])
            val_names = set(saved["validation"])
            train = [p for p in image_paths if p.name in train_names]
            val = [p for p in image_paths if p.name in val_names]
            if train and val and len(train) + len(val) == len(image_paths):
                return train, val

    shuffled = names[:]
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    val_names = set(shuffled[:val_count])
    train = [p for p in image_paths if p.name not in val_names]
    val = [p for p in image_paths if p.name in val_names]

    split_file.write_text(json.dumps({
        "seed": seed,
        "validation_fraction": val_fraction,
        "all_images": names,
        "train": [p.name for p in train],
        "validation": [p.name for p in val],
    }, indent=2), encoding="utf-8")
    return train, val


class PennFudanBinaryDataset(Dataset):
    """
    Binary semantic segmentation dataset:
      0 = background
      1 = pedestrian

    Penn-Fudan masks encode separate pedestrian instances with non-zero
    pixel values. All non-zero pixels are merged into the pedestrian class.
    """
    def __init__(self, image_paths, image_size=128, augment=False):
        self.image_paths = list(image_paths)
        self.image_size = int(image_size)
        self.augment = bool(augment)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = image_path.parent.parent / "PedMasks" / (
            image_path.stem + "_mask.png"
        )

        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for {image_path.name}: {mask_path}")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32)

        if self.augment and random.random() < 0.5:
            image_array = np.ascontiguousarray(image_array[:, ::-1, :])
            mask_array = np.ascontiguousarray(mask_array[:, ::-1])

        # ImageNet normalization; masks remain binary.
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_array = (image_array - mean) / std

        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask_array[None, :, :]).float()
        return image_tensor, mask_tensor, image_path.name
