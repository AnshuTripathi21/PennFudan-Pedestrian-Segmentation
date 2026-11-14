import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import download_and_prepare, make_split, PennFudanBinaryDataset
from model import SmallUNet, count_parameters


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_images = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, masks, _ in loader:
            images = images.to(device)
            masks = masks.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, masks)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_images += images.size(0)

    return total_loss / max(total_images, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["baseline", "dilated"], required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_root = download_and_prepare(args.data_dir)
    image_dir = dataset_root / "PNGImages"
    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise RuntimeError(f"No PNG images found in {image_dir}")

    for image_path in image_paths:
        expected_mask = dataset_root / "PedMasks" / f"{image_path.stem}_mask.png"
        if not expected_mask.exists():
            raise FileNotFoundError(f"Missing paired mask: {expected_mask}")

    train_paths, val_paths = make_split(image_paths, seed=args.seed)
    train_ds = PennFudanBinaryDataset(train_paths, args.image_size, augment=True)
    val_ds = PennFudanBinaryDataset(val_paths, args.image_size, augment=False)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=pin_memory
    )

    dilation = 2 if args.variant == "dilated" else 1
    model = SmallUNet(bottleneck_dilation=dilation).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    checkpoint_path = output_dir / f"{args.variant}_best.pt"
    log_path = output_dir / f"{args.variant}_training.csv"
    best_val_loss = float("inf")
    stale_epochs = 0

    with open(log_path, "w", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["epoch", "train_loss", "validation_loss", "learning_rate", "seconds"])

        for epoch in range(1, args.epochs + 1):
            start = time.perf_counter()
            train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
            val_loss = run_epoch(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            elapsed = time.perf_counter() - start
            current_lr = optimizer.param_groups[0]["lr"]

            writer.writerow([epoch, train_loss, val_loss, current_lr, elapsed])
            log_file.flush()

            print(
                f"{args.variant} | epoch {epoch:02d}/{args.epochs} | "
                f"train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
                f"lr {current_lr:.2e} | {elapsed:.1f}s"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                stale_epochs = 0
                torch.save({
                    "model_state": model.state_dict(),
                    "variant": args.variant,
                    "dilation": dilation,
                    "image_size": args.image_size,
                    "seed": args.seed,
                    "epoch": epoch,
                    "validation_loss": val_loss,
                    "parameter_count": count_parameters(model),
                }, checkpoint_path)
            else:
                stale_epochs += 1
                if stale_epochs >= args.patience:
                    print("Early stopping: validation loss did not improve.")
                    break

    config = vars(args)
    config.update({
        "dataset": "Penn-Fudan Pedestrian Segmentation",
        "train_images": len(train_ds),
        "validation_images": len(val_ds),
        "classes": {"0": "background", "1": "pedestrian"},
        "device": str(device),
        "dilation": dilation,
        "parameter_count": count_parameters(model),
    })
    (output_dir / f"{args.variant}_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(f"Saved best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
