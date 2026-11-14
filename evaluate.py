import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from dataset import download_and_prepare, make_split, PennFudanBinaryDataset
from model import SmallUNet, count_parameters


def save_prediction_panel(image_tensor, mask_tensor, prediction, output_path):
    """Save original image, ground truth and predicted binary mask side by side."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = (image_tensor.cpu() * std + mean).clamp(0, 1)
    image_array = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    image_pil = Image.fromarray(image_array).convert("RGB")

    gt_array = (mask_tensor.squeeze(0).cpu().numpy() > 0.5).astype(np.uint8) * 255
    pred_array = (prediction.squeeze(0).cpu().numpy() > 0.5).astype(np.uint8) * 255
    gt_pil = Image.fromarray(gt_array).convert("RGB")
    pred_pil = Image.fromarray(pred_array).convert("RGB")

    width, height = image_pil.size
    panel = Image.new("RGB", (width * 3, height + 28), "white")
    panel.paste(image_pil, (0, 28))
    panel.paste(gt_pil, (width, 28))
    panel.paste(pred_pil, (width * 2, 28))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 7), "Input image", fill="black")
    draw.text((width + 8, 7), "Ground truth", fill="black")
    draw.text((width * 2 + 8, 7), "Prediction", fill="black")
    panel.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["baseline", "dilated"], required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--latency-runs", type=int, default=20)
    parser.add_argument("--save-predictions", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    prediction_dir = output_dir / f"{args.variant}_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_root = download_and_prepare(args.data_dir)
    image_paths = sorted((dataset_root / "PNGImages").glob("*.png"))
    _, val_paths = make_split(image_paths)
    val_ds = PennFudanBinaryDataset(val_paths, args.image_size, augment=False)
    loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda")
    )

    dilation = 2 if args.variant == "dilated" else 1
    model = SmallUNet(bottleneck_dilation=dilation).to(device)
    checkpoint_path = output_dir / f"{args.variant}_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train this variant first."
        )

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Confusion counts: rows = ground truth, columns = prediction.
    confusion = np.zeros((2, 2), dtype=np.int64)
    saved = 0
    with torch.no_grad():
        for images, masks, names in loader:
            images = images.to(device)
            masks = masks.to(device)
            probabilities = torch.sigmoid(model(images))
            predictions = (probabilities >= 0.5).long()
            targets = (masks >= 0.5).long()

            target_np = targets.cpu().numpy().reshape(-1)
            pred_np = predictions.cpu().numpy().reshape(-1)
            encoded = target_np * 2 + pred_np
            confusion += np.bincount(encoded, minlength=4).reshape(2, 2)

            if saved < args.save_predictions:
                for i in range(images.size(0)):
                    if saved >= args.save_predictions:
                        break
                    filename = Path(names[i]).stem + "_comparison.png"
                    save_prediction_panel(
                        images[i].cpu(), masks[i].cpu(), probabilities[i].cpu(),
                        prediction_dir / filename
                    )
                    saved += 1

    ious = []
    class_results = []
    class_names = ["background", "pedestrian"]
    for class_id, class_name in enumerate(class_names):
        tp = int(confusion[class_id, class_id])
        fp = int(confusion[:, class_id].sum() - tp)
        fn = int(confusion[class_id, :].sum() - tp)
        union = tp + fp + fn
        iou = float(tp / union) if union > 0 else None
        if iou is not None:
            ious.append(iou)
        class_results.append({
            "class_id": class_id,
            "class_name": class_name,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "iou": iou
        })
    miou = float(np.mean(ious)) if ious else 0.0

    parameter_count = count_parameters(model)
    sample = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = []
        for _ in range(args.latency_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms.append((time.perf_counter() - start) * 1000.0)

    result = {
        "variant": args.variant,
        "dataset": "Penn-Fudan validation split",
        "validation_images": len(val_ds),
        "image_size": args.image_size,
        "device": str(device),
        "parameter_count": parameter_count,
        "confusion_matrix_rows_gt_columns_prediction": confusion.tolist(),
        "per_class_iou": class_results,
        "mIoU": miou,
        "pedestrian_IoU": class_results[1]["iou"],
        "median_batch1_latency_ms": float(np.median(elapsed_ms)),
        "latency_warmup_runs": args.warmup,
        "latency_measurement_runs": args.latency_runs,
        "best_epoch": checkpoint.get("epoch"),
        "best_validation_loss": checkpoint.get("validation_loss"),
    }

    (output_dir / f"{args.variant}_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    with open(output_dir / f"{args.variant}_per_class_iou.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=class_results[0].keys())
        writer.writeheader()
        writer.writerows(class_results)

    print(json.dumps(result, indent=2))
    print(f"Prediction comparisons saved in: {prediction_dir}")


if __name__ == "__main__":
    main()
