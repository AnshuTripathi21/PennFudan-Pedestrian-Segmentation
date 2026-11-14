# Penn-Fudan Binary Semantic Segmentation: U-Net vs Dilated U-Net

This project runs a small, reproducible semantic-segmentation comparison on the
Penn-Fudan Pedestrian dataset.

## Dataset
The code downloads the official Penn-Fudan ZIP archive automatically:
https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip

Expected layout:
```
data/PennFudanPed/
    PNGImages/
    PedMasks/
```

The masks are instance masks: non-zero pixel values identify pedestrian
instances. This project merges all non-zero values into one foreground class:
0 = background, 1 = pedestrian.

## Experiment
- Baseline: compact U-Net, bottleneck dilation 1.
- Controlled variant: same U-Net, first bottleneck 3x3 convolution dilation 2.
- Dataset split: deterministic 80% train / 20% validation, seed 42.
- Input size: 128 x 128.
- Augmentation: random horizontal flip on training images only.
- Loss: BCEWithLogitsLoss.
- Optimizer: Adam, learning rate 0.001.
- Batch size: 4.
- Maximum epochs: 12.
- Early stopping: 4 epochs without improved validation loss.
- Evaluation: background IoU, pedestrian IoU, mean of available class IoUs,
  parameter count and median batch-1 latency.

## Install
Use Python 3.10 or 3.11 if possible.

Windows PowerShell:
```
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.11` is not available, use:
```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run
Capture your environment:
```
python capture_environment.py
```

Run both models:
```
python run_experiment.py
```

The first run downloads and extracts the dataset. Internet access is needed
only for the initial download.

## Run one model manually
```
python train.py --variant baseline
python evaluate.py --variant baseline
python train.py --variant dilated
python evaluate.py --variant dilated
```

## If the computer is slow or runs out of memory
Use the same reduced settings for both variants:
```
python train.py --variant baseline --epochs 8 --batch-size 2 --image-size 96
python evaluate.py --variant baseline --image-size 96
python train.py --variant dilated --epochs 8 --batch-size 2 --image-size 96
python evaluate.py --variant dilated --image-size 96
```
If you use these reduced settings, make the same change for both variants and
report the settings actually used. Do not compare runs trained with different
settings.

## Outputs
- `outputs/split.json`: exact deterministic train/validation split.
- `outputs/*_training.csv`: epoch training and validation losses.
- `outputs/*_best.pt`: best checkpoint selected by validation loss.
- `outputs/*_config.json`: run settings.
- `outputs/*_metrics.json`: IoU, mIoU, parameter count and latency.
- `outputs/*_per_class_iou.csv`: class-wise IoU table.
- `outputs/*_predictions/`: input, ground-truth and prediction comparison images.
- `outputs/environment.json`: Python, PyTorch and device information.

## Important limitations
Penn-Fudan is a small pedestrian dataset. Its validation set is also small,
so metrics can vary noticeably with the split and seed. A single run does not
prove that one architecture is universally better. Latency depends on device,
software, image size and system load. This project evaluates binary semantic
segmentation after merging instance IDs; it is not instance segmentation.
