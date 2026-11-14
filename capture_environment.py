import json
import platform
from pathlib import Path

import torch

info = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "pytorch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "cpu_threads": torch.get_num_threads(),
}
Path("outputs").mkdir(exist_ok=True)
Path("outputs/environment.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
print(json.dumps(info, indent=2))
