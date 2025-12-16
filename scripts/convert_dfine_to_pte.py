#!/usr/bin/env python3
"""
Export DFINE checkpoint to ExecuTorch .pte file.

Usage:
  python scripts/convert_dfine_to_pte.py --weights weights/dfine_x_obj2coco.pth --output weights/dfine_x_obj2coco.pte --backend coreml

Notes:
  - Requires `executorch` Python package (pip install executorch) and compatible PyTorch (>=2.2 recommended).
  - Run on macOS/Linux x86_64 or macOS arm64 where ExecuTorch host libs are supported.
  - You may need to provide a D-FINE YAML config via --config if the script cannot infer one from D-FINE-master.
"""
import argparse
import os
import sys
import torch

# Add repo root to path so we can import dfine_count utilities
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import torchvision.models as models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.exir import to_edge_transform_and_lower

def _add_dfine_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dfine_root = os.path.join(repo_root, "D-FINE-master")
    if dfine_root not in sys.path:
        sys.path.insert(0, dfine_root)
    return dfine_root

class DFineIntegrationError(RuntimeError):
    pass

def _find_config(dfine_root):
    # Try to find a default yaml config in D-FINE-master/configs or configs
    cand_dirs = [os.path.join(dfine_root, "configs"), os.path.join(dfine_root, "config")]
    yamls = []
    for d in cand_dirs:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith(".yaml") or f.endswith(".yml"):
                    yamls.append(os.path.join(d, f))
    if len(yamls) == 1:
        return yamls[0]
    return None

def load_model(weights_path, config_path=None, device="cpu"):
    dfine_root = _add_dfine_path()

    # Import YAMLConfig from D-FINE
    try:
        from src.core import YAMLConfig
    except Exception as e:
        raise DFineIntegrationError(f"Failed to import D-FINE src.core.YAMLConfig: {e}")

    if config_path is None:
        config_path = _find_config(dfine_root)
        if config_path is None:
            raise DFineIntegrationError("No config provided and could not infer a config in D-FINE-master; please pass --config <path>")

    # Load checkpoint
    if not os.path.isfile(weights_path):
        raise DFineIntegrationError(f"Weights file not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"].get("module", checkpoint["ema"])
    elif "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint

    cfg = YAMLConfig(config_path, resume=weights_path)

    # In some configs, adjust pretrained flags
    if "HGNetv2" in getattr(cfg, "yaml_cfg", {}):
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    # Load state_dict
    try:
        cfg.model.load_state_dict(state)
    except Exception as e:
        raise DFineIntegrationError(f"Failed to load state_dict into cfg.model: {e}")

    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return outputs

    model = Model().to(device)
    model.eval()
    return model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=None, help="Path to DFINE checkpoint (.pth). If omitted, uses weights/dfine_x_obj2coco.pth in repository")
    p.add_argument("--config", default=None, help="Optional D-FINE YAML config path")
    p.add_argument("--output", required=True, help="Output .pte file path")
    p.add_argument("--backend", choices=["coreml", "xnnpack"], default="coreml", help="Target backend partitioner")
    p.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = p.parse_args()

    # If no config provided, use repository-local D-FINE config for objects365/dfine_hgnetv2_x_obj2coco
    if args.config is None:
        default_cfg = os.path.join(REPO_ROOT, "D-FINE-master", "configs", "dfine", "objects365", "dfine_hgnetv2_x_obj2coco.yml")
        if os.path.isfile(default_cfg):
            args.config = default_cfg
            print(f"Using default D-FINE config: {args.config}")
        else:
            print(f"Warning: default config not found at {default_cfg}; proceeding without explicit config", file=sys.stderr)

    # If no weights provided, use repo-local default weight file
    if args.weights is None:
        default_weights = os.path.join(REPO_ROOT, "weights", "dfine_x_obj2coco.pth")
        if os.path.isfile(default_weights):
            args.weights = default_weights
            print(f"Using default weights: {args.weights}")
        else:
            print(f"Error: default weights not found at {default_weights}; please pass --weights", file=sys.stderr)
            sys.exit(2)


    # Load DFine model (this builds a Model that expects (images, orig_target_sizes))
    try:
        model = load_model(args.weights, config_path=args.config, device=args.device)
    except DFineIntegrationError as e:
        print(f"Failed to load DFine model: {e}", file=sys.stderr)
        sys.exit(2)

    model.eval()

    # Create representative inputs. DFine inference in this repo resizes input to 640x640.
    sample_image = torch.randn(1, 3, 640, 640)
    sample_sizes = torch.tensor([[640.0, 640.0]])
    sample_inputs = (sample_image, sample_sizes)

    print("Lowering to ExecuTorch format and partitioning for backend:", args.backend)
    et_program = to_edge_transform_and_lower(
        torch.export.export(model, sample_inputs),
        partitioner=[CoreMLPartitioner()],
    ).to_executorch()

    with open(args.output, "wb") as file:
        et_program.write_to_file(file)

if __name__ == "__main__":
    main()
