import os
import sys
import torch

# add repo path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dfine_root = os.path.join(repo_root, 'D-FINE-master')
if dfine_root not in sys.path:
    sys.path.insert(0, dfine_root)

from src.core import YAMLConfig

def main(config, resume, output):
    cfg = YAMLConfig(config, resume=resume)
    if "HGNetv2" in getattr(cfg, "yaml_cfg", {}):
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = torch.load(resume, map_location='cpu')
    if "ema" in checkpoint:
        state = checkpoint['ema'].get('module', checkpoint['ema'])
    elif 'model' in checkpoint:
        state = checkpoint['model']
    else:
        state = checkpoint

    try:
        cfg.model.load_state_dict(state)
    except Exception:
        cfg.model.load_state_dict(state, strict=False)

    model = cfg.model.deploy()
    model.eval()

    dummy = torch.randn(1,3,640,640)
    # run once
    with torch.no_grad():
        _ = model(dummy)

    print('Exporting model-only to ONNX ->', output)
    torch.onnx.export(
        model,
        dummy,
        output,
        input_names=['images'],
        output_names=['outputs'],
        opset_version=16,
        dynamic_axes={'images': {0: 'N'}},
        do_constant_folding=True,
        verbose=False,
    )
    print('Export complete.')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    main(args.config, args.resume, args.output)
