#!/usr/bin/env python3
"""
macOS-ready converter for D-FINE deployable model -> Core ML .mlpackage

Usage (on macOS with correct environment):
  python3 scripts/convert_dfine_to_coreml_macos.py \
    --config D-FINE-master/configs/dfine/objects365/dfine_hgnetv2_x_obj2coco.yml \
    --resume weights/dfine_x_obj2coco.pth \
    --output weights/dfine_x_obj2coco_mac.mlpackage

Notes:
- Run this on macOS (Darwin). coremltools requires macOS native libraries (libcoremlpython).
- This script traces the deployable model (cfg.model.deploy()) and converts the traced TorchScript model to an ML Program (.mlpackage).
- If your model has Python-only postprocessing (NMS, custom logic), those steps are NOT included; perform them outside Core ML or reimplement as Core ML layers.
"""
import os
import sys
import platform
import argparse

if platform.system() != 'Darwin':
    sys.stderr.write('WARNING: This script is intended to run on macOS (Darwin). coremltools native libs require macOS.\n')

import torch

try:
    import coremltools as ct
except Exception as e:
    raise RuntimeError('coremltools is required. Install on macOS with: pip install coremltools') from e


def main(config, resume, output):
    # make D-FINE-master importable
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dfine_root = os.path.join(repo_root, 'D-FINE-master')
    if dfine_root not in sys.path:
        sys.path.insert(0, dfine_root)

    from src.core import YAMLConfig

    # load config
    cfg = YAMLConfig(config, resume=resume)
    if 'HGNetv2' in getattr(cfg, 'yaml_cfg', {}):
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    # load checkpoint
    if not os.path.isfile(resume):
        raise FileNotFoundError(f'Checkpoint not found: {resume}')

    checkpoint = torch.load(resume, map_location='cpu')
    if 'ema' in checkpoint:
        state = checkpoint['ema'].get('module', checkpoint['ema'])
    elif 'model' in checkpoint:
        state = checkpoint['model']
    else:
        state = checkpoint

    # load weights
    try:
        cfg.model.load_state_dict(state)
    except Exception as e:
        print('Warning: strict load failed, trying non-strict:', e)
        cfg.model.load_state_dict(state, strict=False)

    # build deploy model
    model = cfg.model.deploy()
    model.eval()

    # dummy input (modify if your deployment image size is different)
    dummy = torch.randn(1, 3, 640, 640)

    # run once to initialize
    with torch.no_grad():
        _ = model(dummy)

    # trace to TorchScript
    print('Tracing model with torch.jit.trace (this may take a while)...')
    traced = torch.jit.trace(model, dummy, strict=False)
    print('Trace complete.')

    # Convert with coremltools
    print('Converting traced model to Core ML (.mlpackage) using coremltools...')

    # Prefer ImageType if you want built-in preprocessing (scale/bias) - otherwise use TensorType
    image_input = ct.ImageType(name='image', shape=dummy.shape, scale=1.0/255.0, color_layout=ct.colorlayout.RGB)

    mlmodel = ct.convert(
        traced,
        inputs=[image_input],
        convert_to='mlprogram',
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS15,
    )

    print(f'Saving Core ML package to: {output}')
    mlmodel.save(output)
    print('Saved. .mlpackage ready for Xcode import.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    main(args.config, args.resume, args.output)
