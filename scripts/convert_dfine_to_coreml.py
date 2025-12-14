import os
import sys
import argparse

import torch

try:
    import coremltools as ct
except Exception as e:
    raise RuntimeError("coremltools is required. Install with: pip install coremltools") from e


def main(config, resume, output):
    # add repo root so D-FINE-master is importable
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dfine_root = os.path.join(repo_root, "D-FINE-master")
    if dfine_root not in sys.path:
        sys.path.insert(0, dfine_root)

    from src.core import YAMLConfig

    # load config
    cfg = YAMLConfig(config, resume=resume)

    # prevent loading external pretrained weights
    if "HGNetv2" in getattr(cfg, "yaml_cfg", {}):
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    # load checkpoint state dict
    if not os.path.isfile(resume):
        raise FileNotFoundError(f"Checkpoint not found: {resume}")

    checkpoint = torch.load(resume, map_location='cpu')
    if "ema" in checkpoint:
        state = checkpoint['ema'].get('module', checkpoint['ema'])
    elif 'model' in checkpoint:
        state = checkpoint['model']
    else:
        state = checkpoint

    # load into cfg.model
    try:
        cfg.model.load_state_dict(state)
    except Exception as e:
        print('Warning: failed to load state dict into cfg.model directly:', e)
        # try more relaxed loading
        try:
            cfg.model.load_state_dict(state, strict=False)
        except Exception as e2:
            raise RuntimeError('Failed to load checkpoint state_dict: ' + str(e2))

    # build deployable model wrapper (same as export_onnx.py)
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            # Note: Do NOT include the Python postprocessor in the traced graph.
            # The postprocessor may contain Python-only logic that cannot be converted.

        def forward(self, images):
            outputs = self.model(images)
            return outputs

    model = Model()
    model.eval()

    # dummy inputs (only images; postprocessing excluded)
    images = torch.rand(1, 3, 640, 640)

    # run once
    with torch.no_grad():
        _ = model(images)

    # trace
    print('Tracing model...')
    traced = torch.jit.trace(model, images, strict=False)
    print('Tracing complete.')

    # convert to Core ML
    print('Converting to Core ML...')
    image_input = ct.TensorType(name='images', shape=images.shape)

    mlmodel = ct.convert(
        traced,
        inputs=[image_input],
        convert_to='mlprogram',
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS15,
    )

    print('Saving Core ML package to', output)
    mlmodel.save(output)
    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    main(args.config, args.resume, args.output)
