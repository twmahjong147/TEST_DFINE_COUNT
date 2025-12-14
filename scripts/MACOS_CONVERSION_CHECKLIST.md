macOS conversion checklist — D-FINE -> Core ML (.mlpackage)

Date: 2025-12-14T02:21:06Z

Recommended environment (tested / expected):

- Operating System: macOS 12+ (Monterey) or macOS 13+ (Ventura). Native Core ML libraries require macOS.
- Python: 3.10 (recommend using pyenv or venv)
- PyTorch: 2.0.x (the traced model uses torch.jit.trace; PyTorch 2.x is recommended)
- coremltools: 6.2.0 (or the latest 6.x release). If you prefer mlprogram features, coremltools 6+ is required.
- onnx (optional, if you export ONNX first): 1.13.0
- onnxruntime (optional): 1.15.0

Suggested pip install commands on macOS (run in a virtualenv):

python3 -m venv venv_coreml
source venv_coreml/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu  # or the appropriate macOS wheel
pip install coremltools==6.2.0 onnx==1.13.0 onnxruntime==1.15.0

Notes and troubleshooting:

1) coremltools native library error (libcoremlpython):
   - This happens when running on non-macOS environments or when the wheel wasn't installed properly.
   - Ensure you install coremltools on macOS with a supported Python version.

2) Op conversion errors (e.g., linear weight shape assertion):
   - Some ops (or weight shapes) may not map directly to Core ML. Strategies:
     a) Export the model-only network (cfg.model.deploy()) to ONNX first (scripts/export_model_only_onnx.py) and try converting the ONNX on macOS with coremltools.
     b) Exclude Python-only postprocessing from the traced graph (NMS, custom logic) and run them on the host side.
     c) If a particular op is unsupported, consider rewriting that subgraph (e.g., reshape linear weights to 2D) or replace with equivalent supported ops.

3) Version compatibility:
   - If coremltools raises warnings about your torch version, try installing the torch version recommended by the coremltools release notes (or downgrade coremltools to match torch).

4) Running the macOS converter script:
   - Example:
     python3 scripts/convert_dfine_to_coreml_macos.py \
       --config D-FINE-master/configs/dfine/objects365/dfine_hgnetv2_x_obj2coco.yml \
       --resume weights/dfine_x_obj2coco.pth \
       --output weights/dfine_x_obj2coco_mac.mlpackage

5) If conversion fails, collect logs and send them with the following files:
   - convert_dfine.log (script stdout/stderr)
   - The exported ONNX if present (weights/dfine_x_obj2coco_model.onnx)

If you want, I can also provide a small shell script for creating a Python venv with exact pinned versions and running the converter on macOS.