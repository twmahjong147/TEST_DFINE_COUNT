dfine_count - count objects in an image using D-FINE

Usage examples:

python -m dfine_count count --image samples/sample.jpg
python -m dfine_count count --image samples/sample.jpg --weights /mnt/AIWorkSpace/work/TEST_DFINE_COUNT/weights/dfine_x_obj2coco.pth --config /path/to/config.yaml --threshold 0.5 --per-class

The command prints a JSON object to stdout, e.g.:
{"total_count": 3}

Note: D-FINE model may require a model YAML config file. If the wrapper cannot find a default config it will ask you to pass --config.

Converting to Core ML (.mlpackage):

The repository includes a converter script that traces the deployable model and converts it to a Core ML package using coremltools. Example:

python3 scripts/convert_dfine_to_coreml.py \
    --config D-FINE-master/configs/dfine/objects365/dfine_hgnetv2_x_obj2coco.yml \
    --resume weights/dfine_x_obj2coco.pth \
    --output weights/dfine_x_obj2coco.mlpackage

Notes:
- The converter traces the model (torch.jit.trace) to produce a static graph; this requires the repository's model code to be importable so the weights can be loaded.
- Some Python-only post-processing (e.g., custom NMS or pure-Python logic) may not be included in the traced graph and therefore not part of the exported Core ML model; additional reimplementation may be required.
- Core ML conversion requires coremltools; install it with: pip install coremltools
