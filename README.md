dfine_count - count objects in an image using D-FINE

Usage examples:

python -m dfine_count count --image samples/sample.jpg
python -m dfine_count count --image samples/sample.jpg --weights /mnt/AIWorkSpace/work/TEST_DFINE_COUNT/weights/dfine_x_obj2coco.pth --config /path/to/config.yaml --threshold 0.5 --per-class

The command prints a JSON object to stdout, e.g.:
{"total_count": 3}

Note: D-FINE model may require a model YAML config file. If the wrapper cannot find a default config it will ask you to pass --config.
