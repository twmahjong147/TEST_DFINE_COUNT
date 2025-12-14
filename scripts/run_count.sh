#!/usr/bin/env bash
python -m dfine_count count --image "$1" --weights "/mnt/AIWorkSpace/work/TEST_DFINE_COUNT/weights/dfine_x_obj2coco.pth" --device cpu
