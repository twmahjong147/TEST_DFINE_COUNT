import json
import os
import sys
import importlib.util
import torch
import torchvision.transforms as T
from PIL import Image
import torch
from typing import List
import argparse
import logging
import torchvision
import torch.nn.functional as F

import numpy as np
from collections import Counter
from PIL import ImageDraw, ImageFont
import ast

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
num_top_queries = 300
def load_coco_names(path=None):
    if path is None:
        path = os.path.join(REPO_ROOT, "scripts", "coco.txt")
    if not os.path.isfile(path):
        return {}
    try:
        txt = open(path, "r", encoding="utf-8").read()
        data = ast.literal_eval(txt)
        return {int(k): v for k, v in data.items()}
    except Exception:
        return {}
def filter_area_outliers(detections, labels, std_factor=2):
    """
    Remove detections whose bounding-box area is an outlier (outside mean ± std_factor * std).
    Inputs:
      detections: supervision.Detections or sequence/ndarray of boxes with .xyxy or [x1,y1,x2,y2].
      labels: list of class names corresponding to detections order.
      std_factor: float (default 2).
    Returns: (filtered_detections, filtered_labels) using the same detection container type when possible.
    """
    try:
        boxes = np.array(detections.xyxy)
        get_subset = lambda idxs: detections[idxs]
    except Exception:
        boxes = np.array([
            (det.xyxy if hasattr(det, 'xyxy') else det[:4])
            for det in detections
        ])
        def get_subset(idxs):
            if isinstance(detections, np.ndarray):
                return detections[idxs]
            return [detections[i] for i in idxs]

    if boxes.size == 0 or len(boxes) == 0:
        return detections, labels

    x_min = boxes[:, 0].astype(float)
    y_min = boxes[:, 1].astype(float)
    x_max = boxes[:, 2].astype(float)
    y_max = boxes[:, 3].astype(float)
    widths = x_max - x_min
    heights = y_max - y_min
    areas = widths * heights
    mean_area = areas.mean()
    std_area = areas.std()
    if std_area == 0 or np.isnan(std_area):
        filtered_indices = list(range(len(areas)))
    else:
        filtered_indices = [int(i) for i, area in enumerate(areas) if abs(area - mean_area) <= std_factor * std_area]
    if not filtered_indices:
        return get_subset(np.array([], dtype=int)), []
    filtered_detections = get_subset(filtered_indices)
    filtered_labels = [labels[i] for i in filtered_indices]
    return filtered_detections, filtered_labels

def remove_contained_detections(detections, labels, ioa_thresh=1.0):
    """
    Remove detections that are (almost) entirely contained inside another detection using IoA.
    Inputs:
      detections: same accepted forms as above.
      labels: list of class names corresponding to detections order.
      ioa_thresh: float (default 1.0).
    Returns: (filtered_detections, filtered_labels) in the same types when possible.
    """
    try:
        boxes = np.array(detections.xyxy)
        get_subset = lambda idxs: detections[idxs]
    except Exception:
        boxes = np.array([
            (det.xyxy if hasattr(det, 'xyxy') else det[:4])
            for det in detections
        ])
        def get_subset(idxs):
            if isinstance(detections, np.ndarray):
                return detections[idxs]
            return [detections[i] for i in idxs]
    if boxes.size == 0 or len(boxes) == 0:
        return detections, labels
    x1 = boxes[:, 0].astype(float)
    y1 = boxes[:, 1].astype(float)
    x2 = boxes[:, 2].astype(float)
    y2 = boxes[:, 3].astype(float)
    widths = (x2 - x1).clip(min=0.0)
    heights = (y2 - y1).clip(min=0.0)
    areas = widths * heights
    xi1 = np.maximum(x1[:, None], x1[None, :])
    yi1 = np.maximum(y1[:, None], y1[None, :])
    xi2 = np.minimum(x2[:, None], x2[None, :])
    yi2 = np.minimum(y2[:, None], y2[None, :])
    inter_w = np.maximum(0.0, xi2 - xi1)
    inter_h = np.maximum(0.0, yi2 - yi1)
    inter = inter_w * inter_h
    eps = 1e-6
    ioa = inter / (areas[:, None] + eps)
    np.fill_diagonal(ioa, 0.0)
    area_cmp = areas[None, :] >= areas[:, None]
    cond = (ioa >= float(ioa_thresh)) & area_cmp
    remove_mask = cond.any(axis=1)
    keep_mask = ~remove_mask
    if keep_mask.sum() == len(keep_mask):
        return detections, labels
    keep_indices = np.where(keep_mask)[0].astype(int)
    filtered_detections = get_subset(keep_indices)
    filtered_labels = [labels[i] for i in keep_indices]
    return filtered_detections, filtered_labels

def filter_aspect_outliers(detections, labels, std_factor=2):
    """
    Remove detections whose bounding-box aspect ratio (width/height) is an outlier
    outside mean ± std_factor * std.
    Inputs:
      detections: supervision.Detections or sequence/ndarray of boxes with .xyxy or [x1,y1,x2,y2].
      labels: list of class names corresponding to detections order.
      std_factor: float (default 2).
    Returns: (filtered_detections, filtered_labels) using the same detection container type when possible.
    """
    try:
        boxes = np.array(detections.xyxy)
        get_subset = lambda idxs: detections[idxs]
    except Exception:
        boxes = np.array([
            (det.xyxy if hasattr(det, 'xyxy') else det[:4])
            for det in detections
        ])
        def get_subset(idxs):
            if isinstance(detections, np.ndarray):
                return detections[idxs]
            return [detections[i] for i in idxs]

    if boxes.size == 0 or len(boxes) == 0:
        return detections, labels

    x_min = boxes[:, 0].astype(float)
    y_min = boxes[:, 1].astype(float)
    x_max = boxes[:, 2].astype(float)
    y_max = boxes[:, 3].astype(float)
    widths = (x_max - x_min)
    heights = (y_max - y_min)
    # avoid division by zero
    heights_safe = np.where(heights <= 0, 1e-6, heights)
    ratios = widths / heights_safe

    mean_ratio = ratios.mean()
    std_ratio = ratios.std()
    if std_ratio == 0 or np.isnan(std_ratio):
        filtered_indices = list(range(len(ratios)))
    else:
        filtered_indices = [int(i) for i, r in enumerate(ratios) if abs(r - mean_ratio) <= std_factor * std_ratio]

    if not filtered_indices:
        return get_subset(np.array([], dtype=int)), []

    filtered_detections = get_subset(filtered_indices)
    filtered_labels = [labels[i] for i in filtered_indices]
    return filtered_detections, filtered_labels

from dfine_count._config import DEFAULT_WEIGHTS, DEFAULT_THRESHOLD, DEFAULT_CONFIG
logger = logging.getLogger("dfine_count")

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
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            post_outputs = self.postprocessor(outputs, orig_target_sizes)
            return post_outputs

    model = Model().to(device)
    model.eval()
    return model


def _save_visualization(im_pil, outputs, threshold, image_path, model_type):
    from PIL import ImageDraw

    try:
        labels_batch, boxes_batch, scores_batch = outputs
    except Exception:
        # Fallback: if model returns a single tensor
        raise DFineIntegrationError("Unexpected model output format")

    labels = labels_batch[0].cpu()
    boxes = boxes_batch[0].cpu()
    scores = scores_batch[0].cpu()

    mask = scores > threshold
    total = int(mask.sum().item())

    result = {"total_count": total}
    per = {}
    labs = labels[mask].tolist()
    for l in labs:
        key = str(int(l))
        per[key] = per.get(key, 0) + 1
    result["per_class"] = per

    draw = ImageDraw.Draw(im_pil)

    for i, val in enumerate(scores):
        if val <= threshold:
            continue
        try:
            b = [float(x) for x in boxes[i]]
        except Exception:
            b = [float(x) for x in boxes[i].tolist()]
        # draw rectangle and label
        try:
            draw.rectangle(b, outline='red', width=2)
        except Exception:
            # older Pillow versions may not support width arg
            draw.rectangle(b, outline='red')
        try:
            label_txt = int(labels[i].item())
        except Exception:
            label_txt = int(labels[i]) if labels is not None else ''
        draw.text((b[0], b[1]), text=f"{label_txt} {val:.4f}", fill='blue')

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    outputs_dir = os.path.join(repo_root, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    fname = os.path.splitext(os.path.basename(image_path))[0] + "_" + model_type + "_vis.jpg"
    out_path = os.path.join(outputs_dir, fname)
    im_pil.save(out_path)

    result["visualization"] = out_path
    return result


# --- New main logic: process all images in samples/, output annotated images and per-class counts ---
def main():
    import glob
    import shutil
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", "-w", default=DEFAULT_WEIGHTS, help="Path to model weights")
    parser.add_argument("--config", "-c", default=DEFAULT_CONFIG, help="Path to model YAML config (if required)")
    parser.add_argument("--threshold", "-t", type=float, default=DEFAULT_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto", help="Device to run inference on")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    # Prepare model
    model = load_model(args.weights, config_path=args.config, device=device)

    SAMPLES_DIR = os.path.join(REPO_ROOT, "samples")
    OUTPUT_DIR = os.path.join(REPO_ROOT, "annotated_samples")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    coco_map = load_coco_names()
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF'}

    for filename in os.listdir(SAMPLES_DIR):
        if not any(filename.endswith(ext) for ext in image_extensions):
            continue
        image_path = os.path.join(SAMPLES_DIR, filename)
        im_pil = Image.open(image_path).convert("RGB")
        w, h = im_pil.size
        orig_size = torch.tensor([[float(w), float(h)]]).to(device)
        transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),            
            # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        im_data = transforms(im_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(im_data, orig_size)
        labels_batch, boxes_batch, scores_batch = outputs
        labels = labels_batch[0].cpu().numpy()
        boxes = boxes_batch[0].cpu().numpy()
        scores = scores_batch[0].cpu().numpy()
        mask = scores > args.threshold
        labels = labels[mask]
        boxes = boxes[mask]
        scores = scores[mask]
        # Map to string labels using COCO names if available
        str_labels = [coco_map.get(int(l), str(int(l))) for l in labels]
        # Save normal annotated image
        annotated_image = im_pil.copy()
        draw = ImageDraw.Draw(annotated_image)
        for i, box in enumerate(boxes):
            b = [float(x) for x in box]
            draw.rectangle(b, outline='red', width=2)
            draw.text((b[0], b[1]), text=f"{str_labels[i]} {scores[i]:.4f}", fill='blue')
        output_path = os.path.join(OUTPUT_DIR, f"annotated_{filename}")
        annotated_image.save(output_path)
        print(f"Annotated image saved to {output_path}")
        pruned_class_counts = {}
        pruned_class_boxes = {}
        # Per-class annotation and filtering
        for class_name in set(str_labels):
            indices = [i for i, l in enumerate(str_labels) if l == class_name]
            if not indices:
                continue
            class_boxes = boxes[indices]
            class_labels = [str_labels[i] for i in indices]
            # Area outlier filter
            class_boxes_filt, class_labels_filt = filter_area_outliers(class_boxes, class_labels, std_factor=2)
            # Aspect-ratio outlier filter (width/height)
            class_boxes_filt, class_labels_filt = filter_aspect_outliers(class_boxes_filt, class_labels_filt, std_factor=2)
            # Remove contained
            class_boxes_filt, class_labels_filt = remove_contained_detections(class_boxes_filt, class_labels_filt, ioa_thresh=0.95)
            pruned_class_counts[class_name] = len(class_boxes_filt)
            # store the pruned boxes for later use (convert to plain Python lists)
            try:
                pruned_class_boxes[class_name] = np.array(class_boxes_filt).tolist()
            except Exception:
                try:
                    pruned_class_boxes[class_name] = [list(b) for b in class_boxes_filt]
                except Exception:
                    pruned_class_boxes[class_name] = list(class_boxes_filt)
            if len(class_boxes_filt) == 0:
                continue
            class_annotated_image = im_pil.copy()
            draw_c = ImageDraw.Draw(class_annotated_image)
            # choose a larger font for label text (try common truetype, fallback to default)
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=18)
            except Exception:
                try:
                    font = ImageFont.truetype("arial.ttf", size=18)
                except Exception:
                    font = ImageFont.load_default()

            for i, box in enumerate(class_boxes_filt):
                b = [float(x) for x in box]
                x1, y1, x2, y2 = [int(round(v)) for v in b]
                # purple box and label like pasted example
                box_color = (160, 32, 240)  # purple
                text_color = (255, 255, 255)  # white
                # draw bbox (outline)
                try:
                    draw_c.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
                except TypeError:
                    draw_c.rectangle([x1, y1, x2, y2], outline=box_color)
                # prepare label background
                label_text = str(class_name)
                try:
                    text_w, text_h = draw_c.textsize(label_text, font=font)
                except Exception:
                    # Pillow compatibility fallback
                    text_w = len(label_text) * 10
                    text_h = 18
                pad = 4
                label_x0 = x1
                label_y1 = y1
                label_y0 = max(0, label_y1 - (text_h + pad * 2))
                label_x1 = x1 + text_w + pad * 2
                # draw filled label rectangle
                draw_c.rectangle([label_x0, label_y0, label_x1, label_y1], fill=box_color)
                # draw text inside label
                draw_c.text((label_x0 + pad, label_y0 + pad), label_text, fill=text_color, font=font)
            filename_no_ext = os.path.splitext(filename)[0]
            class_dir = os.path.join(OUTPUT_DIR, filename_no_ext)
            os.makedirs(class_dir, exist_ok=True)
            class_output_path = os.path.join(class_dir, f"annotated_{class_name}_{filename}")
            class_annotated_image.save(class_output_path)
            print(f"Annotated image for class '{class_name}' saved to {class_output_path}")
        # Write class counts
        from collections import Counter
        class_counts = Counter(str_labels)
        output_lines = ["Class counts (descending):\n"]
        for class_name, count in class_counts.most_common():
            line = f"{class_name}: {count}, Pruned: {pruned_class_counts.get(class_name, 0)}"
            output_lines.append(line + "\n")
        filename_no_ext = os.path.splitext(filename)[0]
        counts_output_path = os.path.join(OUTPUT_DIR, filename_no_ext, "class_counts.txt")
        with open(counts_output_path, "w") as f:
            f.writelines(output_lines)
        print(f"Class counts saved to {counts_output_path}")
        # Print most frequent class in pruned counts, show its first box, and save crop
        if pruned_class_counts:
            most_class = max(pruned_class_counts.items(), key=lambda x: x[1])[0]
            print(f"Most frequent pruned class: {most_class} ({pruned_class_counts[most_class]})")
            boxes_for_class = pruned_class_boxes.get(most_class, [])
            if boxes_for_class:
                first_box = boxes_for_class[0]
                print(f"First box for class '{most_class}': {first_box}")
                try:
                    x1, y1, x2, y2 = [int(round(float(v))) for v in first_box]
                    crop = im_pil.crop((x1, y1, x2, y2))
                    crop_fname = f"crop_{most_class}_{filename}"
                    crop_path = os.path.join(OUTPUT_DIR, filename_no_ext, crop_fname)
                    crop.save(crop_path)
                    print(f"Saved crop to {crop_path}")
                except Exception as e:
                    print(f"Failed to save crop for class {most_class}: {e}")
            else:
                print(f"No pruned boxes available for class {most_class}")


if __name__ == "__main__":
    main()
