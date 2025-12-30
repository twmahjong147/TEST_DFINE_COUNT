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
import cv2
from pathlib import Path

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
    
def filter_area_outliers(detections, scores, std_factor=2):
    """
    Remove detections whose bounding-box area is an outlier (outside mean ± std_factor * std).
    Inputs:
      detections: supervision.Detections or sequence/ndarray of boxes with .xyxy or [x1,y1,x2,y2].
      scores: list or array of confidence scores corresponding to detections order.
      std_factor: float (default 2).
    Returns: filtered_detections using the same detection container type when possible.
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
        return detections, scores

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
    filtered_scores = [scores[i] for i in filtered_indices]
    return filtered_detections, filtered_scores

def remove_contained_detections(detections, scores, ioa_thresh=1.0):
    """
    Remove detections that are (almost) entirely contained inside another detection using IoA.
    Inputs:
      detections: same accepted forms as above.
      scores: list or array of confidence scores corresponding to detections order.
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
        return detections, scores
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
        return detections, scores
    keep_indices = np.where(keep_mask)[0].astype(int)
    filtered_detections = get_subset(keep_indices)
    filtered_scores = [scores[i] for i in keep_indices]
    return filtered_detections, filtered_scores

def filter_aspect_outliers(detections, scores, std_factor=2):
    """
    Remove detections whose bounding-box aspect ratio (width/height) is an outlier
    outside mean ± std_factor * std.
    Inputs:
      detections: supervision.Detections or sequence/ndarray of boxes with .xyxy or [x1,y1,x2,y2].
      scores: list or array of confidence scores corresponding to detections order.
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
        return detections, scores

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
    filtered_scores = [scores[i] for i in filtered_indices]
    return filtered_detections, filtered_scores

def filter_iou_overlaps(detections, scores, iou_thresh=0.5):
    """
    Remove overlapping detections based on IoU. For pairs with IoU >= iou_thresh,
    keep the detection with the higher score (break ties by larger area).
    Returns (filtered_detections, filtered_scores) preserving input container when possible.
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
        return detections, scores

    x1 = boxes[:, 0].astype(float)
    y1 = boxes[:, 1].astype(float)
    x2 = boxes[:, 2].astype(float)
    y2 = boxes[:, 3].astype(float)
    widths = (x2 - x1).clip(min=0.0)
    heights = (y2 - y1).clip(min=0.0)
    areas = widths * heights

    # pairwise intersection
    xi1 = np.maximum(x1[:, None], x1[None, :])
    yi1 = np.maximum(y1[:, None], y1[None, :])
    xi2 = np.minimum(x2[:, None], x2[None, :])
    yi2 = np.minimum(y2[:, None], y2[None, :])
    inter_w = np.maximum(0.0, xi2 - xi1)
    inter_h = np.maximum(0.0, yi2 - yi1)
    inter = inter_w * inter_h

    eps = 1e-6
    union = areas[:, None] + areas[None, :] - inter + eps
    iou = inter / union
    np.fill_diagonal(iou, 0.0)

    # greedy keep based on score then area
    scores_arr = np.array(scores, dtype=float)
    idxs = list(range(len(boxes)))
    keep = []
    # Sort indices by score desc, area desc to prefer stronger/larger boxes
    order = sorted(idxs, key=lambda i: (scores_arr[i], areas[i]), reverse=True)
    removed = np.zeros(len(boxes), dtype=bool)
    for i in order:
        if removed[i]:
            continue
        keep.append(i)
        # remove any remaining boxes that have high IoU with this kept box
        high_iou = np.where(iou[i] >= float(iou_thresh))[0]
        for j in high_iou:
            removed[j] = True

    if not keep:
        return get_subset(np.array([], dtype=int)), []

    keep_indices = np.array(keep, dtype=int)
    filtered_detections = get_subset(keep_indices)
    filtered_scores = [scores[i] for i in keep_indices]
    return filtered_detections, filtered_scores

from PIL import ImageEnhance

def get_extreme_contrast_images(im, factors=None):
    if factors is None:
        factors = list(np.linspace(0.4, 2.0, 10))
    best_img = None
    worst_img = None
    max_std = -1.0
    min_std = float('inf')
    for f in factors:
        im2 = ImageEnhance.Contrast(im).enhance(float(f))
        arr = np.array(im2.convert('L'), dtype=np.float32)
        s = float(arr.std())
        if s > max_std:
            max_std = s
            best_img = im2.copy()
        if s < min_std:
            min_std = s
            worst_img = im2.copy()
    return best_img, worst_img


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

def draw_box(draw, box, label_text, box_color=(255, 0, 0), text_color=(0, 0, 255), font=None):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    try:
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
    except TypeError:
        draw.rectangle([x1, y1, x2, y2], outline=box_color)
    # Prefer `textbbox` (newer Pillow) and fall back to font methods or a simple estimate
    try:
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except Exception:
        try:
            bbox = font.getbbox(label_text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_w = len(label_text) * 10
            text_h = 18
    pad = 4
    label_x0 = x1
    label_y1 = y1
    label_y0 = max(0, label_y1 - (text_h + pad * 2))
    label_x1 = x1 + text_w + pad * 2
    draw.rectangle([label_x0, label_y0, label_x1, label_y1], fill=box_color)
    draw.text((label_x0 + pad, label_y0 + pad), label_text, fill=text_color, font=font)

def draw_boxes_on_image(im_pil, boxes, scores, labels_ary=None, label = ""):
    from PIL import ImageDraw, ImageFont

    annotated_image = im_pil.copy()
    draw = ImageDraw.Draw(annotated_image)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=18)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", size=18)
        except Exception:
            font = ImageFont.load_default()

    for i, box in enumerate(boxes):
        label_text = f"{i+1}: {labels_ary[i]} {scores[i]:.4f}" if labels_ary is not None else f"{i+1}: {label} {scores[i]:.4f}"
        draw_box(draw, box, label_text, box_color=(255, 0, 0), text_color=(0, 0, 255), font=font)

    return annotated_image


def process_image(model, im_pil, im_name, args, OUTPUT_DIR, coco_map, device):
    from collections import Counter
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

    str_labels = [coco_map.get(int(l), str(int(l))) for l in labels]

    details_dir = os.path.join(OUTPUT_DIR, im_name)
    os.makedirs(details_dir, exist_ok=True)

    # Save all boxes normal annotated image
    draw_boxes_on_image(im_pil, boxes, scores, labels_ary=str_labels).save(os.path.join(details_dir, "annotated_all.jpg"))

    pruned_class_counts = {}
    pruned_class_boxes = {}
    # Per-class annotation and filtering

    for class_name in set(str_labels):
        indices = [i for i, l in enumerate(str_labels) if l == class_name]
        if not indices:
            continue

        class_dir = os.path.join(details_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        class_boxes = boxes[indices]
        class_scores = scores[indices]

        draw_boxes_on_image(im_pil, class_boxes, class_scores, label=class_name).save(os.path.join(class_dir, f"0_unfiltered.jpg"))

        class_boxes_filt, class_scores_filt = filter_iou_overlaps(class_boxes, class_scores, iou_thresh=0.5)
        draw_boxes_on_image(im_pil, class_boxes_filt, class_scores_filt, label=class_name).save(os.path.join(class_dir, f"1_iou_filtered.jpg"))
        # # Area outlier filter
        class_boxes_filt, class_scores_filt = filter_area_outliers(class_boxes_filt, class_scores_filt, std_factor=2)
        draw_boxes_on_image(im_pil, class_boxes_filt, class_scores_filt, label=class_name).save(os.path.join(class_dir, f"2_area_filtered.jpg"))
        # Aspect-ratio outlier filter (width/height)
        class_boxes_filt, class_scores_filt = filter_aspect_outliers(class_boxes_filt, class_scores_filt, std_factor=2)
        draw_boxes_on_image(im_pil, class_boxes_filt, class_scores_filt, label=class_name).save(os.path.join(class_dir, f"3_aspect_filtered.jpg"))

        # Remove contained
        class_boxes_filt, class_scores_filt = remove_contained_detections(class_boxes_filt, class_scores_filt, ioa_thresh=0.25)
        draw_boxes_on_image(im_pil, class_boxes_filt, class_scores_filt, label=class_name).save(os.path.join(class_dir, f"4_contained_filtered.jpg"))
        
        pruned_class_counts[class_name] = len(class_boxes_filt)
        # store the pruned boxes for later use (convert to plain Python lists)
        try:
            pruned_class_boxes[class_name] = np.array(class_boxes_filt).tolist()
        except Exception:
            try:
                pruned_class_boxes[class_name] = [list(b) for b in class_boxes_filt]
            except Exception:
                pruned_class_boxes[class_name] = list(class_boxes_filt)

    # Write class counts
    class_counts = Counter(str_labels)
    output_lines = ["Class counts (descending):\n"]
    for class_name, count in class_counts.most_common():
        line = f"{class_name}: {count}, Pruned: {pruned_class_counts.get(class_name, 0)}"
        output_lines.append(line + "\n")
    counts_output_path = os.path.join(details_dir, "class_counts.txt")
    with open(counts_output_path, "w") as f:
        f.writelines(output_lines)

    # most_class = max(pruned_class_counts.items(), key=lambda x: x[1])[0]
    most_class = class_counts.most_common(1)[0][0]
    print(f"{im_name}: Most frequent pruned class: {most_class} ({pruned_class_counts[most_class]})")
    return pruned_class_boxes.get(most_class, [])
            # try:
            #     x1, y1, x2, y2 = [int(round(float(v))) for v in first_box]
            #     crop = im_pil.crop((x1, y1, x2, y2))
            #     crop.save(f"crop_{most_class}.jpg")
            # except Exception as e:
            #     print(f"Failed to save crop for class {most_class}: {e}")

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
        im_name = os.path.splitext(os.path.basename(image_path))[0]
        # get strongest and weakest contrast variants
        strong_im, weak_im = get_extreme_contrast_images(im_pil)

        # ensure details dir exists early so we can save debug images
        details_dir = os.path.join(OUTPUT_DIR, im_name)
        os.makedirs(details_dir, exist_ok=True)
        try:
            if strong_im is not None:
                strong_im.save(os.path.join(details_dir, "strong_contrast.jpg"))
            if weak_im is not None:
                weak_im.save(os.path.join(details_dir, "weak_contrast.jpg"))
        except Exception as e:
            print(f"Failed saving contrast images for {im_name}: {e}")

        # Call process_image on original, strongest, and weakest contrast images
        boxes_orig = process_image(model, im_pil.copy(), im_name + "_orig", args, OUTPUT_DIR, coco_map, device)
        boxes_strong = process_image(model, strong_im.copy(), im_name + "_strong", args, OUTPUT_DIR, coco_map, device) if strong_im is not None else []
        boxes_weak = process_image(model, weak_im.copy(), im_name + "_weak", args, OUTPUT_DIR, coco_map, device) if weak_im is not None else []

        # Merge returned boxes
        merged = []
        for bset in (boxes_orig, boxes_strong, boxes_weak):
            if bset is None:
                continue
            try:
                for bb in bset:
                    merged.append([float(x) for x in bb])
            except Exception:
                pass

        if not merged:
            continue

        merged_arr = np.array(merged)
        dummy_scores = [1.0] * len(merged_arr)

        # IoU overlap filter: remove highly-overlapping duplicates (prefer higher score/larger area)
        try:
            merged_arr, dummy_scores = filter_iou_overlaps(merged_arr, dummy_scores, iou_thresh=0.5)
        except Exception:
            # fallback to originals if something goes wrong
            pass

        # Remove contained detections from merged boxes
        final_boxes, final_scores = remove_contained_detections(merged_arr, dummy_scores, ioa_thresh=0.25)

        # Draw final boxes on original image and save
        details_dir = os.path.join(OUTPUT_DIR, im_name)
        os.makedirs(details_dir, exist_ok=True)
        annotated = draw_boxes_on_image(im_pil, final_boxes, final_scores)
        annotated.save(os.path.join(details_dir, "final_merged.jpg"))
        print(f"Saved final merged boxes for {im_name} to {os.path.join(details_dir, 'final_merged.jpg')}")


if __name__ == "__main__":
    main()
