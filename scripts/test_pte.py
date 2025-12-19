import json
import os
import sys
import importlib.util
import torch
import torchvision.transforms as T
from PIL import Image
import torch
from executorch.runtime import Runtime
from typing import List
import argparse
import logging
import torchvision
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
num_top_queries = 300
num_classes = 80

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
            return (outputs, post_outputs)

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

def mod(a, b):
    out = a - a // b * b
    return out

def postprocessor_pte(outputs, orig_target_sizes: torch.Tensor):
    logits, boxes = outputs

    bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
    bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

    # if self.use_focal_loss:
    scores = F.sigmoid(logits)
    scores, index = torch.topk(scores.flatten(1), num_top_queries, dim=-1)
    # TODO for older tensorrt
    labels = mod(index, num_classes)
    index = index // num_classes
        
    boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))

    return labels, boxes, scores

def count_image(image_path, weights_path=None, config_path=None, threshold=0.001, per_class=False, device="cpu", visualize=True):
    try:
        if weights_path is None:
            raise DFineIntegrationError("weights_path must be provided")

        # Preprocess image
        im_pil = Image.open(image_path).convert("RGB")
        w, h = im_pil.size
        orig_size = torch.tensor([[float(w), float(h)]]).to(device)
        transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])
        im_data = transforms(im_pil).unsqueeze(0).to(device)

        # Save im_data tensor as image for debugging (non-blocking on failure)
        try:
            im_data_cpu = im_data.detach().cpu().squeeze(0)
            pil_from_im_data = T.ToPILImage()(im_data_cpu)
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            outputs_dir = os.path.join(repo_root, "outputs")
            os.makedirs(outputs_dir, exist_ok=True)
            im_data_fname = os.path.splitext(os.path.basename(image_path))[0] + "_im_data.png"
            im_data_path = os.path.join(outputs_dir, im_data_fname)
            pil_from_im_data.save(im_data_path)
            logger.info("Saved im_data tensor to %s", im_data_path)
        except Exception as save_e:
            logger.debug("Could not save im_data for debugging: %s", save_e)

        # Load model via executorch
        default_pte_weight = "weights/dfine_x_obj2coco.pte"
        runtime =Runtime.get()
        program = runtime.load_program(default_pte_weight)
        method = program.load_method("forward")

        outputs_pte = method.execute( (im_data, orig_size))
        outputs = postprocessor_pte(outputs_pte, orig_size)
        print("Run successfully via executorch")

        pte_result  = _save_visualization(im_pil.copy(), outputs, threshold, image_path, "pte")
        print(json.dumps(pte_result))

        # Load model via dfine_count.infer
        model = load_model(weights_path, config_path=config_path, device=device)

        with torch.no_grad():
            (outputs, post_outputs) = model(im_data, orig_size)

        pth_result  = _save_visualization(im_pil.copy(), post_outputs, threshold, image_path, "pth")

        print(json.dumps(pth_result))

    except Exception as e:
        raise RuntimeError('ml-fastvit-main not found or missing dependencies. Ensure the ml-fastvit-main folder is present.') from e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", required=True, help="Path to input image")
    parser.add_argument("--weights", "-w", default=DEFAULT_WEIGHTS, help="Path to model weights")
    parser.add_argument("--config", "-c", default=DEFAULT_CONFIG, help="Path to model YAML config (if required)")
    parser.add_argument("--threshold", "-t", type=float, default=DEFAULT_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--per-class", action="store_true", help="Output per-class counts")
    parser.add_argument("--visualize", default=True, action="store_true", help="Save visualization to outputs/")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto", help="Device to run inference on")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    try:
        args = parser.parse_args()
    except Exception as e:
        raise RuntimeError() from e

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    try:
        # Force cpu if auto and no cuda
        device = args.device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        count_image(
            args.image, weights_path=args.weights, config_path=args.config, threshold=args.threshold, per_class=args.per_class, device=device, visualize=args.visualize
        )
    except Exception as e:
        logger.error("Error during counting: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
