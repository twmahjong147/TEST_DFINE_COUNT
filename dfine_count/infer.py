import os
import sys
import importlib.util
import torch
import torchvision.transforms as T
from PIL import Image


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
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(device)
    model.eval()
    return model


def _save_visualization(im_pil, labels, boxes, scores, threshold, image_path):
    from PIL import ImageDraw
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
    fname = os.path.splitext(os.path.basename(image_path))[0] + "_vis.jpg"
    out_path = os.path.join(outputs_dir, fname)
    im_pil.save(out_path)
    return out_path


def count_image(image_path, weights_path=None, config_path=None, threshold=0.5, per_class=False, device="cpu", visualize=False):
    if weights_path is None:
        raise DFineIntegrationError("weights_path must be provided")

    model = load_model(weights_path, config_path=config_path, device=device)

    # Preprocess image
    im_pil = Image.open(image_path).convert("RGB")
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)
    transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    im_data = transforms(im_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(im_data, orig_size)

    # outputs: labels, boxes, scores
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
    if per_class:
        per = {}
        labs = labels[mask].tolist()
        for l in labs:
            key = str(int(l))
            per[key] = per.get(key, 0) + 1
        result["per_class"] = per

    if visualize:
        try:
            vis_path = _save_visualization(im_pil.copy(), labels, boxes, scores, threshold, image_path)
            result["visualization"] = vis_path
        except Exception as e:
            # don't fail the whole run for visualization errors
            import logging
            logging.getLogger("dfine_count").warning("Failed to save visualization: %s", e)

    return result
