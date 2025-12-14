import argparse
import json
import logging
import sys
from ._config import DEFAULT_WEIGHTS, DEFAULT_THRESHOLD, DEFAULT_CONFIG
from .infer import count_image

logger = logging.getLogger("dfine_count")


def main():
    parser = argparse.ArgumentParser(prog="dfine_count")
    sub = parser.add_subparsers(dest="cmd")

    p_count = sub.add_parser("count", help="Count objects in an image")
    p_count.add_argument("--image", "-i", required=True, help="Path to input image")
    p_count.add_argument("--weights", "-w", default=DEFAULT_WEIGHTS, help="Path to model weights")
    p_count.add_argument("--config", "-c", default=DEFAULT_CONFIG, help="Path to model YAML config (if required)")
    p_count.add_argument("--threshold", "-t", type=float, default=DEFAULT_THRESHOLD, help="Confidence threshold")
    p_count.add_argument("--per-class", action="store_true", help="Output per-class counts")
    p_count.add_argument("--visualize", action="store_true", help="Save visualization to outputs/")
    p_count.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto", help="Device to run inference on")
    p_count.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    if args.cmd == "count":
        try:
            # Force cpu if auto and no cuda
            device = args.device
            if device == "auto":
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    device = "cpu"

            result = count_image(
                args.image, weights_path=args.weights, config_path=args.config, threshold=args.threshold, per_class=args.per_class, device=device, visualize=args.visualize
            )
            print(json.dumps(result))
        except Exception as e:
            logger.error("Error during counting: %s", e)
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
