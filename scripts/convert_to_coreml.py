import argparse
import os
import onnx

try:
    import coremltools as ct
except Exception as e:
    raise RuntimeError("coremltools is required. Install with: pip install coremltools") from e


def convert(onnx_path, output_path):
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    print(f"Loading ONNX model from {onnx_path}...")
    onnx_model = onnx.load(onnx_path)

    print("Converting ONNX model to Core ML using coremltools...")
    mlmodel = ct.converters.onnx.convert(onnx_model)

    print(f"Saving Core ML model to {output_path}...")
    mlmodel.save(output_path)
    print("Conversion complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--output', type=str, required=True, help='Path to output .mlpackage')
    args = parser.parse_args()
    convert(args.onnx, args.output)
