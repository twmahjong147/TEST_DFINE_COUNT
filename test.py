import os
import supervision as sv
from inference import get_model
from PIL import Image
from collections import Counter
import numpy as np


def filter_area_outliers(detections, labels, std_factor=2):
    """
    Filters out detections whose bounding box area is an outlier (outside mean ± std_factor*std).
    Returns filtered detections and labels.
    """
    import numpy as np

    # Prefer using supervision's Detections.xyxy when available
    try:
        # If detections is a supervision.Detections object
        boxes = np.array(detections.xyxy)
        get_subset = lambda idxs: detections[idxs]
    except Exception:
        # Fallback: detections may be a list/ndarray of tuples/arrays
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
        # Nothing to filter if no variance
        filtered_indices = list(range(len(areas)))
    else:
        filtered_indices = [
            int(i) for i, area in enumerate(areas)
            if abs(area - mean_area) <= std_factor * std_area
        ]

    if not filtered_indices:
        # Return empty of the same type as input
        return get_subset(np.array([], dtype=int)), []

    filtered_detections = get_subset(filtered_indices)
    filtered_labels = [labels[i] for i in filtered_indices]
    return filtered_detections, filtered_labels


def remove_contained_detections(detections, labels, ioa_thresh=1.0):
    """
    Remove detections that are (almost) totally contained inside another detection.
    Uses IoA (intersection over area of the smaller box) and keeps the larger-area
    box when there is containment. Returns filtered (detections, labels) in the
    same types as the inputs where possible.
    """
    import numpy as np

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

    # pairwise intersection
    xi1 = np.maximum(x1[:, None], x1[None, :])
    yi1 = np.maximum(y1[:, None], y1[None, :])
    xi2 = np.minimum(x2[:, None], x2[None, :])
    yi2 = np.minimum(y2[:, None], y2[None, :])

    inter_w = np.maximum(0.0, xi2 - xi1)
    inter_h = np.maximum(0.0, yi2 - yi1)
    inter = inter_w * inter_h

    eps = 1e-6
    ioa = inter / (areas[:, None] + eps)

    # zero self IoA
    np.fill_diagonal(ioa, 0.0)

    # prefer keeping larger-area box when contained
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


SAMPLES_DIR = "samples"
OUTPUT_DIR = "annotated_samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = get_model("rfdetr-large")

image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF'}


# Collect all class names

for filename in os.listdir(SAMPLES_DIR):
    if not any(filename.endswith(ext) for ext in image_extensions):
        continue

    all_labels = []
    class_to_detections = {}

    image_path = os.path.join(SAMPLES_DIR, filename)
    image = Image.open(image_path)
    predictions = model.infer(image, confidence=0.001)[0]
    detections = sv.Detections.from_inference(predictions)
    labels = [prediction.class_name for prediction in predictions.predictions]
    all_labels.extend(labels)

    # Save normal annotated image
    annotated_image = image.copy()
    annotated_image = sv.BoxAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(annotated_image, detections)
    annotated_image = sv.LabelAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(annotated_image, detections, labels)
    sv.plot_image(annotated_image)
    output_path = os.path.join(OUTPUT_DIR, f"annotated_{filename}")
    annotated_image.save(output_path)
    print(f"Annotated image saved to {output_path}")
    pruned_class_counts = {}
    # Generate and save per-class annotated images
    for class_name in set(labels):
        # Filter detections and labels for this class
        indices = [i for i, l in enumerate(labels) if l == class_name]
        if not indices:
            continue
        # Filter detections for this class
        class_detections = detections[indices]
        class_labels = [labels[i] for i in indices]

        # Filter out area outliers using mean and std
        class_detections, class_labels = filter_area_outliers(class_detections, class_labels, std_factor=2)

        # Remove detections that are contained in another detection (IoA-based)
        class_detections, class_labels = remove_contained_detections(class_detections, class_labels, ioa_thresh=0.75)

        pruned_class_counts[class_name] = len(class_detections)
        if len(class_detections) == 0:
            continue

        class_annotated_image = image.copy()
        class_annotated_image = sv.BoxAnnotator(
            color=sv.ColorPalette.ROBOFLOW
        ).annotate(class_annotated_image, class_detections)
        class_annotated_image = sv.LabelAnnotator(
            color=sv.ColorPalette.ROBOFLOW
        ).annotate(class_annotated_image, class_detections, class_labels)
        # Create subdirectory for each image (without extension)
        filename_no_ext = os.path.splitext(filename)[0]
        class_dir = os.path.join(OUTPUT_DIR, filename_no_ext)
        os.makedirs(class_dir, exist_ok=True)
        class_output_path = os.path.join(
            class_dir, f"annotated_{class_name}_{filename}"
        )
        class_annotated_image.save(class_output_path)
        print(f"Annotated image for class '{class_name}' saved to {class_output_path}")

    # Group by class name and print count in descending order
    class_counts = Counter(all_labels)
    print("\nClass counts (descending):")
    output_lines = ["Class counts (descending):\n"]
    for class_name, count in class_counts.most_common():
        line = f"{class_name}: {count}, Pruned: {pruned_class_counts.get(class_name, 0)}"
        print(line)
        output_lines.append(line + "\n")
    # Save to file
    filename_no_ext = os.path.splitext(filename)[0]
    counts_output_path = os.path.join(OUTPUT_DIR, filename_no_ext, "class_counts.txt")
    with open(counts_output_path, "w") as f:
        f.writelines(output_lines)
    print(f"Class counts saved to {counts_output_path}")
