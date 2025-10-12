import os
import json
import cv2
import torch
import torchvision
import numpy as np
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms

try:
    import gradio as gr
except Exception:
    gr = None


def load_categories(annotation_path="train/_annotations.coco.json"):
    """Load category names from COCO-format annotation file.
    Returns a list where index i corresponds to category i+1 (since 0 is background).
    If file missing, returns a small default list.
    """
    if os.path.exists(annotation_path):
        with open(annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cats = data.get("categories", [])
        if not cats:
            return ["object"]
        # Build mapping from category id to name
        # We will create a dense list from 1..max_id
        max_id = max(c["id"] for c in cats)
        names = [""] * max_id
        for c in cats:
            names[c["id"] - 1] = c.get("name", f"cat_{c['id']}")
        # Replace empty slots with placeholder
        names = [n if n else "object" for n in names]
        return names
    else:
        # Fallback if no annotation file available
        return ["gun", "knife"]


def build_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_model(checkpoint_path="model_epoch_10.pth", annotation_path="train/_annotations.coco.json"):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    categories = load_categories(annotation_path)

    # Try to detect num_classes from checkpoint so we build a matching model
    state_dict = None
    num_classes = None
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        # Common checkpoint shapes: either the state_dict itself, or a dict containing 'model_state_dict' or 'state_dict'
        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        # Look for the classifier weight to infer number of classes
        cls_key = "roi_heads.box_predictor.cls_score.weight"
        bbox_key = "roi_heads.box_predictor.bbox_pred.weight"
        if isinstance(state_dict, dict) and cls_key in state_dict:
            try:
                num_classes = int(state_dict[cls_key].shape[0])
            except Exception:
                num_classes = None
        elif isinstance(state_dict, dict) and bbox_key in state_dict:
            try:
                out_features = int(state_dict[bbox_key].shape[0])
                # bbox_pred usually has 4 * num_classes outputs
                if out_features % 4 == 0:
                    num_classes = out_features // 4
            except Exception:
                num_classes = None

    # Fallback to categories file if detection failed
    if num_classes is None:
        num_classes = len(categories) + 1

    model = build_model(num_classes)

    # Load checkpoint into model if available
    if state_dict is not None:
        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            # Try common alternative: state dict may be top-level (already handled) or keys mismatch.
            # Retry with strict=False to allow loading the shared parts (backbone) and skipping the head if necessary.
            try:
                model.load_state_dict(state_dict, strict=False)
                print("Loaded checkpoint with strict=False (some keys were ignored).")
            except Exception:
                raise
    else:
        print(f"Warning: checkpoint '{checkpoint_path}' not found. Using pretrained backbone weights only.")

    model.to(device)
    model.eval()
    return model, categories, device


def draw_boxes(image_rgb, boxes, labels, scores, category_names, score_thr=0.5):
    out = image_rgb.copy()
    for box, label, score in zip(boxes, labels, scores):
        if score < score_thr:
            continue
        x1, y1, x2, y2 = [int(x) for x in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        class_name = category_names[label - 1] if 0 < label <= len(category_names) else str(label)
        text = f"{class_name}: {score:.2f}"
        ((text_w, text_h), _) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - text_h - 6), (x1 + text_w + 6, y1), (0, 255, 0), -1)
        cv2.putText(out, text, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return out


def censor_image(image_rgb, boxes, scores, threshold=0.5, blur_ksize=(51, 51)):
    censored = image_rgb.copy()
    for box, score in zip(boxes, scores):
        if score < threshold:
            continue
        x1, y1, x2, y2 = [int(x) for x in box]
        # Clamp coords
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(censored.shape[1], x2), min(censored.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = censored[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        blurred = cv2.GaussianBlur(roi, blur_ksize, 0)
        censored[y1:y2, x1:x2] = blurred
    return censored


# Global model load (lazy when script run)
MODEL = None
CATEGORIES = None
DEVICE = None


def process_pil_image(pil_img, score_thresh=0.5):
    """Gradio handler: takes a PIL image and returns (annotated, censored) images as numpy arrays (RGB)."""
    global MODEL, CATEGORIES, DEVICE
    if MODEL is None:
        MODEL, CATEGORIES, DEVICE = load_model()

    # Convert PIL to RGB numpy
    image_rgb = np.array(pil_img.convert("RGB"))
    # Prepare tensor
    tensor = transforms.ToTensor()(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = MODEL(tensor)
    out = outputs[0]
    boxes = out.get("boxes", torch.zeros((0, 4))).cpu().numpy()
    labels = out.get("labels", torch.zeros((0,), dtype=torch.int64)).cpu().numpy()
    scores = out.get("scores", torch.zeros((0,))).cpu().numpy()

    annotated = draw_boxes(image_rgb, boxes, labels, scores, CATEGORIES, score_thr=score_thresh)
    censored = censor_image(image_rgb, boxes, scores, threshold=score_thresh)

    # Convert BGR? images are RGB arrays already (cv2 uses BGR only for reading/writing)
    return annotated, censored


def main():
    if gr is None:
        print("Gradio not installed. Install with: pip install gradio")
        return

    # Preload model (so first request isn't slow)
    global MODEL, CATEGORIES, DEVICE
    MODEL, CATEGORIES, DEVICE = load_model()

    iface = gr.Interface(
        fn=lambda img, thr: process_pil_image(img, float(thr)),
        inputs=[gr.Image(type="pil", label="Input Image"), gr.Slider(0.0, 1.0, value=0.5, label="Score threshold")],
        outputs=[gr.Image(type="numpy", label="Detections (annotated)"), gr.Image(type="numpy", label="Censored")],
        title="RCNN Harmful Object Detection",
        description="Upload an image and the model will display detections and a censored version.",
        allow_flagging="never"
    )

    iface.launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
