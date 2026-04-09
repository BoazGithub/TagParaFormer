"""
Inference script for TagParaFormer.

Usage:
    python inference/predict.py \
        --checkpoint checkpoints/best.pth \
        --input path/to/image.png \
        --output output/pred.png \
        --threshold 0.3
"""

import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.morphology import skeletonize, remove_small_objects

sys.path.insert(0, str(Path(__file__).parents[1]))
from models.tagparaformer import TagParaFormer

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img_path: str, img_size: int = 256):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]
    img_resized = cv2.resize(img, (img_size, img_size))
    img_norm = (img_resized.astype(np.float32) / 255.0 - MEAN) / STD
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0)
    return tensor, (orig_h, orig_w), img


def postprocess(
    logits: torch.Tensor,
    orig_size: tuple,
    threshold: float = 0.3,
    min_size: int = 200,
) -> np.ndarray:
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    # Resize to original
    prob_resized = cv2.resize(prob, (orig_size[1], orig_size[0]))
    # Threshold
    mask = (prob_resized > threshold).astype(np.uint8)
    # Remove small isolated components
    mask_bool = remove_small_objects(mask.astype(bool), min_size=min_size)
    return mask_bool.astype(np.uint8) * 255


def visualize_predictions(img: np.ndarray, mask: np.ndarray, save_path: str):
    """Overlay prediction mask on original image."""
    overlay = img.copy()
    road_pixels = mask > 127
    overlay[road_pixels] = (0.4 * overlay[road_pixels] + 0.6 * np.array([255, 180, 0])).astype(np.uint8)
    comparison = np.concatenate([img, overlay], axis=1)
    cv2.imwrite(save_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    print(f"Visualization saved → {save_path}")


def predict_tiled(
    model: torch.nn.Module,
    img_path: str,
    device: torch.device,
    tile_size: int = 256,
    overlap: int = 32,
    threshold: float = 0.3,
) -> np.ndarray:
    """
    Sliding-window tiled inference for large images.
    Handles images larger than tile_size with overlap blending.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]

    prob_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)
    stride = tile_size - overlap

    model.eval()
    with torch.no_grad():
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                y2 = min(y + tile_size, H)
                x2 = min(x + tile_size, W)
                y1 = y2 - tile_size
                x1 = x2 - tile_size
                if y1 < 0: y1 = 0
                if x1 < 0: x1 = 0

                tile = img[y1:y2, x1:x2]
                if tile.shape[:2] != (tile_size, tile_size):
                    tile = cv2.resize(tile, (tile_size, tile_size))

                tile_norm = (tile.astype(np.float32) / 255.0 - MEAN) / STD
                tensor = torch.from_numpy(tile_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)
                logits = model(tensor)
                prob = torch.sigmoid(logits).squeeze().cpu().numpy()
                prob_resized = cv2.resize(prob, (x2 - x1, y2 - y1))

                prob_map[y1:y2, x1:x2] += prob_resized
                count_map[y1:y2, x1:x2] += 1.0

    avg_prob = prob_map / (count_map + 1e-6)
    mask = (avg_prob > threshold).astype(np.uint8) * 255
    return mask


def main():
    parser = argparse.ArgumentParser(description="TagParaFormer Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input",      type=str, required=True)
    parser.add_argument("--output",     type=str, default="prediction.png")
    parser.add_argument("--threshold",  type=float, default=0.3)
    parser.add_argument("--img_size",   type=int,   default=256)
    parser.add_argument("--tiled",      action="store_true", help="Use tiled inference for large images")
    parser.add_argument("--visualize",  action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ──
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg  = ckpt.get("config", {})
    model = TagParaFormer(
        in_channels=cfg.get("in_channels", 3),
        img_size=args.img_size,
        embed_dim=cfg.get("embed_dim", 256),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Model loaded from {args.checkpoint}")
    print(f"Fusion weights: α={model.get_fusion_weights()['alpha']:.3f}  β={model.get_fusion_weights()['beta']:.3f}")

    # ── Inference ──
    if args.tiled:
        mask = predict_tiled(model, args.input, device, args.img_size, threshold=args.threshold)
    else:
        tensor, orig_size, orig_img = preprocess(args.input, args.img_size)
        tensor = tensor.to(device)
        with torch.no_grad():
            logits = model(tensor)
        mask = postprocess(logits, orig_size, args.threshold)

    cv2.imwrite(args.output, mask)
    print(f"Prediction saved → {args.output}")

    if args.visualize:
        orig_img = cv2.cvtColor(cv2.imread(args.input), cv2.COLOR_BGR2RGB)
        vis_path = str(Path(args.output).with_suffix("")) + "_vis.jpg"
        visualize_predictions(orig_img, mask, vis_path)


if __name__ == "__main__":
    main()
