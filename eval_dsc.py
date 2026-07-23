"""Evaluate Dynamic Subject Consistency (DSC) for generated videos.

  - DSC_con: generated target versus context frames
  - DSC_tgt: generated target versus ground-truth target frames
"""

import argparse
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align
from transformers import CLIPModel, CLIPProcessor


def load_video(path: str) -> List[np.ndarray]:
    """Decode a video into RGB uint8 frames."""
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    frames = []
    while True:
        success, frame = capture.read()
        if not success:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError(f"No frames decoded from: {path}")
    return frames


def boxes_from_dynamic_mask(
    dynamic_mask_path: str,
    context_length: int,
    target_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read subject boxes from context+target CustomDepth annotations.

    This example assumes that CustomDepth is temporally aligned with the two RGB
    videos and marks the subject in red. Replace this mask parser if another
    instance-mask representation is used.
    """
    depth_frames = load_video(dynamic_mask_path)
    required = context_length + target_length
    if len(depth_frames) < required:
        raise ValueError(
            f"CustomDepth has {len(depth_frames)} frames; expected at least {required}."
        )

    boxes = []
    for rgb in depth_frames[:required]:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask_1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        mask_2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask_1, mask_2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            boxes.append([-1.0, -1.0, -1.0, -1.0])
            continue

        x, y, width, height = cv2.boundingRect(np.concatenate(contours, axis=0))
        frame_height, frame_width = rgb.shape[:2]
        boxes.append([
            x / frame_width,
            y / frame_height,
            (x + width) / frame_width,
            (y + height) / frame_height,
        ])

    boxes = torch.tensor(boxes, dtype=torch.float32)
    return boxes[:context_length], boxes[context_length:required]


def _extract_roi_features(
    frames: Sequence[np.ndarray],
    boxes: torch.Tensor,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    roi_size: int = 7,
) -> torch.Tensor:
    if len(frames) != len(boxes):
        raise ValueError("The number of frames and boxes must match.")

    valid = boxes[:, 0] >= 0
    if not valid.any():
        raise ValueError("No valid subject boxes were provided.")

    valid_ids = torch.where(valid)[0]
    images = [frames[i] for i in valid_ids.tolist()]
    inputs = processor(images=images, return_tensors="pt")

    with torch.no_grad():
        output = model.vision_model(
            pixel_values=inputs.pixel_values.to(device),
            output_hidden_states=False,
        )
        patch_tokens = output.last_hidden_state[:, 1:, :]
        side = int(patch_tokens.shape[1] ** 0.5)
        feature_map = patch_tokens.reshape(len(images), side, side, -1)
        feature_map = feature_map.permute(0, 3, 1, 2)

    roi_boxes = boxes[valid].to(device).clone()
    roi_boxes[:, [0, 2]] *= side
    roi_boxes[:, [1, 3]] *= side
    batch_ids = torch.arange(len(roi_boxes), device=device)[:, None]
    roi_boxes = torch.cat([batch_ids, roi_boxes], dim=1)

    regions = roi_align(
        feature_map,
        roi_boxes,
        output_size=(roi_size, roi_size),
        spatial_scale=1.0,
        aligned=True,
    )
    features = regions.mean(dim=(2, 3))
    return F.normalize(features, dim=-1)


def _cross_frame_similarity(features_a: torch.Tensor, features_b: torch.Tensor) -> float:
    """Average all pairwise cosine similarities between two video segments."""
    similarity = features_a @ features_b.T
    return similarity.mean().item()


def compute_dsc(
    context_frames: Sequence[np.ndarray],
    pred_frames: Sequence[np.ndarray],
    gt_frames: Sequence[np.ndarray],
    context_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> Dict[str, float]:
    """Compute context and target Dynamic Subject Consistency scores.

    All frames must be RGB uint8 arrays. ``context_boxes`` and ``target_boxes``
    are normalized xyxy boxes for context and GT frames respectively. Unlike the
    GT branch, the generated-video branch does not reuse ``target_boxes``:
    ``pred_boxes`` are obtained independently by tracking the generated subject.
    """
    # pred_boxes are obtained by tracking the target object across pred_frames using YOLOv11.

    pred_boxes = track_subject(pred_frames)

    context_features = _extract_roi_features(
        context_frames, context_boxes, model, processor, device
    )
    pred_features = _extract_roi_features(
        pred_frames, pred_boxes, model, processor, device
    )
    gt_features = _extract_roi_features(
        gt_frames, target_boxes, model, processor, device
    )

    return {
        "DSC_con": _cross_frame_similarity(pred_features, context_features),
        "DSC_tgt": _cross_frame_similarity(pred_features, gt_features),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic Subject Consistency evaluation")
    parser.add_argument("--context_video", required=True, help="Context/reference RGB video")
    parser.add_argument("--target_video", required=True, help="Ground-truth target RGB video")
    parser.add_argument("--pred_video", required=True, help="Generated target RGB video")
    parser.add_argument(
        "--dynamic_mask",
        required=True,
        help="Dynamic Mask Video from the dataset",
    )
    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch16")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    context_frames = load_video(args.context_video)
    target_frames = load_video(args.target_video)
    pred_frames = load_video(args.pred_video)
    context_boxes, target_boxes = boxes_from_dynamic_mask(
        args.dynamic_mask,
        context_length=len(context_frames),
        target_length=len(target_frames),
    )

    clip_model = CLIPModel.from_pretrained(args.clip_model).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)
    
    scores = compute_dsc(
        context_frames=context_frames,
        pred_frames=pred_frames,
        gt_frames=target_frames,
        context_boxes=context_boxes,
        target_boxes=target_boxes,
        model=clip_model,
        processor=clip_processor,
        device=device,
    )
    print(f"DSC_con: {scores['DSC_con']:.6f}")
    print(f"DSC_tgt: {scores['DSC_tgt']:.6f}")


if __name__ == "__main__":
    main()
