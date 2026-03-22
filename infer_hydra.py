import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision.transforms import v2

from diffsynth import HyDRAPipeline, ModelManager, save_video
from diffsynth.models.wan_video_dit import DynamicRetrievalAttention, MemoryTokenizer

def _crop_and_resize(image: Image.Image, height: int, width: int) -> Image.Image:
    src_w, src_h = image.size
    scale = max(width / src_w, height / src_h)
    image = torchvision.transforms.functional.resize(
        image,
        (round(src_h * scale), round(src_w * scale)),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
    )
    return torchvision.transforms.functional.center_crop(image, [height, width])


def load_condition_video(
    path: str,
    num_frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Returns normalized video tensor: [C, T, H, W] in [-1, 1].
    """
    frame_process = v2.Compose(
        [
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    reader = imageio.get_reader(path)
    frames: List[torch.Tensor] = []
    try:
        for i in range(num_frames):
            try:
                frame = reader.get_data(i)
            except Exception:
                if not frames:
                    raise
                frames.append(frames[-1].clone())
                continue
            img = Image.fromarray(frame)
            img = _crop_and_resize(img, height=height, width=width)
            frames.append(frame_process(img))
    finally:
        reader.close()

    if len(frames) < num_frames and frames:
        frames.extend([frames[-1].clone() for _ in range(num_frames - len(frames))])

    video = torch.stack(frames, dim=0)  # [T,C,H,W]
    video = video.permute(1, 0, 2, 3).contiguous()  # [C,T,H,W]
    return video


def _parse_camera_matrix(json_str: str) -> torch.Tensor:
    """
    Parse camera matrix string: "[-1 ... ] [ ... ] [ ... ] [ ... ] "
    Returns 4x4 tensor (transposed from column vectors).
    """
    matches = re.findall(r"\[(.*?)\]", json_str)
    if len(matches) != 4:
        return torch.eye(4)

    cols = []
    for m in matches:
        vals = [float(x) for x in m.strip().split()]
        if len(vals) != 4:
            return torch.eye(4)
        cols.append(vals)

    return torch.tensor(cols, dtype=torch.float32).T


def _apply_coordinate_transform(c2w: torch.Tensor) -> torch.Tensor:
    """
    Copied from ContextMemorySegmentsDataset._apply_coordinate_transform (inference_context.py).
    """
    c2w_t = c2w.clone()
    c2w_t = c2w_t[:, [1, 2, 0, 3]]
    c2w_t[:3, 1] *= -1.0
    c2w_t[:3, 3] /= 100.0
    return c2w_t


def _dict77_to_tensor(cam_dict: Dict[str, Any], name: str) -> torch.Tensor:
    mats: List[torch.Tensor] = []
    for i in range(77):
        key = str(i)
        if key not in cam_dict:
            raise ValueError(f"{name} missing key {key}")
        m = torch.as_tensor(cam_dict[key], dtype=torch.float32)
        if m.shape != (4, 4):
            raise ValueError(f"{name}[{key}] must be [4,4], got {tuple(m.shape)}")
        mats.append(m)
    return torch.stack(mats, dim=0)


def _compute_relative_20x12_from_77x4x4(context_77: torch.Tensor, target_77: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cam_idx = list(range(77))[::4]  # 20
    ref_c2w = _apply_coordinate_transform(target_77[0])
    ref_w2c = torch.inverse(ref_c2w)

    def to_rel(seq_77: torch.Tensor) -> torch.Tensor:
        rel_list: List[torch.Tensor] = []
        for idx in cam_idx:
            c2w = _apply_coordinate_transform(seq_77[idx])
            rel = ref_w2c @ c2w
            rel_list.append(rel[:3, :4].contiguous().view(-1))
        return torch.stack(rel_list, dim=0)  # [20,12]

    return to_rel(context_77), to_rel(target_77)


def load_condition_json(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Camera loader for inference.

    Supports:
    - Split camera json (preferred for examples): {"cond_cam": {...}, "tgt_cam": {...}}
      where each is dict of 77 frames, each frame is a 4x4 numeric matrix.
    - Also accepts {"context_camera": ..., "video_camera": ...} where tensors are either [20,12] (relative) or [77,4,4] (absolute).

    """
    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    if "cond_cam" in data and "tgt_cam" in data:
        ctx77 = _dict77_to_tensor(data["cond_cam"], "cond_cam")
        tgt77 = _dict77_to_tensor(data["tgt_cam"], "tgt_cam")
        return _compute_relative_20x12_from_77x4x4(ctx77, tgt77)

    if "context_camera" in data and "video_camera" in data:
        ctx = torch.as_tensor(data["context_camera"], dtype=torch.float32)
        tgt = torch.as_tensor(data["video_camera"], dtype=torch.float32)
        if ctx.shape == (20, 12) and tgt.shape == (20, 12):
            return ctx, tgt
        if ctx.shape == (77, 4, 4) and tgt.shape == (77, 4, 4):
            return _compute_relative_20x12_from_77x4x4(ctx, tgt)
        raise ValueError(f"Unsupported context/video_camera shapes: {tuple(ctx.shape)} / {tuple(tgt.shape)}")

    raw_matrices: List[torch.Tensor] = []
    for i in range(149):
        key = str(i)
        if key in data:
            mat = _parse_camera_matrix(data[key])
        else:
            mat = raw_matrices[-1] if raw_matrices else torch.eye(4)
        raw_matrices.append(mat)

    if len(raw_matrices) > 1:
        raw_matrices[0] = raw_matrices[1]
    raw_matrices.append(raw_matrices[-1])  # 149 -> 150

    context_mats = raw_matrices[:75]
    target_mats = raw_matrices[75:]

    def pad_mats(mats: List[torch.Tensor]) -> List[torch.Tensor]:
        return [mats[0]] + mats + [mats[-1]]

    context_final = pad_mats(context_mats)  # 77
    target_final = pad_mats(target_mats)  # 77

    cam_idx = list(range(77))[::4]  # 20

    ref_c2w = _apply_coordinate_transform(target_final[0])
    ref_w2c = torch.inverse(ref_c2w)

    def to_rel(mats_final: List[torch.Tensor]) -> torch.Tensor:
        rel_list: List[torch.Tensor] = []
        for idx in cam_idx:
            c2w = _apply_coordinate_transform(mats_final[idx])
            rel = ref_w2c @ c2w
            rel_list.append(rel[:3, :4].contiguous().view(-1))
        return torch.stack(rel_list, dim=0)  # [20,12]

    cam_video_rel = to_rel(target_final)
    cam_context_rel = to_rel(context_final)
    return cam_context_rel, cam_video_rel


def build_pipeline(
    device: str,
    ckpt_path: str,
    base_dit_path: str,
    base_text_encoder_path: str,
    base_vae_path: str,
    hydra: bool,
) -> HyDRAPipeline:
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_manager.load_models([base_dit_path, base_text_encoder_path, base_vae_path])
    pipe = HyDRAPipeline.from_model_manager(model_manager, device=device)

    dim = pipe.dit.blocks[0].self_attn.q.weight.shape[0]
    num_heads = pipe.dit.blocks[0].self_attn.num_heads

    for block in pipe.dit.blocks:
        block.self_attn.tokenizer = MemoryTokenizer(dim)
        block.cam_encoder_con = nn.Linear(12, dim)
        block.cam_encoder_tgt = nn.Linear(12, dim)
        block.projector = nn.Linear(dim, dim)
        block.cam_encoder_con.weight.data.zero_()
        block.cam_encoder_con.bias.data.zero_()
        block.cam_encoder_tgt.weight.data.zero_()
        block.cam_encoder_tgt.bias.data.zero_()
        block.projector.weight = nn.Parameter(torch.eye(dim))
        block.projector.bias = nn.Parameter(torch.zeros(dim))

        block.self_attn.hydra = bool(hydra)
        block.self_attn.change_sparse = block.self_attn.hydra
        if hydra:
            block.self_attn.attention_type = "sparse_frame"
            block.self_attn.attn = DynamicRetrievalAttention(
                num_heads=num_heads,
                num_frames=40,
                frame_hw=(30 * 52),
                top_k=9,
                frame_chunk_size=None,
            )

    state = torch.load(ckpt_path, map_location="cpu")
    pipe.dit.load_state_dict(state, strict=True)
    pipe.to(device)
    pipe.to(dtype=torch.bfloat16)
    return pipe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HyDRA inference: concat condition video + prediction")
    p.add_argument("--examples_dir", type=str, default="./examples", help="Run batch inference over examples folder (expects 1.mp4.. and cam1.json.. + caption.txt).")
    p.add_argument("--cond_video", type=str, default=None, help="Condition video path (single mode).")
    p.add_argument("--cond_json", type=str, default=None, help="Camera json path (single mode).")
    p.add_argument("--caption_txt", type=str, default=None, help="Caption txt path (single mode).")
    p.add_argument("--ckpt_path", type=str, default="./ckpts/hydra.ckpt", help="Finetuned HyDRA checkpoint (state_dict).")
    p.add_argument("--output_path", type=str, default="./outputs", help="Output dir (batch) or mp4 path (single).")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--cond_frames", type=int, default=77, help="How many frames to read from condition video.")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hydra", action="store_true", default=True, help="Enable HyDRA (Dynamic Retrieval Attention).")
    p.add_argument("--base_dit_path", type=str, default="./ckpts/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
    p.add_argument("--base_text_encoder_path", type=str, default="./ckpts/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth")
    p.add_argument("--base_vae_path", type=str, default="./ckpts/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    return p.parse_args()


def _infer_one(
    pipe: HyDRAPipeline,
    cond_video_path: str,
    cam_json_path: str,
    prompt: str,
    out_path: str,
    height: int,
    width: int,
    cond_frames: int,
    cfg_scale: float,
    num_inference_steps: int,
    seed: int,
    fps: int,
) -> None:
    cond_video = load_condition_video(cond_video_path, cond_frames, height, width)
    ref_camera, target_camera = load_condition_json(cam_json_path)
    with torch.no_grad():
        pred_frames, _ = pipe(
            prompt=prompt,
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            source_video=cond_video.to(dtype=pipe.torch_dtype, device=pipe.device),
            target_camera=target_camera.to(dtype=pipe.torch_dtype, device=pipe.device),
            ref_camera=ref_camera.to(dtype=pipe.torch_dtype, device=pipe.device),
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            height=height,
            width=width,
            num_frames=77,
            tiled=True,
        )
    cond_frames_list = pipe.tensor2video(cond_video)
    out_frames = list(cond_frames_list) + list(pred_frames)
    save_video(out_frames, out_path, fps=fps, quality=5)


def main() -> None:
    args = parse_args()

    pipe = build_pipeline(
        device=args.device,
        ckpt_path=args.ckpt_path,
        base_dit_path=args.base_dit_path,
        base_text_encoder_path=args.base_text_encoder_path,
        base_vae_path=args.base_vae_path,
        hydra=args.hydra,
    )

    if args.examples_dir is not None:
        examples_dir = os.path.abspath(args.examples_dir)
        if not os.path.isdir(examples_dir):
            raise FileNotFoundError(f"examples_dir not found: {examples_dir}")
        out_dir = args.output_path
        os.makedirs(out_dir, exist_ok=True)

        caption_path = os.path.join(examples_dir, "caption.txt")
        with open(caption_path, "r", encoding="utf-8") as f:
            captions = [line.rstrip("\n") for line in f.readlines()]

        mp4s = [f for f in os.listdir(examples_dir) if f.lower().endswith(".mp4")]

        def _sort_key(name: str) -> Tuple[int, str]:
            base = os.path.splitext(name)[0]
            try:
                return int(base), name
            except Exception:
                return 1_000_000_000, name

        for mp4_name in sorted(mp4s, key=_sort_key):
            base = os.path.splitext(mp4_name)[0]
            try:
                sample_id = int(base)
                caption_idx = sample_id - 1
            except Exception:
                sample_id = 0
                caption_idx = 0
            if caption_idx < 0 or caption_idx >= len(captions):
                raise ValueError(f"caption.txt must have a line for {mp4_name} (index {caption_idx}).")
            prompt = captions[caption_idx].strip()
            if not prompt:
                raise ValueError(f"Empty caption at line {caption_idx+1} for {mp4_name}.")

            cond_video_path = os.path.join(examples_dir, mp4_name)
            cam_json_path = os.path.join(examples_dir, f"cam{sample_id}.json")
            if not os.path.exists(cam_json_path):
                raise FileNotFoundError(f"camera json not found: {cam_json_path}")

            out_path = os.path.join(out_dir, f"{sample_id}_concat.mp4")
            _infer_one(
                pipe=pipe,
                cond_video_path=cond_video_path,
                cam_json_path=cam_json_path,
                prompt=prompt,
                out_path=out_path,
                height=args.height,
                width=args.width,
                cond_frames=args.cond_frames,
                cfg_scale=args.cfg_scale,
                num_inference_steps=args.num_inference_steps,
                seed=args.seed,
                fps=args.fps,
            )
        return

    if args.cond_video is None or args.cond_json is None or args.caption_txt is None:
        raise ValueError("Provide --examples_dir for batch mode, or --cond_video/--cond_json/--caption_txt for single mode.")

    with open(args.caption_txt, "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    if not prompt:
        raise ValueError("caption_txt is empty.")

    if args.output_path.lower().endswith(".mp4"):
        out_path = args.output_path
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    else:
        os.makedirs(args.output_path, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.cond_video))[0]
        out_path = os.path.join(args.output_path, f"{base}_concat.mp4")

    _infer_one(
        pipe=pipe,
        cond_video_path=args.cond_video,
        cam_json_path=args.cond_json,
        prompt=prompt,
        out_path=out_path,
        height=args.height,
        width=args.width,
        cond_frames=args.cond_frames,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
