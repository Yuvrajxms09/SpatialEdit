
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import math
import numpy as np
import pandas as pd
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


# -------------------------
# helpers
# -------------------------
def wrap_deg(d: float) -> float:
    """wrap to [-180, 180)"""
    return (d + 180.0) % 360.0 - 180.0


def rotmat_to_ypr_deg(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to yaw-pitch-roll in degrees.
    Convention: ZYX (yaw around Z, pitch around Y, roll around X).
    R is 3x3.
    """
    # pitch = asin(-r20)
    r20 = float(R[2, 0])
    r20 = max(-1.0, min(1.0, r20))
    pitch = math.asin(-r20)

    # handle gimbal lock
    if abs(abs(r20) - 1.0) < 1e-6:
        # roll set 0, yaw from r01/r11
        roll = 0.0
        yaw = math.atan2(-float(R[0, 1]), float(R[1, 1]))
    else:
        roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
        yaw  = math.atan2(float(R[1, 0]), float(R[0, 0]))

    return np.array([yaw, pitch, roll], dtype=np.float32) * (180.0 / math.pi)


def w2c_to_cam_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    For w2c: x_cam = R x_world + t
    camera center in world: C = -R^T t
    """
    return -R.T @ t


def ypr_error_deg(R_ref: np.ndarray, R_pred: np.ndarray) -> float:
    """
    Angular error as L2 norm of wrapped (yaw,pitch,roll) deltas between ref and pred.
    delta rotation: R_delta = R_pred * R_ref^T
    """
    R_delta = R_pred @ R_ref.T
    ypr = rotmat_to_ypr_deg(R_delta)
    y, p, r = (wrap_deg(float(ypr[0])), wrap_deg(float(ypr[1])), wrap_deg(float(ypr[2])))
    return float(math.sqrt(y*y + p*p + r*r))

def rotation_geodesic_deg(R1, R2):
    R_delta = R1.T @ R2
    cos_theta = (np.trace(R_delta) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return float(np.degrees(theta))

# -------------------------
# VGGT: load once
# -------------------------
def load_vggt(model_path: str, device: str):
    model = VGGT()
    sd = torch.load(model_path, map_location="cpu")
    model.load_state_dict(sd)
    model.eval().to(device)
    return model


# -------------------------
# evaluate_one like YOLO version, but with VGGT poses
# -------------------------
@torch.no_grad()
def evaluate_one_vggt(
    model: VGGT,
    src_path: Path,
    gt_path: Path,
    pred_path: Path,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> dict:
    """
    Returns:
      - src_pred_xyz_err, src_pred_ypr_err
      - gt_pred_xyz_err,  gt_pred_ypr_err
    """
    # VGGT expects preprocessed tensor batch
    images = load_and_preprocess_images([str(src_path), str(gt_path), str(pred_path)]).to(device)

    with torch.cuda.amp.autocast(enabled=(device.startswith("cuda")), dtype=dtype):
        predictions = model(images)

    # decode pose encoding -> extrinsic (w2c) (B, N, 3, 4) or similar; we use [0]
    with torch.cuda.amp.autocast(enabled=(device.startswith("cuda")), dtype=torch.float64):
        extrinsic, _intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        ext = extrinsic[0]  # (3, 3, 4) for 3 images

    # split
    ext_np = ext.detach().cpu().numpy()  # (3, 3, 4)
    def split_rt(e34):
        R = e34[:, :3]
        t = e34[:, 3]
        return R.astype(np.float64), t.astype(np.float64)

    R_src, t_src   = split_rt(ext_np[0])
    R_gt,  t_gt    = split_rt(ext_np[1])
    R_pred, t_pred = split_rt(ext_np[2])

    # camera centers (world)
    C_src  = w2c_to_cam_center(R_src, t_src)
    C_gt   = w2c_to_cam_center(R_gt,  t_gt)
    C_pred = w2c_to_cam_center(R_pred, t_pred)

    # xyz errors
    # src_pred_xyz_err = float(np.linalg.norm(C_pred - C_src))
    # gt_pred_xyz_err  = float(np.linalg.norm(C_pred - C_gt))

    # xyz errors (baseline-normalized)
    baseline = float(np.linalg.norm(C_gt - C_src)) + 1e-8
    src_pred_xyz_err = float(np.linalg.norm(C_pred - C_src) / baseline)
    gt_pred_xyz_err  = float(np.linalg.norm(C_pred - C_gt)  / baseline)

    # yaw-pitch-roll errors (deg)
    src_pred_ypr_err = rotation_geodesic_deg(R_src, R_pred) /90.
    gt_pred_ypr_err  = rotation_geodesic_deg(R_gt,  R_pred) /90.

    return {
        "status": "ok",
        "src_pred_xyz_err": src_pred_xyz_err,
        "src_pred_ypr_err": src_pred_ypr_err,
        "gt_pred_xyz_err": gt_pred_xyz_err,
        "gt_pred_ypr_err": gt_pred_ypr_err,
    }


# -------------------------
# main: compare all candidates to a reference
# -------------------------
def main():
    ap = argparse.ArgumentParser("VGGT view consistency eval vs reference")
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--prefix", type=str, default="Bon_Teapot_Ceramic")
    ap.add_argument("--ref_name", type=str, default="Bon_Teapot_Ceramic_y000_p+33_d0097.png")
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default="vggt_view_consistency.csv")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--exclude_ref", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)

    # find ref (allow passing only filename)
    ref_path = Path(args.ref_name)
    if not ref_path.exists():
        found = None
        for p in root.rglob(args.ref_name):
            if p.is_file():
                found = p
                break
        if found is None:
            raise FileNotFoundError(f"Reference not found under {root}: {args.ref_name}")
        ref_path = found

    # candidates
    cands = []
    for p in sorted(root.rglob("*.png")):
        if not p.is_file():
            continue
        if not p.name.startswith(args.prefix):
            continue
        if args.exclude_ref and p.resolve() == ref_path.resolve():
            continue
        cands.append(p)
    if not cands:
        raise RuntimeError(f"No candidates found under {root} with prefix {args.prefix}")

    # dtype
    if args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    model = load_vggt(args.model_path, device=device)

    rows = []
    for cand in cands:
        try:
            # 关键：ref 同时当 src 和 gt，这样 gt_pred 就是 ref<->cand 的误差（视角一致性）
            m = evaluate_one_vggt(
                model=model,
                src_path=ref_path,
                gt_path=ref_path,
                pred_path=cand,
                device=device,
                dtype=dtype,
            )
            rows.append({
                "filename": str(cand.relative_to(root)),
                "gt_xyz_err": m["gt_pred_xyz_err"],
                "gt_ypr_err": m["gt_pred_ypr_err"],
            })
        except Exception:
            rows.append({
                "filename": str(cand.relative_to(root)),
                "gt_xyz_err": float("nan"),
                "gt_ypr_err": float("nan"),
            })

    df = pd.DataFrame(rows).sort_values(["gt_ypr_err", "gt_xyz_err"], ascending=True, na_position="last").reset_index(drop=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(df.head(30))
    print(f"[DONE] ref={ref_path}  saved={args.out_csv}  n={len(df)}")


if __name__ == "__main__":
    main()