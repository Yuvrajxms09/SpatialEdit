#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.distributed as dist
from ultralytics import YOLO

# ONLY import from your metric module
from FE_metric import evaluate_one

# -----------------------------
# DDP setup
# -----------------------------
def ddp_setup() -> Tuple[int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank, rank, world_size


def ddp_cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


# -----------------------------
# JSON / filesystem utilities
# -----------------------------
def load_datasets_items(datasets_json: Optional[str], datasets_path: Optional[str]) -> List[Tuple[str, str]]:
    if datasets_path:
        with open(datasets_path, "r", encoding="utf-8") as f:
            mp = json.load(f)
        items = list(mp.items())
        if not items:
            raise ValueError("Empty datasets map.")
        return items

    if not datasets_json:
        raise ValueError("Provide --datasets_json or --datasets_path")

    s = datasets_json.strip()

    # strict JSON first
    try:
        mp = json.loads(s)
        items = list(mp.items())
        if not items:
            raise ValueError("Empty datasets map.")
        return items
    except Exception:
        pass

    # fallback: python-literal dict (single quotes etc.)
    try:
        import ast
        mp = ast.literal_eval(s)
        if not isinstance(mp, dict):
            raise ValueError("datasets_json is not a dict after literal_eval")
        items = list(mp.items())
        if not items:
            raise ValueError("Empty datasets map.")
        return items
    except Exception as e:
        raise ValueError(
            "Failed to parse --datasets_json. Use strict JSON (double quotes) or --datasets_path eval_data.json"
        ) from e


def read_cmd_from_json(json_path: Path) -> Tuple[float, float, float]:
    with open(json_path, "r", encoding="utf-8") as f:
        js = json.load(f)
    meta = js.get("metadata", {})
    ypd = meta.get("edit_ypd", {})
    dyaw = float(ypd.get("yaw", 0.0))
    dpitch = float(ypd.get("pitch", 0.0))
    ddist = float(ypd.get("distance", 0.0))
    return dyaw, dpitch, ddist


def find_prefixes_by_gt(folder: Path, gt_suffix: str) -> List[str]:
    prefixes = set()
    for p in folder.rglob("*"):
        if not p.is_file():
            continue
        if not p.name.lower().endswith(gt_suffix.lower()):
            continue
        base = p.name[:-len(gt_suffix)]
        if not base:
            continue
        rel_dir = p.parent.relative_to(folder)
        pref = str(rel_dir / base) if str(rel_dir) != "." else base
        prefixes.add(pref)
    return sorted(prefixes)


def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


# -----------------------------
# CSV summary
# -----------------------------
def summarize_csv(path: Path, min_matches: int = 2, only_status_ok: bool = False) -> Dict[str, Any]:
    df = pd.read_csv(path)
    if only_status_ok:
        df = df[df["status"] == "ok"]

    n = len(df)
    if n == 0:
        return {"image_Num": 0, "note": "No rows after filtering."}

    for c in ["gt_ray_diff_deg", "zoom_dir_err", "matches_gt", "matches_zoom", "gt_det"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "log_scale" in df.columns:
        df["log_scale"] = pd.to_numeric(df["log_scale"], errors="coerce")

    angle_success = df["gt_ray_diff_deg"].notna()
    zoom_success = df["log_scale"].notna()

    angle_error_fail20 = df["gt_ray_diff_deg"].fillna(20.0).mean()
    zoom_error_fail1 = df["zoom_dir_err"].fillna(1.0).mean()

    angle_error_exfail = df["gt_ray_diff_deg"].mean()
    zoom_error_exfail = df["zoom_dir_err"].mean()

    return {
        "angle_error_fail20": float(angle_error_fail20) if np.isfinite(angle_error_fail20) else np.nan,
        "angle_error_exfail": float(angle_error_exfail) if np.isfinite(angle_error_exfail) else np.nan,
        "angle_success_images": int(angle_success.sum()),

        "zoom_error_fail1": float(zoom_error_fail1) if np.isfinite(zoom_error_fail1) else np.nan,
        "zoom_error_exfail": float(zoom_error_exfail) if np.isfinite(zoom_error_exfail) else np.nan,
        "zoom_success_images": int(zoom_success.sum()),

        "image_Num": int(n),
        f"matches_gt_ge_{min_matches}": int((df["matches_gt"] >= min_matches).sum()) if "matches_gt" in df else 0,
        f"matches_src_ge_{min_matches}": int((df["matches_zoom"] >= min_matches).sum()) if "matches_zoom" in df else 0,
        "gt_det_lt_2": int((df["gt_det"] < 2).sum()) if "gt_det" in df else 0,
    }


# -----------------------------
# Per-dataset eval
# -----------------------------
def eval_one_dataset(
    model: YOLO,
    dataset_name: str,
    folder: Path,
    out_dir_rank: Path,
    src_suffix: str,
    gt_suffix: str,
    pred_suffix: str,
    conf: float,
    imgsz: int,
    device: str,
    topk_per_class: int,
    max_match_angle_gt: float,
    max_match_angle_zoom: float,
    zoom_margin: float,
    min_matches: int,
    only_status_ok: bool,
    meta_data_file: str
) -> Dict[str, Any]:

    if not folder.exists():
        return {"model_name": dataset_name, "folder": str(folder), "error": "missing_folder"}

    prefixes = find_prefixes_by_gt(folder, gt_suffix)
    if not prefixes:
        return {"model_name": dataset_name, "folder": str(folder), "error": f"no_gt_files(*{gt_suffix})"}

    with open(meta_data_file, "r") as f:
        data = json.load(f)
    edit_id_set = set()
    for item in data:
        if "edit_id" in item:
            edit_id_set.add(str(item["edit_id"]))
    rows: List[Dict[str, Any]] = []
    for pref in prefixes:
        item_id = pref.split('/')[-1]
        if item_id not in edit_id_set:
            continue
        json_path = folder / f"{pref}.json"
        src_path  = folder / f"{pref}{src_suffix}"
        gt_path   = folder / f"{pref}{gt_suffix}"
        pred_path = folder / f"{pref}{pred_suffix}"
        if not (json_path.exists() and src_path.exists() and gt_path.exists() and pred_path.exists()):
            continue

        cmd_dyaw, cmd_dpitch, cmd_ddist = read_cmd_from_json(json_path)

        m = evaluate_one(
            model=model,
            src_path=src_path,
            gt_path=gt_path,
            pred_path=pred_path,
            cmd_ddist=cmd_ddist,
            conf=conf,
            imgsz=imgsz,
            device=device,
            topk_per_class=topk_per_class,
            max_match_angle_gt=max_match_angle_gt,
            max_match_angle_zoom=max_match_angle_zoom,
            zoom_margin=zoom_margin,
        )

        rows.append({
            "prefix": pref,
            "status": m.get("status", "unk"),
            "cmd_dyaw": cmd_dyaw,
            "cmd_dpitch": cmd_dpitch,
            "cmd_ddist": cmd_ddist,

            "gt_ray_diff_deg": m.get("gt_ray_diff_deg"),
            "gt_pairwise_diff_deg": m.get("gt_pairwise_diff_deg"),
            "matches_gt": m.get("matches_gt", 0),

            "zoom_dir_err": m.get("zoom_dir_err"),
            "log_scale": m.get("log_scale"),
            "matches_zoom": m.get("matches_zoom", 0),

            "src_det": m.get("src_det", 0),
            "gt_det": m.get("gt_det", 0),
            "pred_det": m.get("pred_det", 0),
        })

    out_dir_rank.mkdir(parents=True, exist_ok=True)
    per_csv = out_dir_rank / f"{dataset_name}.csv"
    fieldnames = [
        "prefix", "status",
        "cmd_dyaw", "cmd_dpitch", "cmd_ddist",
        "gt_ray_diff_deg", "gt_pairwise_diff_deg", "matches_gt",
        "zoom_dir_err", "log_scale", "matches_zoom",
        "src_det", "gt_det", "pred_det",
    ]
    with open(per_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in fieldnames})

    summ = summarize_csv(per_csv, min_matches=min_matches, only_status_ok=only_status_ok)

    angle_error = float(summ.get("angle_error_fail20", np.nan))
    zoom_error = float(summ.get("zoom_error_fail1", np.nan))
    angle_norm = clip01(angle_error / 20.0) if np.isfinite(angle_error) else np.nan
    zoom_norm = clip01(zoom_error) if np.isfinite(zoom_error) else np.nan
    overall_norm = (angle_norm + zoom_norm) / 2.0 if (np.isfinite(angle_norm) and np.isfinite(zoom_norm)) else np.nan

    return {
        "model_name": dataset_name,
        "folder": str(folder),
        "n_images": int(summ.get("image_Num", 0)),
        "angle_error": angle_error,
        "zoom_error": zoom_error,
        "angle_norm": angle_norm,
        "zoom_norm": zoom_norm,
        "overall_FE_error": overall_norm,
        "angle_success": int(summ.get("angle_success_images", 0)),
        "zoom_success": int(summ.get("zoom_success_images", 0)),
        f"matches_gt_ge_{min_matches}": int(summ.get(f"matches_gt_ge_{min_matches}", 0)),
        f"matches_src_ge_{min_matches}": int(summ.get(f"matches_src_ge_{min_matches}", 0)),
        "gt_det_lt_2": int(summ.get("gt_det_lt_2", 0)),
        "rows_written": len(rows),
        "per_sample_csv": str(per_csv),
    }


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser("DDP test-only yolo_metric eval (free assignment)")
    ap.add_argument("--yolo_model", type=str, required=True)
    ap.add_argument("--datasets_json", type=str, default=None)
    ap.add_argument("--datasets_path", type=str, default=None)
    ap.add_argument("--meta_data_file", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--src_suffix", type=str, default="_src.png")
    ap.add_argument("--gt_suffix", type=str, default="_gt.png")
    ap.add_argument("--pred_suffix", type=str, default=".png")

    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--topk_per_class", type=int, default=3)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--max_match_angle_gt", type=float, default=60.0)
    ap.add_argument("--max_match_angle_zoom", type=float, default=70.0)
    ap.add_argument("--zoom_margin", type=float, default=0.08)

    ap.add_argument("--min_matches", type=int, default=2)
    ap.add_argument("--only_status_ok", action="store_true")

    args = ap.parse_args()

    local_rank, rank, world_size = ddp_setup()

    try:
        items = load_datasets_items(args.datasets_json, args.datasets_path)
        num_datasets = len(items)

        out_root = Path(args.out_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        out_rank = out_root / f"rank{rank:02d}_gpu{local_rank:02d}"
        out_rank.mkdir(parents=True, exist_ok=True)

        # -------- FREE ASSIGNMENT (round-robin) --------
        my_items = [items[i] for i in range(rank, num_datasets, world_size)]
        print(f"[R{rank}/{world_size}] local_rank={local_rank} assigned {len(my_items)} dataset(s)")

        # One YOLO model per rank (reused for multiple datasets on the same GPU)
        model = YOLO(args.yolo_model)

        my_results: List[Dict[str, Any]] = []
        for name, folder_str in my_items:
            folder = Path(folder_str)
            print(f"[R{rank}/{world_size}] -> {name}: {folder}")
            res = eval_one_dataset(
                model=model,
                dataset_name=name,
                folder=folder,
                out_dir_rank=out_rank,
                src_suffix=args.src_suffix,
                gt_suffix=args.gt_suffix,
                pred_suffix=args.pred_suffix,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                topk_per_class=args.topk_per_class,
                max_match_angle_gt=args.max_match_angle_gt,
                max_match_angle_zoom=args.max_match_angle_zoom,
                zoom_margin=args.zoom_margin,
                min_matches=args.min_matches,
                only_status_ok=args.only_status_ok,
                meta_data_file=args.meta_data_file
            )
            res["assigned_rank"] = rank
            res["assigned_local_rank"] = local_rank
            my_results.append(res)

        # Gather lists from all ranks
        gathered: List[Optional[List[Dict[str, Any]]]] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, my_results)
        dist.barrier()

        if rank == 0:
            all_rows: List[Dict[str, Any]] = []
            for part in gathered:
                if not part:
                    continue
                all_rows.extend(part)

            df = pd.DataFrame(all_rows)
            if len(df) > 0 and "overall_FE_error" in df.columns:
                df = df.sort_values(["overall_FE_error", "angle_norm", "zoom_norm"], ascending=True).reset_index(drop=True)

            out_csv = out_root / "summary_models.csv"
            df.to_csv(out_csv, index=False)

            cols = [c for c in [
                "model_name", "angle_error", "zoom_error",
                "angle_norm", "zoom_norm", "overall_FE_error",
                "n_images", "angle_success", "zoom_success",
                "rows_written", "assigned_rank", "per_sample_csv"
            ] if c in df.columns]

            print("\n===== SUMMARY (rank0) =====")
            if len(df) == 0:
                print("No results gathered. Check paths/suffixes.")
            else:
                print(df[cols].to_string(index=False))
                print(f"\n[WROTE] {out_csv}")
            print("===========================\n")

    finally:
        ddp_cleanup()


if __name__ == "__main__":
    main()