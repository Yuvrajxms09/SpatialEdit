#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from dense_vggt import evaluate_one_vggt, load_vggt


# def find_triplets(root: Path, src_suffix="_src.png", gt_suffix="_gt.png", pred_suffix=".png"):
#     triplets = []
#     for gt_path in root.rglob(f"*{gt_suffix}"):
#         if not gt_path.is_file():
#             continue
#         base = gt_path.name[:-len(gt_suffix)]
#         if not base:
#             continue
#         src_path = gt_path.with_name(base + src_suffix)
#         pred_path = gt_path.with_name(base + pred_suffix)
#         if src_path.exists() and pred_path.exists():
#             triplets.append((src_path, gt_path, pred_path))
#     return triplets

def find_triplets(root: Path, src_suffix="_src.png", gt_suffix="_gt.png", pred_suffix=".png", meta_data_file=""):
    # 1️⃣ 读取 JSON，提取 edit_id
    with open(meta_data_file, "r") as f:
        data = json.load(f)
    # 假设 JSON 是 list[dict]
    edit_id_set = set()
    for item in data:
        if "edit_id" in item:
            edit_id_set.add(str(item["edit_id"]))  # 转成 str，避免类型不一致
    # 2️⃣ 遍历文件并过滤
    triplets = []
    root = Path(root)
    for gt_path in root.rglob(f"*{gt_suffix}"):
        if not gt_path.is_file():
            continue
        base = gt_path.name[:-len(gt_suffix)]
        if not base:
            continue
        # 🔥 核心过滤逻辑
        if base not in edit_id_set:
            continue
        src_path = gt_path.with_name(base + src_suffix)
        pred_path = gt_path.with_name(base + pred_suffix)
        if src_path.exists() and pred_path.exists():
            triplets.append((src_path, gt_path, pred_path))

    return triplets


def init_distributed():
    """
    torchrun 会自动设置这些环境变量：
      RANK, WORLD_SIZE, LOCAL_RANK
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", init_method="env://")
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def shard_keys(keys, rank, world_size):
    """
    “每张卡测评一个key对应的路径”的最简单实现：
    - 如果 keys >= world_size：按 rank 分配 1 个或多个 key（round-robin）
    - 如果 keys < world_size：只有前 len(keys) 个 rank 有活干
    """
    return [k for i, k in enumerate(keys) if (i % world_size) == rank]


def main():
    import os

    ap = argparse.ArgumentParser("Torchrun batch eval multiple model folders with VGGT evaluate_one_vggt")
    ap.add_argument("--models_json", type=str, required=True,
                    help='JSON dict: {"model_name":"pred_path", ...}')
    ap.add_argument("--vggt_ckpt", type=str, required=True)
    ap.add_argument("--meta_data_file", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default=None)

    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])

    ap.add_argument("--src_suffix", type=str, default="_src.png")
    ap.add_argument("--gt_suffix", type=str, default="_gt.png")
    ap.add_argument("--pred_suffix", type=str, default=".png")

    args = ap.parse_args()

    # ---------- distributed init ----------
    is_dist, rank, world_size, local_rank = (False, 0, 1, 0)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        is_dist = True
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", init_method="env://")

    # ---------- device ----------
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    if args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    # ---------- parse models dict ----------
    models = json.loads(args.models_json)
    if not isinstance(models, dict) or not models:
        raise ValueError("models_json must be a non-empty JSON dict")

    keys = list(models.keys())
    my_keys = shard_keys(keys, rank, world_size)

    # 如果这个 rank 没有分到 key，直接参与 gather 并退出
    if len(my_keys) == 0:
        local_rows = []
        if is_dist:
            gathered = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, local_rows)
            if rank == 0:
                all_rows = []
                for part in gathered:
                    if part:
                        all_rows.extend(part)
                df = pd.DataFrame(all_rows).sort_values("gt_ypr_err", ascending=True, na_position="last")
                if args.out_csv:
                    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(args.out_csv, index=False, encoding="utf-8")
                    print(f"[DONE] saved: {args.out_csv}")
                print(df)
        return

    # ---------- load VGGT once per rank ----------
    vggt = load_vggt(args.vggt_ckpt, device=device)

    # ---------- eval assigned keys ----------
    local_rows = []
    for model_name in my_keys:
        pred_root = models[model_name]
        root = Path(pred_root)

        triplets = find_triplets(
            root,
            src_suffix=args.src_suffix,
            gt_suffix=args.gt_suffix,
            pred_suffix=args.pred_suffix,
            meta_data_file=args.meta_data_file
        )

        n_total = len(triplets)
        if n_total == 0:
            local_rows.append({
                "model_name": model_name,
                "gt_xyz_err": float("nan"),
                "gt_ypr_err": float("nan"),
                "overall_VE_error": float("nan"),
                "n_ok": 0,
                "n_total": 0,
                "rank": rank,
            })
            continue

        xyz_list = []
        ypr_list = []
        n_ok = 0

        for src_path, gt_path, pred_path in triplets:
            try:
                m = evaluate_one_vggt(
                    model=vggt,
                    src_path=src_path,
                    gt_path=gt_path,
                    pred_path=pred_path,
                    device=device,
                    dtype=dtype,
                )
                xyz = m.get("gt_pred_xyz_err", None)
                ypr = m.get("gt_pred_ypr_err", None)
                if xyz is None or ypr is None:
                    continue
                xyz_list.append(float(xyz))
                ypr_list.append(float(ypr))
                n_ok += 1
            except Exception:
                continue

        gt_xyz_err = float(np.mean(xyz_list)) if xyz_list else float("nan")
        gt_ypr_err = float(np.mean(ypr_list)) if ypr_list else float("nan")
        overall_error = (gt_xyz_err + gt_ypr_err) / 2.0
        local_rows.append({
            "model_name": model_name,
            "gt_xyz_err": gt_xyz_err,
            "gt_ypr_err": gt_ypr_err,
            "overall_VE_error": overall_error,
            "n_ok": n_ok,
            "n_total": n_total,
            "rank": rank,
        })

    # ---------- gather to rank0 ----------
    if is_dist:
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_rows)

        if rank == 0:
            all_rows = []
            for part in gathered:
                if part:
                    all_rows.extend(part)

            df = pd.DataFrame(all_rows).sort_values("gt_ypr_err", ascending=True, na_position="last").reset_index(drop=True)

            if args.out_csv:
                Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(args.out_csv, index=False, encoding="utf-8")
                print(f"[DONE] saved: {args.out_csv}")

            print(df)

        dist.barrier()
        dist.destroy_process_group()
    else:
        df = pd.DataFrame(local_rows).sort_values("gt_ypr_err", ascending=True, na_position="last").reset_index(drop=True)
        if args.out_csv:
            Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.out_csv, index=False, encoding="utf-8")
            print(f"[DONE] saved: {args.out_csv}")
        print(df)


if __name__ == "__main__":
    main()