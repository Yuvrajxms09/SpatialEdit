#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import re
import json
import math
import csv
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import cv2
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment


def classify_edit(cmd_dyaw: float, cmd_dpitch: float, cmd_ddist: float, eps: float = 1e-6) -> str:
    """
    Returns: "angle", "zoom", "mixed", "none"
    - angle: yaw/pitch changes, distance ~ 0
    - zoom : distance changes, yaw/pitch ~ 0
    - mixed: both changed
    - none : all ~ 0
    """
    has_angle = (abs(cmd_dyaw) > eps) or (abs(cmd_dpitch) > eps)
    has_zoom = abs(cmd_ddist) > eps
    if has_angle and not has_zoom:
        return "angle"
    if has_zoom and not has_angle:
        return "zoom"
    if has_angle and has_zoom:
        return "mixed"
    return "none"

# ----------------------------
# Utils: concat viz
# ----------------------------
def hconcat_three(img_a: np.ndarray, img_b: np.ndarray, img_c: np.ndarray, interp=cv2.INTER_AREA) -> np.ndarray:
    def resize_to_h(img, H):
        h, w = img.shape[:2]
        if h == H:
            return img
        new_w = int(round(w * (H / float(h))))
        return cv2.resize(img, (new_w, H), interpolation=interp)

    H = min(img_a.shape[0], img_b.shape[0], img_c.shape[0])
    a = resize_to_h(img_a, H)
    b = resize_to_h(img_b, H)
    c = resize_to_h(img_c, H)
    return cv2.hconcat([a, b, c])


# ----------------------------
# Rays from bbox centers
# ----------------------------
def center_to_ray(u: float, v: float, w: int, h: int, f: Optional[float] = None) -> np.ndarray:
    cx, cy = 0.5 * w, 0.5 * h
    if f is None:
        f = 0.9 * max(w, h)
    x = (u - cx) / f
    y = (v - cy) / f
    d = np.array([x, y, 1.0], dtype=np.float32)
    d /= (np.linalg.norm(d) + 1e-9)
    return d

def ang_between(u: np.ndarray, v: np.ndarray) -> float:
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return math.degrees(math.acos(dot))


# ----------------------------
# YOLO detection extraction
# ----------------------------
def extract_det(r, conf_thres: float = 0.25, topk_per_class: int = 3, prefer_large: bool = True):
    if r.boxes is None or len(r.boxes) == 0:
        return []

    xyxy = r.boxes.xyxy.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    cls  = r.boxes.cls.cpu().numpy().astype(int)

    dets = []
    for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
        if float(c) < conf_thres:
            continue
        u = 0.5 * (x1 + x2)
        v = 0.5 * (y1 + y2)
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        area = bw * bh
        dets.append({
            "cls": int(k),
            "conf": float(c),
            "center": (float(u), float(v)),
            "wh": (float(bw), float(bh)),
            "area": float(area),
        })

    by_cls: Dict[int, List[dict]] = {}
    for d in dets:
        by_cls.setdefault(d["cls"], []).append(d)

    out = []
    for k, lst in by_cls.items():
        if prefer_large:
            lst.sort(key=lambda x: x["area"], reverse=True)
        else:
            lst.sort(key=lambda x: x["conf"], reverse=True)
        out.extend(lst[:topk_per_class])

    return out

def build_rays(dets, img_w: int, img_h: int, f: Optional[float] = None):
    objs = []
    for d in dets:
        u, v = d["center"]
        ray = center_to_ray(u, v, img_w, img_h, f=f)
        objs.append({**d, "ray": ray})
    return objs


# ----------------------------
# Matching (per class Hungarian)
# ----------------------------
def match_by_class(ref_objs, cur_objs, max_angle_deg: float = 60.0, lambda_area: float = 2.0):
    matches = []

    ref_by = {}
    cur_by = {}
    for i, o in enumerate(ref_objs):
        ref_by.setdefault(o["cls"], []).append(i)
    for j, o in enumerate(cur_objs):
        cur_by.setdefault(o["cls"], []).append(j)

    shared = sorted(set(ref_by.keys()) & set(cur_by.keys()))
    for k in shared:
        ri = ref_by[k]
        cj = cur_by[k]
        if not ri or not cj:
            continue

        C = np.zeros((len(ri), len(cj)), dtype=np.float32)
        A = np.zeros_like(C)

        for a, i in enumerate(ri):
            for b, j in enumerate(cj):
                ang = ang_between(ref_objs[i]["ray"], cur_objs[j]["ray"])
                ar = ref_objs[i]["area"] / max(cur_objs[j]["area"], 1e-6)
                area_cost = abs(math.log(ar))
                cost = ang + lambda_area * area_cost
                C[a, b] = cost
                A[a, b] = ang

        row_ind, col_ind = linear_sum_assignment(C)
        for a, b in zip(row_ind, col_ind):
            ang = float(A[a, b])
            cost = float(C[a, b])
            if ang <= max_angle_deg:
                matches.append((ri[a], cj[b], ang, cost))

    return matches


# ----------------------------
# GT angle similarity metrics (GT <-> PRED)
# ----------------------------
def pairwise_layout_matrix(objs, idxs):
    rays = [objs[i]["ray"] for i in idxs]
    n = len(rays)
    if n < 2:
        return None
    M = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            a = ang_between(rays[i], rays[j])
            M[i, j] = M[j, i] = a
    return M

def pairwise_layout_diff_deg(M0, M1):
    if M0 is None or M1 is None:
        return None
    if M0.shape != M1.shape:
        return None
    n = M0.shape[0]
    if n < 2:
        return None
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(abs(float(M0[i, j] - M1[i, j])))
    return float(np.mean(diffs)) if diffs else None

def gt_similarity_metrics(gt_objs, pred_objs, matches):
    if len(matches) < 2:
        return None, None

    pairs = [(ri, cj) for (ri, cj, _, _) in matches]
    ray_vals = [ang_between(gt_objs[ri]["ray"], pred_objs[cj]["ray"]) for (ri, cj) in pairs]
    gt_ray_diff_deg = float(np.mean(ray_vals)) if ray_vals else None

    ref_ids = [ri for (ri, _) in pairs]
    cur_ids = [cj for (_, cj) in pairs]
    M0 = pairwise_layout_matrix(gt_objs, ref_ids)
    M1 = pairwise_layout_matrix(pred_objs, cur_ids)
    gt_pairwise_diff_deg = pairwise_layout_diff_deg(M0, M1)
    return gt_ray_diff_deg, gt_pairwise_diff_deg


# ----------------------------
# Zoom direction metric (SRC <-> PRED)
# ----------------------------
def median_log_scale_from_matches(ref_objs, cur_objs, matches, eps=1e-6):
    if len(matches) == 0:
        return None
    logs = []
    for ri, cj, _, _ in matches:
        a0 = max(ref_objs[ri]["area"], eps)
        a1 = max(cur_objs[cj]["area"], eps)
        logs.append(0.5 * math.log(a1 / a0))
    if not logs:
        return None
    return float(np.median(np.array(logs, dtype=np.float32)))

def zoom_dir_error_from_log_scale(log_scale: Optional[float], cmd_dd: float) -> Optional[float]:
    """
    cmd_dd sign:
      dd < 0 => closer => zoom in => log_scale > 0 expected
      dd > 0 => farther => zoom out => log_scale < 0 expected
      dd == 0 => ignore direction (return None)
    Returns:
      0.0 correct, 1.0 wrong, None if cannot decide
    """
    if log_scale is None:
        return None
    if cmd_dd == 0:
        return None
    exp_sign = 1.0 if cmd_dd < 0 else -1.0
    return 0.0 if (log_scale * exp_sign) > 0 else 1.0



# ----------------------------
# Read command from json
# ----------------------------
def read_cmd_from_json(json_path: Path) -> Tuple[float, float, float]:
    with open(json_path, "r", encoding="utf-8") as f:
        js = json.load(f)
    meta = js.get("metadata", {})
    ypd = meta.get("edit_ypd", {})
    dyaw = float(ypd.get("yaw", 0.0))
    dpitch = float(ypd.get("pitch", 0.0))
    ddist = float(ypd.get("distance", 0.0))
    return dyaw, dpitch, ddist


# ----------------------------
# Evaluate one sample
# ----------------------------
def evaluate_one(
    model: YOLO,
    src_path: Path,
    gt_path: Path,
    pred_path: Path,
    cmd_ddist: float,
    conf: float,
    imgsz: int,
    device,
    topk_per_class: int,
    max_match_angle_gt: float,
    max_match_angle_zoom: float,
    zoom_margin: float,
) -> Dict[str, Any]:
    src_img = cv2.imread(str(src_path))
    gt_img  = cv2.imread(str(gt_path))
    pr_img  = cv2.imread(str(pred_path))
    if src_img is None or gt_img is None or pr_img is None:
        return {"status": "read_fail"}

    Hs, Ws = src_img.shape[:2]
    Hg, Wg = gt_img.shape[:2]
    Hp, Wp = pr_img.shape[:2]

    # One batch predict
    res = model.predict(
        source=[str(src_path), str(gt_path), str(pred_path)],
        conf=conf, imgsz=imgsz, device=device,
        verbose=False, stream=False
    )
    r_src, r_gt, r_pr = res[0], res[1], res[2]

    src_dets = extract_det(r_src, conf_thres=conf, topk_per_class=topk_per_class, prefer_large=True)
    gt_dets  = extract_det(r_gt,  conf_thres=conf, topk_per_class=topk_per_class, prefer_large=True)
    pr_dets  = extract_det(r_pr,  conf_thres=conf, topk_per_class=topk_per_class, prefer_large=True)

    src_objs = build_rays(src_dets, Ws, Hs)
    gt_objs  = build_rays(gt_dets,  Wg, Hg)
    pr_objs  = build_rays(pr_dets,  Wp, Hp)

    out: Dict[str, Any] = {
        "status": "ok",
        "src_det": len(src_objs),
        "gt_det": len(gt_objs),
        "pred_det": len(pr_objs),

        # Angle similarity (GT<->PRED)
        "matches_gt": 0,
        "gt_ray_diff_deg": None,
        "gt_pairwise_diff_deg": None,

        # Zoom dir (SRC<->PRED)
        "matches_zoom": 0,
        "log_scale": None,
        "zoom_dir_err": None,
    }

    # Angle: GT <-> PRED
    matches_gt = match_by_class(gt_objs, pr_objs, max_angle_deg=max_match_angle_gt, lambda_area=2.0)
    out["matches_gt"] = len(matches_gt)
    if len(matches_gt) >= 2:
        ray_diff, pair_diff = gt_similarity_metrics(gt_objs, pr_objs, matches_gt)
        out["gt_ray_diff_deg"] = ray_diff
        out["gt_pairwise_diff_deg"] = pair_diff

    # Zoom: SRC <-> PRED
    matches_zoom = match_by_class(src_objs, pr_objs, max_angle_deg=max_match_angle_zoom, lambda_area=2.0)
    out["matches_zoom"] = len(matches_zoom)
    if len(matches_zoom) >= 1:
        log_scale = median_log_scale_from_matches(src_objs, pr_objs, matches_zoom)
        out["log_scale"] = log_scale
        out["zoom_dir_err"] = zoom_dir_error_from_log_scale(log_scale, cmd_ddist)

    # If nothing computed
    if out["gt_ray_diff_deg"] is None and out["zoom_dir_err"] is None:
        out["status"] = "det_fail"
    return out

def summarize_spatial_eval_csv(
    out_csv: str,
    min_matches: int = 2,
    only_status_ok: bool = False,
) -> Dict[str, Any]:

    path = Path(out_csv)
    if not path.exists():
        raise FileNotFoundError(out_csv)

    df = pd.read_csv(path)

    if only_status_ok:
        df = df[df["status"] == "ok"]

    n = len(df)
    if n == 0:
        return {"n_rows": 0, "note": "No rows after filtering."}

    # 数值列转 float（None 自动变 NaN）
    num_cols = [
        "gt_ray_diff_deg", "zoom_dir_err",
        "matches_gt", "matches_zoom", "gt_det"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ----------------------
    # 均值
    # ----------------------
    angle_mean_nofail = df["gt_ray_diff_deg"].mean()
    zoom_mean_nofail = df["zoom_dir_err"].mean()

    angle_mean_fail0 = df["gt_ray_diff_deg"].fillna(20).mean()
    zoom_mean_fail0 = df["zoom_dir_err"].fillna(1).mean()

    zoom_acc_nofail = None
    if not np.isnan(zoom_mean_nofail):
        zoom_acc_nofail = 1.0 - zoom_mean_nofail

    # ----------------------
    # 成功样本统计
    # ----------------------
    angle_success = df["gt_ray_diff_deg"].notna()
    zoom_success = df["log_scale"].notna()

    # ----------------------
    # 汇总
    # ----------------------
    summary = {
        
        # angle
        "angle_error_fail20": angle_mean_fail0,
        "angle_error_exfail": angle_mean_nofail,
        "angle_success_images": angle_success.sum(),

        # zoom
        "zoom_error_fail1": zoom_mean_fail0,
        "zoom_error_excfail": zoom_mean_nofail,
        "zoom_success_images": zoom_success.sum(),

        # staus
        "image_Num": n,
        # 条件统计
        f"matches_gt_ge_{min_matches}":
            (df["matches_gt"] >= min_matches).sum(),
        f"matches_src_ge_{min_matches}":
            (df["matches_zoom"] >= min_matches).sum(),
        "gt_det_lt_2":
            (df["gt_det"] < 2).sum(),
    }


    # print("\n===== Spatial Eval Summary (pandas) =====")
    # for k, v in summary.items():
    #     print(f"{k}: {v}")
    # print("=========================================\n")

    return summary

# 用法示例：

# ----------------------------
# Main
# ----------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser("GT-only angle similarity + SRC-only zoom direction, recursive folder support.")
    ap.add_argument("--folder", type=str, required=True, help="root folder (may contain nested subfolders)")
    ap.add_argument("--yolo_model", type=str, required=True, help="path to YOLO .pt")
    ap.add_argument("--out_csv", type=str, required=True, help="output csv path")

    # Naming (suffix-based)
    ap.add_argument("--src_suffix", type=str, default="_src.png", help="e.g. _src.png or _0.jpg")
    ap.add_argument("--gt_suffix", type=str, default="_gt.png", help="e.g. _gt.png or _1.jpg")
    ap.add_argument("--pred_suffix", type=str, default=".png", help="e.g. _edit0.jpg or .jpg")

    # YOLO params
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--topk_per_class", type=int, default=3)

    # Matching thresholds
    ap.add_argument("--max_match_angle_gt", type=float, default=60.0, help="GT<->PRED matching max ray angle")
    ap.add_argument("--max_match_angle_zoom", type=float, default=70.0, help="SRC<->PRED matching max ray angle for zoom")

    # Zoom tiebreak only (not a reported metric)
    ap.add_argument("--zoom_margin", type=float, default=0.08)

    # Visualization
    ap.add_argument("--viz_root", type=str, default=None, help="if set, will create angle_topk/ and zoom_topk/ under it")
    ap.add_argument("--angle_topk", type=int, default=200)
    ap.add_argument("--zoom_topk", type=int, default=200)

    args = ap.parse_args()

    folder = Path(args.folder)
    model = YOLO(args.yolo_model)

    # ----------------------------
    # Find prefixes recursively by GT files: "<prefix><gt_suffix>"
    # prefix may include nested subdirs relative to folder
    # ----------------------------
    prefixes = set()
    gt_suffix = args.gt_suffix

    for p in folder.rglob("*"):
        if not p.is_file():
            continue
        name = p.name
        if not name.lower().endswith(gt_suffix.lower()):
            continue
        base = name[:-len(gt_suffix)]
        if base == "":
            continue
        rel_dir = p.parent.relative_to(folder)
        pref = str(rel_dir / base) if str(rel_dir) != "." else base
        prefixes.add(pref)

    prefixes = sorted(prefixes)
    if not prefixes:
        raise RuntimeError(f"No prefixes found via *{gt_suffix} under {folder}")
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    # ----------------------------
    # Evaluate
    # ----------------------------
    for pref in prefixes:
        json_path = folder / f"{pref}.json"
        src_path  = folder / f"{pref}{args.src_suffix}"
        gt_path   = folder / f"{pref}{args.gt_suffix}"
        pred_path = folder / f"{pref}{args.pred_suffix}"

        if not (json_path.exists() and src_path.exists() and gt_path.exists() and pred_path.exists()):
            continue

        # _, _, cmd_ddist = read_cmd_from_json(json_path)
        cmd_dyaw, cmd_dpitch, cmd_ddist = read_cmd_from_json(json_path)

        m = evaluate_one(
            model=model,
            src_path=src_path,
            gt_path=gt_path,
            pred_path=pred_path,
            cmd_ddist=cmd_ddist,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            topk_per_class=args.topk_per_class,
            max_match_angle_gt=args.max_match_angle_gt,
            max_match_angle_zoom=args.max_match_angle_zoom,
            zoom_margin=args.zoom_margin,
        )

        rows.append({
            "prefix": pref,
            "status": m.get("status", "unk"),

            "cmd_dyaw": cmd_dyaw,
            "cmd_dpitch": cmd_dpitch,
            "cmd_ddist": cmd_ddist,

            # Angle (GT<->PRED): lower is better
            "gt_ray_diff_deg": m.get("gt_ray_diff_deg"),
            "gt_pairwise_diff_deg": m.get("gt_pairwise_diff_deg"),
            "matches_gt": m.get("matches_gt", 0),

            # Zoom (SRC<->PRED): zoom_dir_err only (0 correct, 1 wrong)
            "zoom_dir_err": m.get("zoom_dir_err"),
            "log_scale": m.get("log_scale"),
            "matches_zoom": m.get("matches_zoom", 0),

            # Diagnostics
            "src_det": m.get("src_det", 0),
            "gt_det": m.get("gt_det", 0),
            "pred_det": m.get("pred_det", 0),
        })

    # ----------------------------
    # Write CSV (no total score)
    # ----------------------------
    

    fieldnames = [
        "prefix", "status",
        "cmd_dyaw", "cmd_dpitch", "cmd_ddist",
        "gt_ray_diff_deg", "gt_pairwise_diff_deg", "matches_gt",
        "zoom_dir_err", "log_scale", "matches_zoom",
        "src_det", "gt_det", "pred_det",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in fieldnames})

    print(f"[DONE] rows={len(rows)} -> {out_csv}")
    

    sum_df = summarize_spatial_eval_csv(out_csv, min_matches=2, only_status_ok=False)
    name_ = '_'.join(args.folder.split('/')[-2:])
    sum_df = pd.DataFrame([sum_df], index=[name_]).T

    sum_df.to_csv(args.out_csv.replace('.csv','_sum.csv'))
    print(sum_df)
    # import pdb; pdb.set_trace()

    # ----------------------------
    # Visualization: split into two folders
    # ----------------------------
    if args.viz_root is not None:
        viz_root = Path(args.viz_root)
        angle_dir = viz_root / "angle_topk"
        zoom_dir  = viz_root / "zoom_topk"
        angle_dir.mkdir(parents=True, exist_ok=True)
        zoom_dir.mkdir(parents=True, exist_ok=True)

        # --- angle ranking: smallest gt_ray_diff_deg (ok first)
        # 
        angle_rows = [
            r for r in rows
            if r.get("status") == "ok"
            and r.get("gt_ray_diff_deg") is not None
            and classify_edit(float(r.get("cmd_dyaw", 0.0)),
                            float(r.get("cmd_dpitch", 0.0)),
                            float(r.get("cmd_ddist", 0.0))) == "angle"
        ]
        angle_rows.sort(key=lambda r: (float(r["gt_ray_diff_deg"])))

        zoom_rows = [
            r for r in rows
            if r.get("status") == "ok"
            and r.get("zoom_dir_err") is not None
            and classify_edit(float(r.get("cmd_dyaw", 0.0)),
                            float(r.get("cmd_dpitch", 0.0)),
                            float(r.get("cmd_ddist", 0.0))) == "zoom"
        ]

        # --- zoom ranking: zoom_dir_err (0 first), tie-break by zoom_mag_err (smaller better), then matches
        def _zoom_key(r):
            zdir = float(r["zoom_dir_err"])
            ls = r.get("log_scale")
            return (zdir, abs(ls))
            # return (zdir, zmag, -int(r.get("matches_zoom", 0)))
        zoom_rows.sort(key=_zoom_key)

        def _safe_name(pref: str) -> str:
            s = re.sub(r"[^\w\-.~#/]+", "_", pref)
            return s.replace("/", "_")

        # save angle topk
        saved = 0
        for idx, r in enumerate(angle_rows[:args.angle_topk], 1):
            pref = r["prefix"]
            src_path  = folder / f"{pref}{args.src_suffix}"
            gt_path   = folder / f"{pref}{args.gt_suffix}"
            pred_path = folder / f"{pref}{args.pred_suffix}"

            src = cv2.imread(str(src_path))
            gt  = cv2.imread(str(gt_path))
            pr  = cv2.imread(str(pred_path))
            if src is None or gt is None or pr is None:
                continue

            cat = hconcat_three(src, gt, pr)
            val = float(r["gt_ray_diff_deg"])
            out_name = f"{idx:04d}_ray{val:06.2f}_{_safe_name(pref)}.jpg"
            cv2.imwrite(str(angle_dir / out_name), cat)
            saved += 1
        print(f"[VIZ] angle saved={saved} -> {angle_dir}")

        # save zoom topk
        saved = 0
        for idx, r in enumerate(zoom_rows[:args.zoom_topk], 1):
            pref = r["prefix"]
            src_path  = folder / f"{pref}{args.src_suffix}"
            gt_path   = folder / f"{pref}{args.gt_suffix}"
            pred_path = folder / f"{pref}{args.pred_suffix}"

            src = cv2.imread(str(src_path))
            gt  = cv2.imread(str(gt_path))
            pr  = cv2.imread(str(pred_path))
            if src is None or gt is None or pr is None:
                continue

            cat = hconcat_three(src, gt, pr)
            zdir = float(r["zoom_dir_err"])
            ls = r.get("log_scale")
            ls = float(ls) if ls is not None else 0.0
            out_name = f"{idx:04d}_dir{int(zdir)}_ls{ls:+.3f}_{_safe_name(pref)}.jpg"
            cv2.imwrite(str(zoom_dir / out_name), cat)
            saved += 1
        print(f"[VIZ] zoom saved={saved} -> {zoom_dir}")


if __name__ == "__main__":
    main()