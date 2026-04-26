import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

TARGET = 384
PADDING = 0.40
S_MAX = 1.5


def bbox_from_shapes(shapes):
    pts = []
    for sh in shapes:
        for p in sh.get("points", []):
            pts.append(p)
    if not pts:
        return None
    pts = np.array(pts, dtype=np.float32)
    return (
        int(np.min(pts[:, 0])),
        int(np.min(pts[:, 1])),
        int(np.max(pts[:, 0])),
        int(np.max(pts[:, 1])),
    )


def polygon_centroid_area(points):
    if len(points) < 3:
        return None, 0.0
    x = points[:, 0]
    y = points[:, 1]
    x1 = np.roll(x, -1)
    y1 = np.roll(y, -1)
    cross = x * y1 - x1 * y
    area_signed = 0.5 * np.sum(cross)
    if abs(area_signed) < 1e-8:
        return np.array([float(np.mean(x)), float(np.mean(y))]), 0.0
    cx = np.sum((x + x1) * cross) / (6 * area_signed)
    cy = np.sum((y + y1) * cross) / (6 * area_signed)
    return np.array([cx, cy]), abs(area_signed)


def f9_slide(img_shape, x1, y1, x2, y2):
    h, w = img_shape[:2]
    win_w = x2 - x1
    win_h = y2 - y1

    if x1 < 0:
        shift = -x1
        x1 += shift
        x2 = min(w, x2 + shift)
    if x2 > w:
        shift = x2 - w
        x2 -= shift
        x1 = max(0, x1 - shift)

    if y1 < 0:
        shift = -y1
        y1 += shift
        y2 = min(h, y2 + shift)
    if y2 > h:
        shift = y2 - h
        y2 -= shift
        y1 = max(0, y1 - shift)

    cur_w = max(0, x2 - x1)
    cur_h = max(0, y2 - y1)
    need_w = max(0, win_w - cur_w)
    need_h = max(0, win_h - cur_h)

    final_w = cur_w + need_w
    final_h = cur_h + need_h
    used_reflect = need_w > 0 or need_h > 0

    return int(x1), int(y1), int(final_w), int(final_h), bool(used_reflect)


def collect_dataset_paths(root):
    out = {}
    for split in ["treino", "val", "teste"]:
        for cls in ["benigno", "maligno"]:
            d = root / split / cls
            if not d.exists():
                continue
            for fn in os.listdir(d):
                if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")):
                    out[(split, cls, fn)] = d / fn
    return out


def analyze(base_dir):
    base = Path(base_dir)
    dir_images = base / "data" / "images"
    dir_ann = base / "data" / "Annotations"
    f6_dir = base / "fases" / "Fase6" / "dataset"
    f9_dir = base / "fases" / "Fase9" / "dataset"

    f6_files = collect_dataset_paths(f6_dir)
    f9_files = collect_dataset_paths(f9_dir)
    keys = sorted(set(f6_files).intersection(set(f9_files)))

    rows = []
    for split, cls, fn in keys:
        stem = os.path.splitext(fn)[0]
        ann_path = dir_ann / f"{stem}.json"
        img_path = dir_images / fn

        if not ann_path.exists() or not img_path.exists():
            continue

        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        bbox = bbox_from_shapes(ann.get("shapes", []))
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        h, w = img.shape[:2]
        pad_x = int((x2 - x1) * PADDING)
        pad_y = int((y2 - y1) * PADDING)
        nx1, ny1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        nx2, ny2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

        crop_w = max(1, nx2 - nx1)
        crop_h = max(1, ny2 - ny1)

        s_fit = min(TARGET / crop_w, TARGET / crop_h)
        s_used = min(s_fit, S_MAX) if s_fit > 1.0 else s_fit

        new_w = max(1, int(round(crop_w * s_used)))
        new_h = max(1, int(round(crop_h * s_used)))
        ox = (TARGET - new_w) // 2
        oy = (TARGET - new_h) // 2

        cx = (nx1 + nx2) // 2
        cy = (ny1 + ny2) // 2
        ctx_w = max(1, int(round(TARGET / max(s_used, 1e-6))))
        ctx_h = max(1, int(round(TARGET / max(s_used, 1e-6))))
        wx1 = cx - ctx_w // 2
        wy1 = cy - ctx_h // 2
        wx2 = wx1 + ctx_w
        wy2 = wy1 + ctx_h

        f6_border_touch = wx1 < 0 or wy1 < 0 or wx2 > w or wy2 > h

        sx1, sy1, sw, sh, used_reflect = f9_slide(img.shape, wx1, wy1, wx2, wy2)

        all_polys = []
        poly_areas = []
        poly_centroids = []
        for sh_obj in ann.get("shapes", []):
            pts = np.array(sh_obj.get("points", []), dtype=np.float32)
            if len(pts) < 3:
                continue
            c, a = polygon_centroid_area(pts)
            if c is None:
                continue
            all_polys.append(pts)
            poly_centroids.append(c)
            poly_areas.append(a)

        if not poly_areas:
            continue

        total_area = float(np.sum(poly_areas))
        centroid_orig = np.sum(
            [poly_centroids[i] * poly_areas[i] for i in range(len(poly_areas))], axis=0
        ) / total_area

        c6x = ox + (centroid_orig[0] - nx1) * (new_w / crop_w)
        c6y = oy + (centroid_orig[1] - ny1) * (new_h / crop_h)
        c9x = (centroid_orig[0] - sx1) * (TARGET / sw)
        c9y = (centroid_orig[1] - sy1) * (TARGET / sh)

        d6 = float(np.hypot(c6x - TARGET / 2, c6y - TARGET / 2))
        d9 = float(np.hypot(c9x - TARGET / 2, c9y - TARGET / 2))

        area6 = 0.0
        area9 = 0.0
        for pts in all_polys:
            p6 = np.empty_like(pts)
            p6[:, 0] = ox + (pts[:, 0] - nx1) * (new_w / crop_w)
            p6[:, 1] = oy + (pts[:, 1] - ny1) * (new_h / crop_h)
            _, a6 = polygon_centroid_area(p6)
            area6 += a6

            p9 = np.empty_like(pts)
            p9[:, 0] = (pts[:, 0] - sx1) * (TARGET / sw)
            p9[:, 1] = (pts[:, 1] - sy1) * (TARGET / sh)
            _, a9 = polygon_centroid_area(p9)
            area9 += a9

        frac6 = area6 / (TARGET * TARGET)
        frac9 = area9 / (TARGET * TARGET)

        min_border_dist = min(nx1, ny1, w - nx2, h - ny2)

        im6 = cv2.imread(str(f6_files[(split, cls, fn)]), cv2.IMREAD_COLOR)
        im9 = cv2.imread(str(f9_files[(split, cls, fn)]), cv2.IMREAD_COLOR)
        if im6 is not None and im9 is not None and im6.shape == im9.shape:
            diff = cv2.absdiff(im6, im9)
            diff_mean = float(np.mean(diff))
            diff_p95 = float(np.percentile(diff, 95))
        else:
            diff_mean = float("nan")
            diff_p95 = float("nan")

        rows.append(
            {
                "split": split,
                "class": cls,
                "file": fn,
                "min_border_dist": int(min_border_dist),
                "f6_border_touch": bool(f6_border_touch),
                "f9_reflect": bool(used_reflect),
                "s_fit": float(s_fit),
                "s_used": float(s_used),
                "cent_dist_f6": d6,
                "cent_dist_f9": d9,
                "cent_dist_delta_f9_minus_f6": d9 - d6,
                "lesion_frac_f6": frac6,
                "lesion_frac_f9": frac9,
                "lesion_frac_delta": frac9 - frac6,
                "img_diff_mean": diff_mean,
                "img_diff_p95": diff_p95,
            }
        )

    df = pd.DataFrame(rows)

    r6 = pd.read_csv(base / "fases" / "Fase6" / "RESULTADOS_TESTE_FASE6.csv")
    r9 = pd.read_csv(base / "fases" / "Fase9" / "RESULTADOS_TESTE_FASE9.csv")
    merged = r6.merge(r9, on="Rede", suffixes=("_F6", "_F9"))
    for col in ["Acuracia", "AUC", "F1-Score", "Precisao", "Recall (Sensibilidade)"]:
        merged[f"Delta_{col}"] = merged[f"{col}_F9"] - merged[f"{col}_F6"]

    return df, merged


def main():
    base = Path(__file__).resolve().parents[2]
    df, merged = analyze(base)

    print("rows analyzed:", len(df))
    print("\nMetric deltas F9 - F6:")
    print(merged.to_string(index=False))

    if len(df) == 0:
        return

    print("\n== Center displacement summary ==")
    print(df[["cent_dist_f6", "cent_dist_f9", "cent_dist_delta_f9_minus_f6"]].describe().round(3))

    print("\n== Means grouped by f6_border_touch ==")
    print(
        df.groupby("f6_border_touch")[[
            "cent_dist_delta_f9_minus_f6",
            "lesion_frac_delta",
            "img_diff_mean",
        ]]
        .mean()
        .round(4)
        .to_string()
    )

    print("\n== Means grouped by split ==")
    print(
        df.groupby("split")[[
            "cent_dist_delta_f9_minus_f6",
            "lesion_frac_delta",
            "img_diff_mean",
        ]]
        .mean()
        .round(4)
        .to_string()
    )

    print("\n== Reflect fallback usage ==")
    print(df["f9_reflect"].value_counts(dropna=False).to_string())

    show_cols = [
        "split",
        "class",
        "file",
        "min_border_dist",
        "f6_border_touch",
        "cent_dist_f6",
        "cent_dist_f9",
        "cent_dist_delta_f9_minus_f6",
        "img_diff_mean",
    ]
    print("\nTop 15 largest F9 center drift increases:")
    print(df.sort_values("cent_dist_delta_f9_minus_f6", ascending=False)[show_cols].head(15).to_string(index=False))

    case_name = "IMG000819.jpeg"
    one = df[df["file"] == case_name]
    print("\nSpecific case IMG000819:")
    if len(one) == 0:
        print("  Not found in paired set.")
    else:
        print(
            one[
                show_cols + ["lesion_frac_f6", "lesion_frac_f9", "lesion_frac_delta"]
            ].to_string(index=False)
        )

    out_csv = base / "results" / "analise_fase6_vs_fase9_geometria.csv"
    df.to_csv(out_csv, index=False)
    print("\nSaved detailed analysis:", out_csv)


if __name__ == "__main__":
    main()
