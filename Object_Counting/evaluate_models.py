"""Evaluation and Comparison Script for Shelf PC Case Counting.

Compares 3 methods:
1. YOLO11 (weights pc_v7)
2. Grounding DINO (zero-shot transformers)
3. LOCA (density map integral)

Runs both 'Mặc định (Baseline)' and 'Đã chỉnh (Tuned)' configurations.
Generates:
- CSV report with columns: file, gt, model, cấu hình, pred (plus runtime & device)
- Summary metrics table: MAE, RMSE, Bias, Exact Match %, Within ±1 %, Avg Time, Device
- Scatter plot (GT on x-axis, Pred on y-axis, diagonal y=x reference)
- Annotated visualization for worst-case error images
"""

from __future__ import annotations
import argparse
import csv
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from count import annotate as yolo_annotate
from object_counting.cli import ROOT
from object_counting.detection import CaseDetector
from object_counting.grounding_dino.detector import GroundingDinoDetector
from object_counting.grounding_dino.visualizer import annotate_image as dino_annotate
from object_counting.loca_detector import LOCADetector
from object_counting.shelf_pipeline import (
    DEFAULT_FIXED_EXEMPLARS,
    get_device_name,
    get_device_type,
)


def find_image_file(filename: str, search_dirs: List[Path]) -> Optional[Path]:
    """Search for image filename in candidate directories."""
    path = Path(filename)
    if path.is_file():
        return path

    base_name = path.name
    for d in search_dirs:
        cand = d / base_name
        if cand.is_file():
            return cand
    return None


def read_ground_truth_csv(gt_path: Path) -> List[Tuple[str, int]]:
    """Read CSV with columns 'file' and 'gt'."""
    if not gt_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file Ground Truth: {gt_path}")

    records = []
    with open(gt_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = None
        file_idx = 0
        gt_idx = 1
        for row in reader:
            if not row or not any(row):
                continue
            if header is None:
                header = [col.strip().lower() for col in row]
                for i, col in enumerate(header):
                    if "file" in col or "image" in col or "img" in col:
                        file_idx = i
                    elif "gt" in col or "count" in col or "label" in col:
                        gt_idx = i
                continue

            fname = row[file_idx].strip()
            try:
                gt_val = int(round(float(row[gt_idx].strip())))
                records.append((fname, gt_val))
            except (ValueError, IndexError):
                continue

    if not records:
        raise ValueError(f"File {gt_path} không có dữ liệu nhãn hợp lệ.")
    return records


class BenchmarkRunner:
    def __init__(self, search_dirs: Optional[List[Path]] = None):
        self.device = get_device_type()
        self.device_name = get_device_name()
        self.search_dirs = search_dirs or [
            ROOT / "datasets" / "video_frames" / "ke_pc_20261118" / "split_50_train_10_test" / "test",
            ROOT / "datasets" / "video_frames" / "ke_pc_20261118" / "candidates",
            ROOT / "web",
            ROOT,
        ]

        # Model instances (lazy loaded)
        self.yolo_detector: Optional[CaseDetector] = None
        self.dino_detector: Optional[GroundingDinoDetector] = None
        self.loca_detector: Optional[LOCADetector] = None

    def get_yolo(self) -> CaseDetector:
        if self.yolo_detector is None:
            weights = ROOT / "models" / "pc_v7_best.pt"
            if not weights.is_file():
                weights = ROOT / "models" / "best.pt"
            self.yolo_detector = CaseDetector(
                weights=str(weights),
                device=self.device,
                imgsz=960,
                confidence=0.35,
            )
        return self.yolo_detector

    def get_dino(self) -> GroundingDinoDetector:
        if self.dino_detector is None:
            self.dino_detector = GroundingDinoDetector(
                model_id="IDEA-Research/grounding-dino-tiny",
                device=self.device,
            )
        return self.dino_detector

    def get_loca(self) -> LOCADetector:
        if self.loca_detector is None:
            self.loca_detector = LOCADetector(
                models_dir=ROOT / "models",
                device=self.device,
            )
        return self.loca_detector

    def run_image(
        self,
        image_bgr: np.ndarray,
        model_name: str,
        config_name: str,
    ) -> Tuple[float, Any, float]:
        """Run single image prediction.

        Returns:
            (pred_value, visual_canvas, elapsed_ms)
        """
        start = time.perf_counter()

        if model_name == "YOLO11":
            detector = self.get_yolo()
            if config_name == "Mặc định (Baseline)":
                detector.confidence = 0.40
                detector.imgsz = 640
                detector.nms_iou = 0.50
                detector.use_split = False
                detector.use_roi = False
            else:  # Tuned
                detector.confidence = 0.35
                detector.imgsz = 960
                detector.nms_iou = 0.45
                detector.use_split = True
                detector.use_roi = True

            dets = detector.detect(image_bgr)
            elapsed = (time.perf_counter() - start) * 1000.0
            canvas = yolo_annotate(image_bgr, dets)
            return float(len(dets)), canvas, elapsed

        elif model_name == "Grounding DINO":
            detector = self.get_dino()
            if config_name == "Mặc định (Baseline)":
                prompt = "pc case."
                box_th = 0.35
                text_th = 0.25
                nms_th = 0.50
                filter_large = False
                use_split = False
                use_roi = False
            else:  # Tuned
                prompt = "pc case."
                box_th = 0.20
                text_th = 0.20
                nms_th = 0.45
                filter_large = True
                use_split = True
                use_roi = True

            dets = detector.detect(
                image_bgr,
                prompt=prompt,
                box_threshold=box_th,
                text_threshold=text_th,
                nms_threshold=nms_th,
                filter_large_boxes=filter_large,
                max_area_ratio=0.30,
                use_split=use_split,
                use_roi=use_roi,
            )
            elapsed = (time.perf_counter() - start) * 1000.0
            canvas = dino_annotate(image_bgr, dets, title_prefix=f"DINO ({prompt})")
            return float(len(dets)), canvas, elapsed

        elif model_name == "LOCA":
            detector = self.get_loca()
            if config_name == "Mặc định (Baseline)":
                # Zero-shot, no split, no padding
                res = detector.count(
                    image_bgr,
                    zero_shot=True,
                    use_fixed_exemplars=False,
                    pad_square=False,
                    use_split=False,
                    use_roi=False,
                )
            else:  # Tuned
                # Few-shot with fixed 3-exemplars, square padding, 2-half split, ROI
                res = detector.count(
                    image_bgr,
                    zero_shot=False,
                    use_fixed_exemplars=True,
                    pad_square=True,
                    use_split=True,
                    use_roi=True,
                    peak_min_relative=0.35,
                )
            elapsed = (time.perf_counter() - start) * 1000.0
            # For LOCA, return raw continuous density sum for MAE calculation
            raw_density = float(res["density_sum"])
            canvas = res["canvas"]
            return raw_density, canvas, elapsed

        else:
            raise ValueError(f"Unknown model name: {model_name}")


def evaluate(
    gt_file: str = "gt.csv",
    output_csv: str = "outputs/evaluation_results.csv",
    scatter_file: str = "outputs/scatter_comparison.png",
    worst_dir: str = "outputs/worst_cases",
    models_to_test: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run full evaluation on all 3 models with both baseline and tuned settings."""
    gt_path = Path(gt_file)
    records = read_ground_truth_csv(gt_path)
    print(f"[*] Đã nạp {len(records)} ảnh từ ground truth: {gt_path.name}")

    out_csv_path = Path(output_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    worst_path = Path(worst_dir)
    worst_path.mkdir(parents=True, exist_ok=True)

    runner = BenchmarkRunner(search_dirs=[gt_path.parent] + [
        ROOT / "datasets" / "video_frames" / "ke_pc_20261118" / "split_50_train_10_test" / "test",
        ROOT / "datasets" / "video_frames" / "ke_pc_20261118" / "candidates",
        ROOT / "web",
        ROOT,
    ])

    all_models = ["YOLO11", "Grounding DINO", "LOCA"]
    if models_to_test:
        def norm(s): return s.lower().replace("_", "").replace(" ", "").replace("-", "")
        allowed = [norm(mt) for mt in models_to_test]
        all_models = [m for m in all_models if norm(m) in allowed]

    configs = ["Mặc định (Baseline)", "Đã chỉnh (Tuned)"]

    # Storage for all row items: (file, gt, model, config, pred, error, elapsed, device)
    all_rows: List[Dict[str, Any]] = []
    # Metrics grouped by (model, config)
    grouped_data: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    for model in all_models:
        for config in configs:
            grouped_data[(model, config)] = []

    print(f"[*] Thiết bị kiểm thử: {runner.device_name}")
    print("[*] Đang bắt đầu đánh giá...")

    for model in all_models:
        for config in configs:
            key = (model, config)
            print(f"  -> Đang chạy {model} [{config}]...")
            for fname, gt_val in records:
                img_path = find_image_file(fname, runner.search_dirs)
                if img_path is None:
                    print(f"     [!] Không tìm thấy ảnh '{fname}'. Bỏ qua.")
                    continue

                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    print(f"     [!] Không đọc được ảnh '{img_path}'. Bỏ qua.")
                    continue

                pred_val, canvas, elapsed = runner.run_image(bgr, model, config)

                # Convention:
                # - For LOCA: pred_val is float density sum.
                #   Error for MAE = pred_val - gt_val.
                #   Integer count for Exact match & ±1 = round(pred_val).
                # - For YOLO & DINO: pred_val is integer count.
                if model == "LOCA":
                    rounded_pred = int(round(pred_val))
                    error = pred_val - gt_val
                    exact = (rounded_pred == gt_val)
                    within_one = (abs(rounded_pred - gt_val) <= 1)
                    display_pred = round(pred_val, 2)
                else:
                    int_pred = int(round(pred_val))
                    error = float(int_pred - gt_val)
                    exact = (int_pred == gt_val)
                    within_one = (abs(int_pred - gt_val) <= 1)
                    display_pred = int_pred

                row_dict = {
                    "file": fname,
                    "gt": gt_val,
                    "model": model,
                    "cấu hình": config,
                    "pred": display_pred,
                    "error": round(error, 2),
                    "abs_error": abs(error),
                    "exact": exact,
                    "within_one": within_one,
                    "elapsed_ms": round(elapsed, 1),
                    "device": runner.device_name,
                    "canvas": canvas,
                }
                all_rows.append(row_dict)
                grouped_data[key].append(row_dict)

    # 1. Write CSV with required columns: file, gt, model, cấu hình, pred (plus time & device)
    fieldnames = ["file", "gt", "model", "cấu hình", "pred", "sai_số", "thời_gian_ms", "thiết_bị"]
    with open(out_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({
                "file": r["file"],
                "gt": r["gt"],
                "model": r["model"],
                "cấu hình": r["cấu hình"],
                "pred": r["pred"],
                "sai_số": r["error"],
                "thời_gian_ms": r["elapsed_ms"],
                "thiết_bị": r["device"],
            })

    print(f"\n[+] Đã xuất kết quả chi tiết từng ảnh vào CSV: {out_csv_path}")

    # 2. Compute Summary Metrics Table
    summary_table = []
    print("\n" + "=" * 122)
    print("BẢNG TỔNG HỢP HIỆU NĂNG SO SÁNH CÁC PHƯƠNG PHÁP ĐẾM PC CASE TRÊN KỆ")
    print("=" * 122)
    header_str = (
        f"{'Model':<16} | {'Cấu hình':<22} | {'MAE':<7} | {'RMSE':<7} | {'Bias':<7} | "
        f"{'Đúng 100%':<10} | {'Đúng ±1':<9} | {'Thời gian TB':<13} | {'Thiết bị'}"
    )
    print(header_str)
    print("-" * 122)

    for (model, config), rows in grouped_data.items():
        if not rows:
            continue
        n = len(rows)
        mae = sum(r["abs_error"] for r in rows) / n
        rmse = math.sqrt(sum(r["error"] ** 2 for r in rows) / n)
        bias = sum(r["error"] for r in rows) / n
        exact_pct = (sum(1 for r in rows if r["exact"]) / n) * 100.0
        within1_pct = (sum(1 for r in rows if r["within_one"]) / n) * 100.0
        avg_time = sum(r["elapsed_ms"] for r in rows) / n

        summary_table.append({
            "model": model,
            "config": config,
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "bias": round(bias, 2),
            "exact_pct": round(exact_pct, 1),
            "within1_pct": round(within1_pct, 1),
            "avg_time": round(avg_time, 1),
            "device": runner.device_name,
        })

        row_str = (
            f"{model:<16} | {config:<22} | {mae:<7.2f} | {rmse:<7.2f} | {bias:<+7.2f} | "
            f"{exact_pct:<9.1f}% | {within1_pct:<8.1f}% | {avg_time:<10.1f} ms | {runner.device_name}"
        )
        print(row_str)

    print("=" * 122)
    print("(*) Lưu ý: LOCA dùng số mật độ thực để tính MAE/RMSE/Bias, và số làm tròn để tính tỉ lệ Đúng 100% và Đúng ±1.")

    # 3. Generate Scatter Plot Comparison
    generate_scatter_plot(grouped_data, scatter_file)

    # 4. Save Worst-Case Error Visualizations
    save_worst_case_images(grouped_data, worst_path)

    return {"summary": summary_table, "csv": str(out_csv_path), "scatter": scatter_file}


def generate_scatter_plot(
    grouped_data: Dict[Tuple[str, str], List[Dict[str, Any]]],
    output_path: str,
):
    """Create scatter plot with GT on x-axis, Pred on y-axis, and y=x diagonal."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True, sharex=True)
    models = ["YOLO11", "Grounding DINO", "LOCA"]

    colors = {
        "Mặc định (Baseline)": "#e74c3c",  # Red
        "Đã chỉnh (Tuned)": "#27ae60",     # Green
    }
    markers = {
        "Mặc định (Baseline)": "x",
        "Đã chỉnh (Tuned)": "o",
    }

    all_gt = []
    all_pred = []

    for ax, model in zip(axes, models):
        # Diagonal y=x reference line
        ax.plot([4, 15], [4, 15], "k--", alpha=0.5, label="y = x (Lý tưởng)")

        for config in ["Mặc định (Baseline)", "Đã chỉnh (Tuned)"]:
            key = (model, config)
            rows = grouped_data.get(key, [])
            if not rows:
                continue

            gt_vals = [r["gt"] for r in rows]
            pred_vals = [r["pred"] for r in rows]
            all_gt.extend(gt_vals)
            all_pred.extend(pred_vals)

            mae = sum(r["abs_error"] for r in rows) / len(rows)
            exact_pct = (sum(1 for r in rows if r["exact"]) / len(rows)) * 100.0

            # Small jitter on x to avoid complete overlap of identical points
            jitter = np.random.uniform(-0.08, 0.08, size=len(gt_vals))
            ax.scatter(
                np.array(gt_vals) + jitter,
                pred_vals,
                c=colors[config],
                marker=markers[config],
                s=70,
                alpha=0.85,
                label=f"{config}\n(MAE: {mae:.2f}, Đúng: {exact_pct:.0f}%)",
            )

        ax.set_title(f"Model: {model}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Số thực tế (Ground Truth)", fontsize=11)
        ax.set_ylabel("Số dự đoán (Prediction)", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", fontsize=9)

    min_val = max(0, min(all_gt + all_pred, default=4) - 1)
    max_val = max(all_gt + all_pred, default=14) + 2
    for ax in axes:
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)

    plt.suptitle("So Sánh Số Đếm Dự Đoán vs Số Thực Tế Trên Kệ PC (Scatter Plot)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Đã lưu biểu đồ scatter so sánh tại: {output_path}")


def save_worst_case_images(
    grouped_data: Dict[Tuple[str, str], List[Dict[str, Any]]],
    output_dir: Path,
):
    """Save visualization images for top worst errors in each model & configuration."""
    print(f"[*] Đang lưu ảnh minh họa các trường hợp sai lệch nhiều nhất vào: {output_dir.name}/...")

    for (model, config), rows in grouped_data.items():
        if not rows:
            continue
        # Sort by absolute error descending
        sorted_rows = sorted(rows, key=lambda r: r["abs_error"], reverse=True)
        # Take top 2 worst
        worst_cases = sorted_rows[:2]

        safe_model = model.lower().replace(" ", "_")
        safe_config = "baseline" if "baseline" in config.lower() or "mặc định" in config.lower() else "tuned"

        for idx, item in enumerate(worst_cases):
            canvas = item.get("canvas")
            if canvas is None:
                continue

            # Overlay a clear label banner at the bottom
            h, w = canvas.shape[:2]
            banner_h = 60
            banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
            banner[:] = (20, 20, 20)

            text1 = f"{model} [{config}] | File: {Path(item['file']).name}"
            text2 = f"GT: {item['gt']} case  |  Pred: {item['pred']}  |  Sai lech: {item['error']:+.2f}"

            cv2.putText(banner, text1, (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(banner, text2, (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2, cv2.LINE_AA)

            annotated_out = np.vstack([canvas, banner])

            out_filename = f"{safe_model}_{safe_config}_worst{idx + 1}_{Path(item['file']).stem}.jpg"
            save_path = output_dir / out_filename
            cv2.imwrite(str(save_path), annotated_out)

    print(f"[+] Đã lưu xong các ảnh minh họa ca sai lệch nhiều nhất.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=str, default="gt.csv", help="Đường dẫn file Ground Truth CSV (cột: file, gt)")
    parser.add_argument("--output", type=str, default="outputs/evaluation_results.csv", help="Đường dẫn file CSV kết quả")
    parser.add_argument("--scatter", type=str, default="outputs/scatter_comparison.png", help="Đường dẫn ảnh biểu đồ scatter")
    parser.add_argument("--worst-dir", type=str, default="outputs/worst_cases", help="Thư mục lưu ảnh minh họa sai số lớn")
    parser.add_argument("--models", nargs="+", default=["yolo11", "grounding_dino", "loca"], help="Các model cần đánh giá")
    args = parser.parse_args()

    evaluate(
        gt_file=args.gt,
        output_csv=args.output,
        scatter_file=args.scatter,
        worst_dir=args.worst_dir,
        models_to_test=args.models,
    )


if __name__ == "__main__":
    main()
