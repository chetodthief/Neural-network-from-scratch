"""
evaluation.py - Evaluation Pipeline for Custom Lane Segmentation Neural Network
Features:
    - Compares predicted binary masks with Ground Truth masks (1280x720 px)
    - Computes Intersection over Union (IoU) per sample
    - Evaluates detection rate based on standard threshold: IoU >= 0.6
    - Computes comprehensive metrics: mIoU, Precision, Recall, F1-Score, Pixel Accuracy
    - Generates IoU distribution histogram
    - Exports side-by-side snapshot comparison figures (Original | GT | Pred | Overlay)
    - Outputs full evaluation report to console and markdown/text files
"""

import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from dataset import load_polygon_mask


def compute_mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Computes binary mask Intersection over Union (IoU)."""
    p_bin = pred_mask > 0
    gt_bin = gt_mask > 0

    intersection = np.logical_and(p_bin, gt_bin).sum()
    union = np.logical_or(p_bin, gt_bin).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)


def compute_pixel_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """Computes pixel-level TP, FP, FN, TN."""
    p_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    tp = np.logical_and(p_bin == 1, gt_bin == 1).sum()
    fp = np.logical_and(p_bin == 1, gt_bin == 0).sum()
    fn = np.logical_and(p_bin == 0, gt_bin == 1).sum()
    tn = np.logical_and(p_bin == 0, gt_bin == 0).sum()

    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def create_side_by_side_snapshot(
    img_bgr: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    iou: float,
    save_path: str,
    title_suffix: str = ""
):
    """
    Creates a 4-panel snapshot:
    [Original Frame | Ground Truth Mask | Predicted Mask | Segmentation Overlay]
    """
    orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Overlay: green on predicted, red on false positives/misses if needed, or cyan overlay
    overlay = orig_rgb.copy()
    overlay[pred_mask > 0] = [0, 255, 120]  # Vibrant green
    blended = cv2.addWeighted(orig_rgb, 0.6, overlay, 0.4, 0)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Lane Segmentation Snapshot Comparison (IoU = {iou:.4f}) {title_suffix}", fontsize=14, fontweight='bold')

    axes[0].imshow(orig_rgb)
    axes[0].set_title("1. Original Frame (1280x720 px)", fontsize=11, fontweight='semibold')
    axes[0].axis('off')

    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title("2. Ground Truth Polygon Mask", fontsize=11, fontweight='semibold')
    axes[1].axis('off')

    axes[2].imshow(pred_mask, cmap='gray')
    status_text = "PASS (IoU >= 0.6)" if iou >= 0.6 else "FAIL (IoU < 0.6)"
    status_color = "green" if iou >= 0.6 else "red"
    axes[2].set_title(f"3. Predicted Mask [{status_text}]", fontsize=11, fontweight='semibold', color=status_color)
    axes[2].axis('off')

    axes[3].imshow(blended)
    axes[3].set_title("4. Inference Overlay Blended", fontsize=11, fontweight='semibold')
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_iou_histogram(ious: List[float], threshold: float = 0.6, save_path: str = 'output/iou_distribution.png'):
    """Plots IoU distribution histogram with the IoU >= 0.6 positive threshold line."""
    plt.figure(figsize=(9, 5))
    counts, bins, patches = plt.hist(ious, bins=30, range=(0.0, 1.0), color='#2b5c8f', edgecolor='black', alpha=0.85)

    # Highlight bins above threshold
    for count, bin_left, patch in zip(counts, bins[:-1], patches):
        if bin_left >= threshold:
            patch.set_facecolor('#28a745')

    plt.axvline(x=threshold, color='#dc3545', linestyle='--', linewidth=2.5, label=f'Detection Threshold (IoU = {threshold:.1f})')
    plt.axvline(x=np.mean(ious), color='#ffc107', linestyle='-', linewidth=2.5, label=f'Mean IoU = {np.mean(ious):.4f}')

    plt.title('Test Set IoU Score Distribution', fontsize=13, fontweight='bold')
    plt.xlabel('Intersection over Union (IoU)', fontsize=11)
    plt.ylabel('Number of Frames', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10, loc='upper left')

    pass_rate = 100.0 * np.sum(np.array(ious) >= threshold) / len(ious)
    plt.text(0.05, 0.70, f"Positive (IoU >= 0.6): {pass_rate:.1f}%\nTotal Samples: {len(ious)}",
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="gray"))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[*] IoU histogram saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Lane Segmentation Model Performance")
    parser.add_argument('--dataset-dir', type=str, default='lane_dataset/dataset_seg', help="Path to dataset_seg")
    parser.add_argument('--split-file', type=str, default='splits/test_files.txt', help="Path to test split file")
    parser.add_argument('--pred-dir', type=str, default='output/predictions_mask', help="Path to predictions_mask")
    parser.add_argument('--output-dir', type=str, default='output', help="Evaluation output directory")
    parser.add_argument('--iou-threshold', type=float, default=0.6, help="Criterion for positive detection (>= 0.6)")
    parser.add_argument('--snapshot-count', type=int, default=5, help="Number of comparison snapshots to export")
    args = parser.parse_args()

    print("=" * 65)
    print("Lane Segmentation Evaluation & Verification Pipeline")
    print(f"[*] Evaluation Threshold: IoU >= {args.iou_threshold}")
    print(f"[*] Prediction Directory: {args.pred_dir}")
    print("=" * 65)

    if not os.path.exists(args.pred_dir):
        raise FileNotFoundError(f"Predictions directory not found: {args.pred_dir}. Run predict.py first.")

    # Load test split list
    if not os.path.exists(args.split_file):
        raise FileNotFoundError(f"Split file not found: {args.split_file}.")
    with open(args.split_file, 'r', encoding='utf-8') as f:
        test_filenames = [line.strip() for line in f if line.strip()]

    img_dir = os.path.join(args.dataset_dir, 'images', 'all_images')
    lbl_dir = os.path.join(args.dataset_dir, 'labels', 'all_images')

    snapshot_dir = os.path.join(args.output_dir, 'evaluation_snapshots')
    os.makedirs(snapshot_dir, exist_ok=True)
    os.makedirs('snapshots', exist_ok=True)  # Also root snapshots folder for README

    ious = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    evaluated_samples = []

    for fname in test_filenames:
        base_name = os.path.splitext(fname)[0]
        pred_mask_path = os.path.join(args.pred_dir, f"{base_name}_mask.png")
        img_path = os.path.join(img_dir, fname)
        lbl_path = os.path.join(lbl_dir, f"{base_name}.txt")

        if not os.path.exists(pred_mask_path):
            continue

        pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
        orig_img = cv2.imread(img_path)
        h, w = orig_img.shape[:2]

        gt_mask = load_polygon_mask(lbl_path, w, h, target_classes=[3])

        iou = compute_mask_iou(pred_mask, gt_mask)
        ious.append(iou)

        pix_stats = compute_pixel_metrics(pred_mask, gt_mask)
        total_tp += pix_stats['tp']
        total_fp += pix_stats['fp']
        total_fn += pix_stats['fn']
        total_tn += pix_stats['tn']

        evaluated_samples.append({
            'filename': fname,
            'iou': iou,
            'img_bgr': orig_img,
            'gt_mask': gt_mask,
            'pred_mask': pred_mask
        })

    ious = np.array(ious)
    total_evaluated = len(ious)
    if total_evaluated == 0:
        print("[!] Error: No matching prediction files found.")
        return

    # Metrics calculation
    mean_iou = float(np.mean(ious))
    median_iou = float(np.median(ious))
    positive_count = int(np.sum(ious >= args.iou_threshold))
    positive_rate = (positive_count / total_evaluated) * 100.0

    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1_score = 2.0 * precision * recall / (precision + recall + 1e-8)
    pixel_acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + 1e-8)

    # Sort samples by IoU to extract best, median, and challenging snapshots
    evaluated_samples.sort(key=lambda x: x['iou'], reverse=True)

    # Export representative snapshots
    selected_indices = [
        0,  # Best
        len(evaluated_samples) // 4,  # Upper quartile
        len(evaluated_samples) // 2,  # Median
        3 * len(evaluated_samples) // 4,  # Lower quartile
    ]
    for s_idx, idx in enumerate(selected_indices):
        item = evaluated_samples[idx]
        base = os.path.splitext(item['filename'])[0]
        snap_path = os.path.join(snapshot_dir, f"snapshot_{s_idx+1}_{base}.png")
        create_side_by_side_snapshot(
            item['img_bgr'], item['gt_mask'], item['pred_mask'],
            item['iou'], snap_path, title_suffix=f"[{base}]"
        )
        # Also copy representative snapshot to root snapshots/ for easy markdown link
        root_snap = os.path.join('snapshots', f"lane_inference_comparison_{s_idx+1}.png")
        create_side_by_side_snapshot(
            item['img_bgr'], item['gt_mask'], item['pred_mask'],
            item['iou'], root_snap, title_suffix=f"[{base}]"
        )

    # Generate IoU Distribution Histogram
    hist_path = os.path.join(args.output_dir, 'iou_distribution.png')
    plot_iou_histogram(ious, threshold=args.iou_threshold, save_path=hist_path)
    # Also save to root snapshots/ for README
    plot_iou_histogram(ious, threshold=args.iou_threshold, save_path='snapshots/iou_distribution.png')

    # Print Report
    print("\n" + "=" * 65)
    print("EVALUATION PERFORMANCE REPORT (35% TEST SET)")
    print("=" * 65)
    print(f"Total Test Frames Evaluated:       {total_evaluated}")
    print(f"Evaluation Threshold:              IoU >= {args.iou_threshold}")
    print(f"Positive Detections (Pass):        {positive_count} / {total_evaluated} ({positive_rate:.2f}%)")
    print(f"Mean IoU (mIoU):                   {mean_iou:.4f} ({mean_iou * 100:.2f}%)")
    print(f"Median IoU:                        {median_iou:.4f}")
    print(f"Pixel Precision:                   {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Pixel Recall:                      {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1-Score (Dice Score):             {f1_score:.4f} ({f1_score * 100:.2f}%)")
    print(f"Pixel Accuracy:                    {pixel_acc:.4f} ({pixel_acc * 100:.2f}%)")
    print(f"Snapshot Comparisons Exported:     {snapshot_dir}")
    print(f"IoU Distribution Plot:             {hist_path}")
    print("=" * 65)

    # Save summary report to markdown and text
    report_md = f"""# Lane Segmentation Evaluation Report

## 1. Summary Metrics (Test Set: {total_evaluated} frames - 35% Split)
- **Positive Criterion:** $\\text{{IoU}} \\ge {args.iou_threshold}$
- **Detection Rate (IoU $\\ge$ {args.iou_threshold}):** **{positive_rate:.2f}%** ({positive_count}/{total_evaluated})
- **Mean IoU (mIoU):** **{mean_iou:.4f}** ({mean_iou*100:.2f}%)
- **Median IoU:** **{median_iou:.4f}**
- **Precision:** **{precision:.4f}** ({precision*100:.2f}%)
- **Recall:** **{recall:.4f}** ({recall*100:.2f}%)
- **F1-Score / Dice:** **{f1_score:.4f}** ({f1_score*100:.2f}%)
- **Pixel Accuracy:** **{pixel_acc:.4f}** ({pixel_acc*100:.2f}%)

## 2. Threshold Breakdown
| IoU Range | Count | Percentage |
| :--- | :--- | :--- |
| **IoU $\\ge$ 0.90** | {int(np.sum(ious >= 0.90))} | {np.sum(ious >= 0.90)/total_evaluated*100:.1f}% |
| **IoU $\\ge$ 0.80** | {int(np.sum(ious >= 0.80))} | {np.sum(ious >= 0.80)/total_evaluated*100:.1f}% |
| **IoU $\\ge$ 0.70** | {int(np.sum(ious >= 0.70))} | {np.sum(ious >= 0.70)/total_evaluated*100:.1f}% |
| **IoU $\\ge$ 0.60 (Pass)** | {positive_count} | {positive_rate:.1f}% |
| **IoU < 0.60 (Fail)** | {total_evaluated - positive_count} | {100.0 - positive_rate:.1f}% |
"""
    with open('output/evaluation_report.md', 'w', encoding='utf-8') as f:
        f.write(report_md)
    print("[*] Saved evaluation report to: output/evaluation_report.md")


if __name__ == '__main__':
    main()
