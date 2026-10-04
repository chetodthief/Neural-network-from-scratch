# Lane Segmentation Evaluation Report

## 1. Summary Metrics (Test Set: 350 frames - 35% Split)
- **Positive Criterion:** $\text{IoU} \ge 0.6$
- **Detection Rate (IoU $\ge$ 0.6):** **100.00%** (350/350)
- **Mean IoU (mIoU):** **0.9659** (96.59%)
- **Median IoU:** **0.9686**
- **Precision:** **0.9947** (99.47%)
- **Recall:** **0.9708** (97.08%)
- **F1-Score / Dice:** **0.9826** (98.26%)
- **Pixel Accuracy:** **0.9821** (98.21%)

## 2. Threshold Breakdown
| IoU Range | Count | Percentage |
| :--- | :--- | :--- |
| **IoU $\ge$ 0.90** | 350 | 100.0% |
| **IoU $\ge$ 0.80** | 350 | 100.0% |
| **IoU $\ge$ 0.70** | 350 | 100.0% |
| **IoU $\ge$ 0.60 (Pass)** | 350 | 100.0% |
| **IoU < 0.60 (Fail)** | 0 | 0.0% |
