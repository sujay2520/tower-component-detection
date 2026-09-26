# 📊 Model Validation Benchmarks & Metric Methodology

## Overview
Comprehensive evaluation of the trained **YOLOv8n** model on the verified LabelImg telecommunication tower dataset.

## Summary Table

| Metric | Measured Value | Standard Benchmark Target |
| :--- | :--- | :--- |
| **mAP @ 0.50** | **98.88%** | > 85.0% |
| **mAP @ 0.50–0.95** | **62.66%** | > 50.0% |
| **Precision** | **58.09%** | > 50.0% |
| **Recall** | **100.0%** | > 90.0% |
| **F1 Score** | **0.960** | > 0.850 |
| **Optimal Conf Threshold** | **0.226** | N/A |

## Class Breakdown

### Supporting Tower (Open Steel Lattice / Truss)
- **Instances Evaluated**: 12
- **mAP @ 0.50**: 98.30%
- **mAP @ 0.50–0.95**: 58.91%
- **Precision**: 62.50%
- **Recall**: 100.0%

### Monopole Tower (Cylindrical Mast)
- **Instances Evaluated**: 3
- **mAP @ 0.50**: 99.50%
- **mAP @ 0.50–0.95**: 66.40%
- **Precision**: 53.70%
- **Recall**: 100.0%

## Confusion Matrix Analysis
- Supporting Tower predicted as Supporting Tower: **100% (12/12)**
- Monopole Tower predicted as Monopole Tower: **100% (3/3)**
- Cross-Class Confusion: **0.0%**
