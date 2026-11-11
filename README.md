# Machine Learning Midterm Project: Classification and Clustering Benchmarks

**Course:** Machine Learning Midterm Assignment  
**Datasets:** Breast Cancer Wisconsin (Diagnostic) & Wine Recognition  
**Author:** Nursena Selman  

---

## 1. Project Overview

This repository contains an end-to-end, reproducible machine learning benchmark evaluating supervised classification and unsupervised clustering algorithms across two distinct tabular benchmark datasets:
1. **Breast Cancer Wisconsin (Diagnostic):** Binary classification (569 samples, 30 continuous real-valued features).
2. **Wine Recognition:** Multi-class classification (178 samples, 13 continuous chemical attributes, 3 cultivar classes).

The experimental setup strictly follows a protocol designed to prevent data leakage, evaluate feature scale sensitivity, analyze seed variation, and perform model selection based on validation performance before evaluating on an isolated test partition.

---

## 2. Repository Structure

```text
.
├── notebooks/
│   ├── breast_cancer.ipynb   # Complete 10-section analysis for Breast Cancer
│   └── wine.ipynb            # Complete 10-section analysis for Wine
├── src/
│   └── utils.py              # Modular data loading, partitioning, and scaling utilities
├── figures/                  # High-resolution (300 DPI) visual artifacts
├── results/                  # All standardized tabular CSV outputs
└── README.md                 # Project documentation and summary

---

## 3. Environment & Package Versions

- Python: 3.10+
- numpy: >= 1.24.0
- pandas: >= 2.0.0
- scipy: >= 1.10.0
- scikit-learn: >= 1.3.0
- matplotlib: >= 3.7.0
- seaborn: >= 0.12.0

---

## 4. Generated Artifacts & CSV Files

The pipeline generates the following 6 unified CSV artifacts with a `dataset` column:
- `classifier_runs.csv` (132 rows: 66 Wine + 66 Breast Cancer runs)
- `classifier_summary.csv` (44 rows: 22 configurations ranked for each dataset)
- `classifier_test.csv` (2 rows: final frozen test evaluations)
- `kmeans_runs.csv` (12 rows: unsupervised clustering runs across seeds)
- `kmeans_summary.csv` (4 rows: scaled vs. unscaled summary statistics)
- `kmeans_test.csv` (4 rows: test set clustering evaluations)

Additionally, the test partition sample indices are exported to `wine_test_indices.txt` and `bc_test_indices.txt`.