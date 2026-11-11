"""
Helper utility functions for ML Midterm Experiments
Covers data loading, splitting, scaling, metric calculation, and reporting.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

SEEDS = [42, 52, 62]

def load_data(dataset_name):
    if dataset_name == "breast_cancer":
        raw = load_breast_cancer()
    elif dataset_name == "wine":
        raw = load_wine()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    X = pd.DataFrame(raw.data, columns=raw.feature_names)
    y = pd.Series(raw.target, name="target")
    return X, y, raw.target_names

def prepare_dataset_splits(X, y):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    splits = {'test': (X_test, y_test), 'by_seed': {}}
    for s in SEEDS:
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=s
        )
        splits['by_seed'][s] = {
            'X_train': X_tr, 'X_val': X_va,
            'y_train': y_tr, 'y_val': y_va
        }
    return splits

def scale_subsets(splits):
    scaled_data = {'by_seed': {}}
    scalers = {}
    for s in SEEDS:
        scaler = StandardScaler()
        X_tr = splits['by_seed'][s]['X_train']
        X_va = splits['by_seed'][s]['X_val']
        X_tr_sc = pd.DataFrame(scaler.fit_transform(X_tr), index=X_tr.index, columns=X_tr.columns)
        X_va_sc = pd.DataFrame(scaler.transform(X_va), index=X_va.index, columns=X_va.columns)
        scaled_data['by_seed'][s] = {'X_train_scaled': X_tr_sc, 'X_val_scaled': X_va_sc}
        scalers[s] = scaler
    X_te, _ = splits['test']
    scaled_data['test_scaled'] = pd.DataFrame(scalers[42].transform(X_te), index=X_te.index, columns=X_te.columns)
    return scaled_data, scalers