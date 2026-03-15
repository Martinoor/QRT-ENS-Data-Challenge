"""
optimize.py  –  End-to-end optimised training pipeline for the QRT-ENS challenge.

Improvements over the original pipeline:
  1. All features (benchmark + rowwise + temporal + advanced + cross-sectional).
  2. GroupKFold by timestamp – no cross-date leakage.
  3. Binary classification objective (binary cross-entropy / Logloss) instead
     of regression MSE → better calibrated probabilities.
  4. Three-model ensemble: LightGBM + CatBoost + XGBoost.
  5. Grid-searched blend weights using OOF predictions.
  6. Optimal decision threshold search on OOF predictions.
  7. Optional per-group model evaluation for diagnostics.

Usage:
    python optimize.py                    # standard cross-validation + predict
    python optimize.py --save             # also write submission CSV
    python optimize.py --no-xgb          # skip XGBoost (faster)
    python optimize.py --n-splits 5      # number of CV folds (default 5)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import feature_eng
import pipeline as pl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(temporal: bool = False):
    if temporal:
        X_train = pd.read_csv("data/X_train_reconstructed.csv", index_col="ROW_ID")
    else:
        X_train = pd.read_csv("data/X_train.csv", index_col="ROW_ID")
    y_train = pd.read_csv("data/y_train.csv", index_col="ROW_ID")
    X_test = pd.read_csv("data/X_test.csv", index_col="ROW_ID")
    return X_train, y_train, X_test


def build_features(X_train, X_test, temporal: bool = False):
    print("Building benchmark features …")
    X_train, X_test, features = feature_eng.FE_benchmark(X_train, X_test)

    print("Building rowwise features …")
    X_train, X_test, features = feature_eng.add_rowwise_features(X_train, X_test, features)

    print("Building advanced signal features …")
    X_train, X_test, features = feature_eng.add_advanced_features(X_train, X_test, features)

    if temporal:
        print("Building temporal features …")
        X_train, X_test, features = feature_eng.add_temporal_FE(X_train, X_test, features)

    print("Building cross-sectional context features …")
    X_train, X_test, features = feature_eng.add_cross_sectional_context_features(
        X_train, X_test, features
    )

    print(f"Total features: {len(features)}")
    return X_train, X_test, features


def save_submission(preds, index, filename="submission_optimized.csv"):
    os.makedirs("submissions", exist_ok=True)
    path = os.path.join("submissions", filename)
    df = pd.DataFrame({"target": preds.astype(int)}, index=index)
    df.index.name = "ROW_ID"
    df.to_csv(path)
    print(f"Submission saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Optimised QRT-ENS pipeline")
    parser.add_argument("--save", action="store_true", help="Save submission CSV")
    parser.add_argument("--no-xgb", action="store_true", help="Skip XGBoost model")
    parser.add_argument("--temporal", action="store_true",
                        help="Use time-series reconstructed data + temporal features")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    args = parser.parse_args()

    use_xgb = not args.no_xgb and pl._XGB_AVAILABLE
    if args.no_xgb:
        print("XGBoost skipped (--no-xgb flag).")
    elif not pl._XGB_AVAILABLE:
        print("XGBoost not installed; skipping. Install with: pip install xgboost")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n=== Loading data ===")
    X_train, y_train, X_test = load_data(temporal=args.temporal)
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    print("\n=== Feature engineering ===")
    X_train, X_test, features = build_features(X_train, X_test, temporal=args.temporal)

    # ------------------------------------------------------------------
    # 3. LightGBM GroupKFold CV
    # ------------------------------------------------------------------
    print("\n=== LightGBM (GroupKFold by TS) ===")
    lgbm_models, lgbm_oof, lgbm_scores = pl.lgbm_cv_grouped(
        X_train,
        y_train,
        features,
        n_splits=args.n_splits,
        num_boost_round=3000,
        learning_rate=0.02,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        early_stopping_rounds=200,
        see_folds=True,
    )

    # ------------------------------------------------------------------
    # 4. CatBoost GroupKFold CV
    # ------------------------------------------------------------------
    print("\n=== CatBoost (GroupKFold by TS) ===")
    cat_models, cat_oof, cat_scores = pl.catboost_cv_grouped(
        X_train,
        y_train,
        features,
        n_splits=args.n_splits,
        iterations=2000,
        learning_rate=0.02,
        depth=6,
        l2_leaf_reg=8.0,
        early_stopping_rounds=200,
        see_folds=True,
    )

    # ------------------------------------------------------------------
    # 5. XGBoost GroupKFold CV (optional)
    # ------------------------------------------------------------------
    xgb_models, xgb_oof = None, None
    if use_xgb:
        print("\n=== XGBoost (GroupKFold by TS) ===")
        xgb_models, xgb_oof, xgb_scores = pl.xgb_cv_grouped(
            X_train,
            y_train,
            features,
            n_splits=args.n_splits,
            n_estimators=3000,
            learning_rate=0.02,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=200,
            see_folds=True,
        )

    # ------------------------------------------------------------------
    # 6. Find optimal blend weights from OOF
    # ------------------------------------------------------------------
    print("\n=== Optimal ensemble blend ===")
    oof_list = [lgbm_oof, cat_oof]
    model_names = ["lgbm", "catboost"]
    if use_xgb and xgb_oof is not None:
        oof_list.append(xgb_oof)
        model_names.append("xgb")

    best_weights, best_blend_acc = pl.find_optimal_blend(oof_list, y_train, model_names)

    # Compute blended OOF for threshold search
    blended_oof = sum(w * oof for w, oof in zip(best_weights, oof_list))

    # ------------------------------------------------------------------
    # 7. Find optimal decision threshold
    # ------------------------------------------------------------------
    print("\n=== Optimal decision threshold ===")
    best_thresh, thresh_acc = pl.find_optimal_threshold(blended_oof, y_train)

    # ------------------------------------------------------------------
    # 8. Generate test predictions
    # ------------------------------------------------------------------
    print("\n=== Test predictions ===")
    models_list = [
        (lgbm_models, "lgbm"),
        (cat_models, "catboost"),
    ]
    if use_xgb and xgb_models is not None:
        models_list.append((xgb_models, "xgb"))

    _, test_proba = pl.predict_ensemble(models_list, X_test, features, best_weights)
    test_preds = (test_proba >= best_thresh).astype(int)

    pos_rate = test_preds.mean() * 100
    print(f"  Positive prediction rate: {pos_rate:.1f}%")

    # ------------------------------------------------------------------
    # 9. Per-group OOF accuracy diagnostic
    # ------------------------------------------------------------------
    if "GROUP" in X_train.columns:
        print("\n=== Per-group OOF accuracy ===")
        y_bin = (y_train["target"].values > 0).astype(int)
        oof_bin = (blended_oof >= best_thresh).astype(int)
        for grp in sorted(X_train["GROUP"].unique()):
            mask = X_train["GROUP"].values == grp
            grp_acc = accuracy_score(y_bin[mask], oof_bin[mask]) * 100
            print(f"  GROUP={grp}: {grp_acc:.2f}%  (n={mask.sum()})")

    # ------------------------------------------------------------------
    # 10. Summary
    # ------------------------------------------------------------------
    print("\n=== Summary ===")
    print(f"  LGBM OOF accuracy:     {np.mean(lgbm_scores) * 100:.2f}%")
    print(f"  CatBoost OOF accuracy: {np.mean(cat_scores) * 100:.2f}%")
    if use_xgb and xgb_models is not None:
        print(f"  XGBoost OOF accuracy:  {np.mean(xgb_scores) * 100:.2f}%")
    weight_str = ", ".join(f"{n}={w:.2f}" for n, w in zip(model_names, best_weights))
    print(f"  Best blend:  {weight_str}")
    print(f"  Best thresh: {best_thresh:.2f}")
    print(f"  Blended OOF accuracy (at thresh): {thresh_acc * 100:.2f}%")

    # ------------------------------------------------------------------
    # 11. Save submission
    # ------------------------------------------------------------------
    if args.save:
        save_submission(test_preds, X_test.index)

    return test_preds, test_proba


if __name__ == "__main__":
    main()
