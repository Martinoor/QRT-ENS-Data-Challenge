"""
optimize.py  –  End-to-end optimised training pipeline for the QRT-ENS challenge.

All printed output is also appended to gs_log.txt (runs accumulate in the
same file, each session prefixed by a timestamp).

Usage
-----
Standard run (CV + predict, no grid search):
    python optimize.py [--save] [--no-xgb] [--temporal] [--n-splits 5]

Short grid search (~1-1.5 h on a modern multi-core CPU):
    python optimize.py --gs [--save]

Long grid search (~8-10 h):
    python optimize.py --long-gs [--save]

Notes on grid search budget allocation
---------------------------------------
CatBoost receives ~60% of the parameter combinations (LGBM ~30%, XGB ~10%)
because:
  1. CatBoost consistently outperforms LGBM in every reported CV run on this
     dataset (52.50% vs 52.09% in the notebooks, 100% optimal blend weight in
     the latest run).
  2. Ordered boosting naturally reduces within-fold leakage on panel data where
     rows at the same timestamp are correlated — a structural advantage here.
  3. CatBoost has more tunable hyperparameters with meaningful impact: depth,
     l2_leaf_reg, learning_rate, and bagging_temperature all move accuracy
     meaningfully. LGBM's main lever is num_leaves; its advantage is speed
     rather than accuracy on this problem.
  4. We still include LGBM and XGB (with smaller grids) because ensemble
     diversity has value even from individually weaker models that make
     different errors. One run is not enough to rule that out completely.

Timing estimates (421k rows, ~100 features, 3-fold GS CV / 5-fold GS CV):
  Short GS  (3-fold): CatBoost 6 configs + LGBM 4 configs ≈ 60-90 min
  Long GS   (5-fold): CatBoost 18 configs + LGBM 12 configs + XGB 8 configs ≈ 8-10 h
  Actual times will vary with hardware; adjust grid sizes in GRIDS dict if needed.
"""

import argparse
import os
import sys
import itertools
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import feature_eng
import pipeline as pl


# ---------------------------------------------------------------------------
# Tee: duplicate all stdout to a log file (append mode)
# ---------------------------------------------------------------------------

class _Tee:
    """Write to both the real stdout and a file simultaneously."""

    def __init__(self, filepath: str):
        self._file = open(filepath, "a", encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, data: str):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, *_):
        sys.stdout = self._stdout
        self._file.close()


# ---------------------------------------------------------------------------
# Grid definitions
# ---------------------------------------------------------------------------

# Short grid search (~1-1.5 h).
# Uses 3-fold CV to keep each config cheap.
# CatBoost: 6 configs  |  LGBM: 4 configs  |  XGB: none
SHORT_GRIDS = {
    "catboost": list(itertools.product(
        [5, 6, 7],           # depth          (3 values)
        [4, 12],             # l2_leaf_reg    (2 values)  → 6 configs
        [0.03],              # learning_rate  (fixed, explore in long GS)
    )),
    "lgbm": list(itertools.product(
        [31, 63],            # num_leaves     (2 values)
        [0.02, 0.03],        # learning_rate  (2 values)  → 4 configs
        [0.8],               # subsample      (fixed)
    )),
    "xgb": [],               # skip XGB in short GS to stay within time budget
}

# Long grid search (~8-10 h).
# Uses 5-fold CV for more reliable estimates.
# CatBoost: 18 configs  |  LGBM: 12 configs  |  XGB: 8 configs
LONG_GRIDS = {
    "catboost": list(itertools.product(
        [5, 6, 7],           # depth          (3 values)
        [3, 8, 15],          # l2_leaf_reg    (3 values)
        [0.01, 0.02, 0.03],  # learning_rate  (3 values)  → 27 configs
    )),
    "lgbm": list(itertools.product(
        [31, 63, 127],       # num_leaves     (3 values)
        [0.01, 0.03],        # learning_rate  (2 values)
        [0.7, 0.9],          # subsample      (2 values)  → 12 configs
    )),
    "xgb": list(itertools.product(
        [5, 6],              # max_depth      (2 values)
        [0.01, 0.03],        # learning_rate  (2 values)
        [0.7, 0.9],          # subsample      (2 values)  → 8 configs
    )),
}


# ---------------------------------------------------------------------------
# Data / feature helpers
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
    """
    Returns X_train, X_test, features_cat, features_all.

    features_cat : feature list for CatBoost (no advanced features — they hurt CB)
    features_all : feature list for LGBM / XGBoost (includes advanced features)
    Both lists share the same underlying DataFrame columns.
    """
    print("  Building benchmark features …")
    X_train, X_test, features = feature_eng.FE_benchmark(X_train, X_test)

    print("  Building rowwise features …")
    X_train, X_test, features = feature_eng.add_rowwise_features(X_train, X_test, features)

    if temporal:
        print("  Building temporal features …")
        X_train, X_test, features = feature_eng.add_temporal_FE(X_train, X_test, features)

    print("  Building cross-sectional context features …")
    X_train, X_test, features = feature_eng.add_cross_sectional_context_features(
        X_train, X_test, features
    )

    # CatBoost stops here — advanced features degrade its performance
    features_cat = list(features)

    print("  Building advanced signal features (LGBM / XGBoost only) …")
    X_train, X_test, features_all = feature_eng.add_advanced_features(X_train, X_test, features)

    print(f"  Features (CatBoost): {len(features_cat)}  |  Features (LGBM/XGB): {len(features_all)}")
    return X_train, X_test, features_cat, features_all


def save_submission(preds, index, filename="submission_optimized.csv"):
    os.makedirs("submissions", exist_ok=True)
    path = os.path.join("submissions", filename)
    pd.DataFrame({"target": preds.astype(int)}, index=index).to_csv(path)
    print(f"  Submission saved → {path}")


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def run_grid_search(
    X_train,
    y_train,
    features_cat,
    features_all,
    mode: str = "short",
    use_xgb: bool = True,
):
    """
    Run hyperparameter grid search using OOF accuracy as the criterion.

    Parameters
    ----------
    features_cat : feature list for CatBoost (no advanced features)
    features_all : feature list for LGBM / XGBoost (includes advanced features)
    mode : 'short' or 'long'
    use_xgb : if False, XGB grid is skipped even in long mode.

    Returns
    -------
    best_cat_params, best_lgbm_params, best_xgb_params  (dicts, or None)
    """
    grids = SHORT_GRIDS if mode == "short" else LONG_GRIDS
    n_splits_gs = 3 if mode == "short" else 5

    print(f"\n{'=' * 60}")
    print(f"  GRID SEARCH  mode={mode}  cv_folds={n_splits_gs}")
    print(f"  CatBoost configs : {len(grids['catboost'])}")
    print(f"  LGBM configs     : {len(grids['lgbm'])}")
    print(f"  XGB configs      : {len(grids['xgb'])}")
    print(f"{'=' * 60}")

    # ---- CatBoost --------------------------------------------------------
    best_cat_params = None
    best_cat_score = -np.inf
    cat_results = []

    if grids["catboost"]:
        print(f"\n--- CatBoost grid ({len(grids['catboost'])} configs) ---")
        for idx, (depth, l2, lr) in enumerate(grids["catboost"], 1):
            label = f"depth={depth}, l2={l2}, lr={lr}"
            print(f"  [{idx:02d}/{len(grids['catboost'])}]  {label} …", flush=True)
            _, oof, scores = pl.catboost_cv_grouped(
                X_train, y_train, features_cat,
                n_splits=n_splits_gs,
                iterations=2000,
                learning_rate=lr,
                depth=depth,
                l2_leaf_reg=l2,
                early_stopping_rounds=200,
                see_folds=False,
            )
            mean_acc = float(np.mean(scores))
            oof_acc = float(accuracy_score(
                (y_train["target"].values > 0).astype(int),
                (oof > 0.5).astype(int),
            ))
            cat_results.append(dict(depth=depth, l2_leaf_reg=l2,
                                    learning_rate=lr, mean_fold_acc=mean_acc,
                                    oof_acc=oof_acc))
            print(f"         mean_fold={mean_acc * 100:.2f}%  oof={oof_acc * 100:.2f}%")
            if oof_acc > best_cat_score:
                best_cat_score = oof_acc
                best_cat_params = dict(depth=depth, l2_leaf_reg=l2, learning_rate=lr)
                print(f"         *** new best CatBoost ***")

        print(f"\n  Best CatBoost: {best_cat_params}  oof_acc={best_cat_score * 100:.2f}%")

        # Print ranked table
        cat_results.sort(key=lambda r: r["oof_acc"], reverse=True)
        print("\n  CatBoost ranking:")
        for r in cat_results:
            print(f"    depth={r['depth']} l2={r['l2_leaf_reg']} lr={r['learning_rate']}"
                  f"  oof={r['oof_acc'] * 100:.2f}%  mean_fold={r['mean_fold_acc'] * 100:.2f}%")

    # ---- LGBM ------------------------------------------------------------
    best_lgbm_params = None
    best_lgbm_score = -np.inf
    lgbm_results = []

    if grids["lgbm"]:
        print(f"\n--- LGBM grid ({len(grids['lgbm'])} configs) ---")
        for idx, (num_leaves, lr, subsample) in enumerate(grids["lgbm"], 1):
            label = f"num_leaves={num_leaves}, lr={lr}, sub={subsample}"
            print(f"  [{idx:02d}/{len(grids['lgbm'])}]  {label} …", flush=True)
            _, oof, scores = pl.lgbm_cv_grouped(
                X_train, y_train, features_all,
                n_splits=n_splits_gs,
                num_boost_round=3000,
                learning_rate=lr,
                num_leaves=num_leaves,
                subsample=subsample,
                colsample_bytree=0.8,
                min_child_samples=20,
                early_stopping_rounds=200,
                see_folds=False,
            )
            mean_acc = float(np.mean(scores))
            oof_acc = float(accuracy_score(
                (y_train["target"].values > 0).astype(int),
                (oof > 0.5).astype(int),
            ))
            lgbm_results.append(dict(num_leaves=num_leaves, learning_rate=lr,
                                     subsample=subsample, mean_fold_acc=mean_acc,
                                     oof_acc=oof_acc))
            print(f"         mean_fold={mean_acc * 100:.2f}%  oof={oof_acc * 100:.2f}%")
            if oof_acc > best_lgbm_score:
                best_lgbm_score = oof_acc
                best_lgbm_params = dict(num_leaves=num_leaves, learning_rate=lr,
                                        subsample=subsample)
                print(f"         *** new best LGBM ***")

        print(f"\n  Best LGBM: {best_lgbm_params}  oof_acc={best_lgbm_score * 100:.2f}%")

        lgbm_results.sort(key=lambda r: r["oof_acc"], reverse=True)
        print("\n  LGBM ranking:")
        for r in lgbm_results:
            print(f"    leaves={r['num_leaves']} lr={r['learning_rate']} sub={r['subsample']}"
                  f"  oof={r['oof_acc'] * 100:.2f}%  mean_fold={r['mean_fold_acc'] * 100:.2f}%")

    # ---- XGBoost ---------------------------------------------------------
    best_xgb_params = None
    best_xgb_score = -np.inf
    xgb_results = []

    if grids["xgb"] and use_xgb and pl._XGB_AVAILABLE:
        print(f"\n--- XGBoost grid ({len(grids['xgb'])} configs) ---")
        for idx, (depth, lr, subsample) in enumerate(grids["xgb"], 1):
            label = f"depth={depth}, lr={lr}, sub={subsample}"
            print(f"  [{idx:02d}/{len(grids['xgb'])}]  {label} …", flush=True)
            _, oof, scores = pl.xgb_cv_grouped(
                X_train, y_train, features_all,
                n_splits=n_splits_gs,
                n_estimators=3000,
                learning_rate=lr,
                max_depth=depth,
                subsample=subsample,
                colsample_bytree=0.8,
                early_stopping_rounds=200,
                see_folds=False,
            )
            mean_acc = float(np.mean(scores))
            oof_acc = float(accuracy_score(
                (y_train["target"].values > 0).astype(int),
                (oof > 0.5).astype(int),
            ))
            xgb_results.append(dict(max_depth=depth, learning_rate=lr,
                                    subsample=subsample, mean_fold_acc=mean_acc,
                                    oof_acc=oof_acc))
            print(f"         mean_fold={mean_acc * 100:.2f}%  oof={oof_acc * 100:.2f}%")
            if oof_acc > best_xgb_score:
                best_xgb_score = oof_acc
                best_xgb_params = dict(max_depth=depth, learning_rate=lr,
                                       subsample=subsample)
                print(f"         *** new best XGBoost ***")

        print(f"\n  Best XGBoost: {best_xgb_params}  oof_acc={best_xgb_score * 100:.2f}%")

        xgb_results.sort(key=lambda r: r["oof_acc"], reverse=True)
        print("\n  XGBoost ranking:")
        for r in xgb_results:
            print(f"    depth={r['max_depth']} lr={r['learning_rate']} sub={r['subsample']}"
                  f"  oof={r['oof_acc'] * 100:.2f}%  mean_fold={r['mean_fold_acc'] * 100:.2f}%")
    elif grids["xgb"] and not pl._XGB_AVAILABLE:
        print("\n  XGBoost grid skipped (not installed).")

    print(f"\n{'=' * 60}")
    print("  GRID SEARCH COMPLETE")
    if best_cat_params:
        print(f"  Best CatBoost : {best_cat_params}  ({best_cat_score * 100:.2f}%)")
    if best_lgbm_params:
        print(f"  Best LGBM     : {best_lgbm_params}  ({best_lgbm_score * 100:.2f}%)")
    if best_xgb_params:
        print(f"  Best XGBoost  : {best_xgb_params}  ({best_xgb_score * 100:.2f}%)")
    print(f"{'=' * 60}")

    return best_cat_params, best_lgbm_params, best_xgb_params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Default hyperparameters (used when no grid search is requested)
_DEFAULT_CAT_PARAMS = dict(iterations=2000, learning_rate=0.02, depth=6, l2_leaf_reg=8.0)
_DEFAULT_LGBM_PARAMS = dict(num_boost_round=3000, learning_rate=0.02, num_leaves=63,
                             subsample=0.8, colsample_bytree=0.8, min_child_samples=20)
_DEFAULT_XGB_PARAMS = dict(n_estimators=3000, learning_rate=0.02, max_depth=6,
                            subsample=0.8, colsample_bytree=0.8)


def main():
    parser = argparse.ArgumentParser(description="Optimised QRT-ENS pipeline")
    parser.add_argument("--save", action="store_true",
                        help="Save submission CSV after training")
    parser.add_argument("--no-xgb", action="store_true",
                        help="Skip XGBoost model")
    parser.add_argument("--temporal", action="store_true",
                        help="Use X_train_reconstructed.csv + temporal features")
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Number of CV folds for final training (default 5)")
    parser.add_argument("--gs", action="store_true",
                        help="Run short grid search (~1-1.5 h) before final training")
    parser.add_argument("--long-gs", action="store_true",
                        help="Run long grid search (~8-10 h) before final training")
    parser.add_argument("--log", type=str, default="gs_log.txt",
                        help="Path to log file (default: gs_log.txt, appended)")
    args = parser.parse_args()

    use_xgb = not args.no_xgb and pl._XGB_AVAILABLE

    with _Tee(args.log):
        _run(args, use_xgb)


def _run(args, use_xgb):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'#' * 70}")
    print(f"# RUN STARTED  {timestamp}")
    flags = []
    if args.gs:
        flags.append("--gs")
    if args.long_gs:
        flags.append("--long-gs")
    if args.temporal:
        flags.append("--temporal")
    if not use_xgb:
        flags.append("--no-xgb")
    flags.append(f"--n-splits {args.n_splits}")
    print(f"# FLAGS: {' '.join(flags) if flags else '(none)'}")
    print(f"{'#' * 70}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n=== Loading data ===")
    X_train, y_train, X_test = load_data(temporal=args.temporal)
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    print("\n=== Feature engineering ===")
    X_train, X_test, features_cat, features_all = build_features(X_train, X_test, temporal=args.temporal)

    # ------------------------------------------------------------------
    # 3. Grid search (optional)
    # ------------------------------------------------------------------
    cat_params = dict(_DEFAULT_CAT_PARAMS)
    lgbm_params = dict(_DEFAULT_LGBM_PARAMS)
    xgb_params = dict(_DEFAULT_XGB_PARAMS)

    if args.gs or args.long_gs:
        gs_mode = "long" if args.long_gs else "short"
        best_cat, best_lgbm, best_xgb = run_grid_search(
            X_train, y_train, features_cat, features_all,
            mode=gs_mode,
            use_xgb=use_xgb,
        )
        # Merge best GS params into the defaults (keep non-searched params intact)
        if best_cat:
            cat_params.update(best_cat)
        if best_lgbm:
            lgbm_params.update(best_lgbm)
        if best_xgb and use_xgb:
            xgb_params.update(best_xgb)

    # ------------------------------------------------------------------
    # 4. Final CV with best / default parameters
    # ------------------------------------------------------------------
    print("\n=== Final CV  (LightGBM) ===")
    print(f"  params: {lgbm_params}")
    lgbm_models, lgbm_oof, lgbm_scores = pl.lgbm_cv_grouped(
        X_train, y_train, features_all,
        n_splits=args.n_splits,
        early_stopping_rounds=200,
        see_folds=True,
        **lgbm_params,
    )

    print("\n=== Final CV  (CatBoost) ===")
    print(f"  params: {cat_params}")
    cat_models, cat_oof, cat_scores = pl.catboost_cv_grouped(
        X_train, y_train, features_cat,
        n_splits=args.n_splits,
        early_stopping_rounds=200,
        see_folds=True,
        **cat_params,
    )

    xgb_models, xgb_oof, xgb_scores = None, None, None
    if use_xgb:
        print("\n=== Final CV  (XGBoost) ===")
        print(f"  params: {xgb_params}")
        xgb_models, xgb_oof, xgb_scores = pl.xgb_cv_grouped(
            X_train, y_train, features_all,
            n_splits=args.n_splits,
            early_stopping_rounds=200,
            see_folds=True,
            **xgb_params,
        )

    # ------------------------------------------------------------------
    # 5. Optimal ensemble blend
    # ------------------------------------------------------------------
    print("\n=== Optimal ensemble blend ===")
    oof_list = [lgbm_oof, cat_oof]
    model_names = ["lgbm", "catboost"]
    if use_xgb and xgb_oof is not None:
        oof_list.append(xgb_oof)
        model_names.append("xgb")

    best_weights, _ = pl.find_optimal_blend(oof_list, y_train, model_names)
    blended_oof = sum(w * oof for w, oof in zip(best_weights, oof_list))

    # ------------------------------------------------------------------
    # 6. Optimal decision threshold
    # ------------------------------------------------------------------
    print("\n=== Optimal decision threshold ===")
    best_thresh, thresh_acc = pl.find_optimal_threshold(blended_oof, y_train)

    # ------------------------------------------------------------------
    # 7. Test predictions
    # ------------------------------------------------------------------
    print("\n=== Test predictions ===")
    models_list = [
        (lgbm_models, "lgbm", features_all),
        (cat_models, "catboost", features_cat),
    ]
    if use_xgb and xgb_models is not None:
        models_list.append((xgb_models, "xgb", features_all))

    _, test_proba = pl.predict_ensemble(models_list, X_test, features_cat, best_weights)
    test_preds = (test_proba >= best_thresh).astype(int)
    print(f"  Positive prediction rate: {test_preds.mean() * 100:.1f}%")

    # ------------------------------------------------------------------
    # 8. Per-group OOF accuracy
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
    # 9. Summary
    # ------------------------------------------------------------------
    print("\n=== Summary ===")
    print(f"  LGBM OOF accuracy:     {np.mean(lgbm_scores) * 100:.2f}%")
    print(f"  CatBoost OOF accuracy: {np.mean(cat_scores) * 100:.2f}%")
    if xgb_scores is not None:
        print(f"  XGBoost OOF accuracy:  {np.mean(xgb_scores) * 100:.2f}%")
    weight_str = ", ".join(f"{n}={w:.2f}" for n, w in zip(model_names, best_weights))
    print(f"  Best blend:  {weight_str}")
    print(f"  Best thresh: {best_thresh:.2f}")
    print(f"  Blended OOF accuracy (at thresh): {thresh_acc * 100:.2f}%")
    print(f"\n  Final CatBoost params : {cat_params}")
    print(f"  Final LGBM params     : {lgbm_params}")
    if use_xgb:
        print(f"  Final XGBoost params  : {xgb_params}")

    # ------------------------------------------------------------------
    # 10. Save submission
    # ------------------------------------------------------------------
    if args.save:
        gs_tag = "_gs-long" if args.long_gs else ("_gs-short" if args.gs else "")
        fname = f"submission_optimized{gs_tag}.csv"
        save_submission(test_preds, X_test.index, filename=fname)

    end_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n# RUN FINISHED  {end_ts}")
    print(f"{'#' * 70}\n")

    return test_preds, test_proba


if __name__ == "__main__":
    main()
