import pandas as pd
import numpy as np
import seaborn as sns

sns.set()
import matplotlib.pyplot as plt

from sklearn import linear_model
import lightgbm as lgbm
import catboost
try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, GroupKFold

import feature_eng


# ---------------------------- LGBM CV FUNCTIONS -----------------------------


def lgbm_cv(
    X_train,
    y_train,
    X_test,
    features,
    num_boost_round=500,
    learning_rate=0.05,
    max_depth=6,
    see_folds=False,
):
    features_lgbm = features
    # A quite large number of trees with low depth to prevent overfits
    lgbm_params = {
        "objective": "mse",
        "metric": "mse",
        "num_threads": 50,
        "seed": 42,
        "verbosity": -1,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
    }
    NUM_BOOST_ROUND = num_boost_round

    train_dates = X_train["TS"].unique()
    test_dates = X_test["TS"].unique()

    n_splits = 8
    scores_lgbm = []
    models_lgbm = []

    splits = KFold(
        n_splits=n_splits,
        random_state=0,
        shuffle=True,
    ).split(train_dates)

    for i, (local_train_dates_ids, local_test_dates_ids) in enumerate(splits):
        local_train_dates = train_dates[local_train_dates_ids]
        local_test_dates = train_dates[local_test_dates_ids]

        local_train_ids = X_train["TS"].isin(local_train_dates)
        local_test_ids = X_train["TS"].isin(local_test_dates)

        X_local_train = X_train.loc[local_train_ids, [x for x in features_lgbm]]
        y_local_train = y_train.loc[local_train_ids, "target"]

        X_local_test = X_train.loc[local_test_ids, [x for x in features_lgbm]]
        y_local_test = y_train.loc[local_test_ids, "target"]

        X_local_train = X_local_train
        X_local_test = X_local_test

        train_data = lgbm.Dataset(X_local_train, label=y_local_train.values)

        model_lgbm = lgbm.train(
            lgbm_params, train_data, num_boost_round=NUM_BOOST_ROUND
        )

        y_local_pred = model_lgbm.predict(
            X_local_test.values, num_threads=lgbm_params["num_threads"]
        )

        models_lgbm.append(model_lgbm)
        score = accuracy_score(
            (y_local_test > 0).astype(int), (y_local_pred > 0).astype(int)
        )
        scores_lgbm.append(score)

        if see_folds:
            print(f"Fold {i+1} - Accuracy: {score* 100:.2f}%")

    mean = np.mean(scores_lgbm) * 100
    std = np.std(scores_lgbm) * 100

    u = mean + std
    l = mean - std

    print(f"Accuracy: {mean:.2f}% [{l:.2f} ; {u:.2f}] (+- {std:.2f})")
    return models_lgbm, scores_lgbm


def lgbm_cv_temporal(
    X_train,
    y_train,
    features,
    n_splits=8,
    num_boost_round=500,
    learning_rate=1e-2,
    max_depth=3,
    see_folds=False,
):
    features_lgbm = list(features)

    lgbm_params = {
        "objective": "mse",
        "metric": "mse",
        "num_threads": 50,
        "seed": 42,
        "verbosity": -1,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
    }
    NUM_BOOST_ROUND = num_boost_round

    scores_lgbm = []
    models_lgbm = []

    # safer: sort unique TS values using Python sorting
    train_dates = sorted(X_train["TS"].unique())
    n_dates = len(train_dates)

    test_block_size = n_dates // (n_splits + 1)

    for i in range(n_splits):
        train_end = test_block_size * (i + 1)
        test_end = test_block_size * (i + 2) if i < n_splits - 1 else n_dates

        local_train_dates = train_dates[:train_end]
        local_test_dates = train_dates[train_end:test_end]

        if len(local_test_dates) == 0:
            break

        local_train_ids = X_train["TS"].isin(local_train_dates)
        local_test_ids = X_train["TS"].isin(local_test_dates)

        X_local_train = X_train.loc[local_train_ids, features_lgbm]
        y_local_train = y_train.loc[local_train_ids, "target"]

        X_local_test = X_train.loc[local_test_ids, features_lgbm]
        y_local_test = y_train.loc[local_test_ids, "target"]

        train_data = lgbm.Dataset(X_local_train, label=y_local_train.values)

        model_lgbm = lgbm.train(
            lgbm_params,
            train_data,
            num_boost_round=NUM_BOOST_ROUND,
        )

        y_local_pred = model_lgbm.predict(
            X_local_test,
            num_threads=lgbm_params["num_threads"],
        )

        score = accuracy_score(
            (y_local_test > 0).astype(int),
            (y_local_pred > 0).astype(int),
        )

        models_lgbm.append(model_lgbm)
        scores_lgbm.append(score)

        train_min = min(local_train_dates)
        train_max = max(local_train_dates)
        test_min = min(local_test_dates)
        test_max = max(local_test_dates)

        if see_folds:
            print(
                f"Fold {i+1} | "
                f"train: [{train_min} -> {train_max}] | "
                f"test: [{test_min} -> {test_max}] | "
                f"Accuracy: {score * 100:.2f}%"
            )

    mean = np.mean(scores_lgbm) * 100
    std = np.std(scores_lgbm) * 100

    if see_folds:
        print(
            f"Accuracy: {mean:.2f}% [{mean - std:.2f} ; {mean + std:.2f}] (+- {std:.2f})"
        )

    return models_lgbm, scores_lgbm


# ---------------------------- CATBOOST CV FUNCTIONS -----------------------------


def catboost_cv(
    X_train,
    y_train,
    X_test,
    features,
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    see_folds=False,
):
    # A quite large number of trees with low depth to prevent overfits

    # this CV only work on the origina X_train and not on the X_train_reconstructed
    # because it does not preserve the same TS distribution in the folds,

    catboost_features = features

    catboost_params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": iterations,
        "learning_rate": learning_rate,
        "depth": depth,
        "verbose": False,
    }

    train_dates = X_train["TS"].unique()
    scores_catboost = []
    models_catboost = []

    splits = KFold(
        n_splits=8,
        random_state=0,
        shuffle=True,
    ).split(train_dates)

    for i, (local_train_dates_ids, local_test_dates_ids) in enumerate(splits):
        local_train_dates = train_dates[local_train_dates_ids]
        local_test_dates = train_dates[local_test_dates_ids]

        local_train_ids = X_train["TS"].isin(local_train_dates)
        local_test_ids = X_train["TS"].isin(local_test_dates)

        X_local_train = X_train.loc[local_train_ids, catboost_features]
        y_local_train = y_train.loc[local_train_ids, "target"]

        X_local_test = X_train.loc[local_test_ids, catboost_features]
        y_local_test = y_train.loc[local_test_ids, "target"]

        train_data = catboost.Pool(X_local_train, label=y_local_train.values)

        model_catboost = catboost.CatBoostRegressor(**catboost_params)
        model_catboost.fit(train_data)

        y_local_pred = model_catboost.predict(X_local_test)

        models_catboost.append(model_catboost)
        score = accuracy_score(
            (y_local_test > 0).astype(int), (y_local_pred > 0).astype(int)
        )
        scores_catboost.append(score)

        if see_folds:
            print(f"Fold {i+1} - Accuracy: {score * 100:.2f}%")

    mean_catboost = np.mean(scores_catboost) * 100
    std_catboost = np.std(scores_catboost) * 100

    print(
        f"Accuracy: {mean_catboost:.2f}% "
        f"[{mean_catboost - std_catboost:.2f} ; {mean_catboost + std_catboost:.2f}] "
        f"(+- {std_catboost:.2f})"
    )
    return models_catboost, scores_catboost


def catboost_cv_temporal(
    X_train,
    y_train,
    features,
    n_splits=8,
    iterations=1000,
    depth=6,
    learning_rate=0.05,
    see_folds=False,
):
    features_catboost = list(features)

    catboost_params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": iterations,
        "learning_rate": learning_rate,
        "depth": depth,
        "verbose": False,
        "random_seed": 42,
    }

    scores_catboost = []
    models_catboost = []

    train_dates = sorted(X_train["TS"].unique())
    n_dates = len(train_dates)

    test_block_size = n_dates // (n_splits + 1)

    for i in range(n_splits):
        train_end = test_block_size * (i + 1)
        test_end = test_block_size * (i + 2) if i < n_splits - 1 else n_dates

        local_train_dates = train_dates[:train_end]
        local_test_dates = train_dates[train_end:test_end]

        if len(local_test_dates) == 0:
            break

        local_train_ids = X_train["TS"].isin(local_train_dates)
        local_test_ids = X_train["TS"].isin(local_test_dates)

        X_local_train = X_train.loc[local_train_ids, features_catboost]
        y_local_train = y_train.loc[local_train_ids, "target"]

        X_local_test = X_train.loc[local_test_ids, features_catboost]
        y_local_test = y_train.loc[local_test_ids, "target"]

        train_data = catboost.Pool(X_local_train, label=y_local_train.values)
        valid_data = catboost.Pool(X_local_test, label=y_local_test.values)

        model_catboost = catboost.CatBoostRegressor(**catboost_params)
        model_catboost.fit(
            train_data,
            eval_set=valid_data,
            use_best_model=True,
            verbose=False,
        )

        y_local_pred = model_catboost.predict(X_local_test)

        score = accuracy_score(
            (y_local_test > 0).astype(int),
            (y_local_pred > 0).astype(int),
        )

        models_catboost.append(model_catboost)
        scores_catboost.append(score)

        train_min = min(local_train_dates)
        train_max = max(local_train_dates)
        test_min = min(local_test_dates)
        test_max = max(local_test_dates)

        print(
            f"Fold {i+1} | "
            f"train: [{train_min} -> {train_max}] | "
            f"test: [{test_min} -> {test_max}] | "
            f"best_iter: {model_catboost.get_best_iteration()} | "
            f"Accuracy: {score * 100:.2f}%"
        )

    mean_catboost = np.mean(scores_catboost) * 100
    std_catboost = np.std(scores_catboost) * 100

    if see_folds:
        print(
            f"Accuracy: {mean_catboost:.2f}% "
            f"[{mean_catboost - std_catboost:.2f} ; {mean_catboost + std_catboost:.2f}] "
            f"(+- {std_catboost:.2f})"
        )

    return models_catboost, scores_catboost


# ----------------------LGBM HYPERPARAMETERS GRID SEARCH ----------------------


def find_hyperparameters_lgbm(X_train, y_train, features, temporal=False):
    param_grid = {
        "num_boost_round": [300, 500, 1000],
        "learning_rate": [0.01, 0.03, 0.05],
        "max_depth": [3, 4, 6],
    }

    best_score = -np.inf
    best_params = None
    results = []

    for num_boost_round in param_grid["num_boost_round"]:
        for learning_rate in param_grid["learning_rate"]:
            for max_depth in param_grid["max_depth"]:
                print(
                    f"Testing: num_boost_round={num_boost_round}, "
                    f"learning_rate={learning_rate}, max_depth={max_depth}"
                )

                if temporal:
                    _, scores_lgbm = lgbm_cv_temporal(
                        X_train,
                        y_train,
                        features,
                        n_splits=8,
                        num_boost_round=num_boost_round,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        see_folds=False,
                    )
                else:
                    X_test = pd.read_csv("data/X_test.csv", index_col="ROW_ID")
                    _, scores_lgbm = lgbm_cv(
                        X_train,
                        y_train,
                        X_test,
                        features,
                        num_boost_round=num_boost_round,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                    )

                mean_score = float(np.mean(scores_lgbm))
                std_score = float(np.std(scores_lgbm))

                results.append(
                    {
                        "num_boost_round": num_boost_round,
                        "learning_rate": learning_rate,
                        "max_depth": max_depth,
                        "mean_score": mean_score,
                        "std_score": std_score,
                    }
                )

                print(f"Mean CV Accuracy: {mean_score:.4f} " f"(std={std_score:.4f})")

                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {
                        "num_boost_round": num_boost_round,
                        "learning_rate": learning_rate,
                        "max_depth": max_depth,
                    }
                    print(
                        f"🟢 New best: {best_params} "
                        f"with CV Accuracy: {best_score:.4f}"
                    )

    print(f"✅✅✅ Best parameters: {best_params} with CV Accuracy: {best_score:.4f}")
    return best_params, results


# ----------------------CATBOOST HYPERPARAMETERS GRID SEARCH -----------------


def find_hyperparameters_catboost(X_train, y_train, features, temporal=False):

    param_grid = {
        "iterations": [300, 600, 1000],
        "learning_rate": [0.02, 0.05, 0.1],
        "depth": [4, 6, 8],
    }

    best_score = -np.inf
    best_params = None
    results = []

    for iterations in param_grid["iterations"]:
        for learning_rate in param_grid["learning_rate"]:
            for depth in param_grid["depth"]:
                print(
                    f"Testing: iterations={iterations}, "
                    f"learning_rate={learning_rate}, depth={depth}"
                )

                if temporal:
                    _, scores_catboost = catboost_cv_temporal(
                        X_train,
                        y_train,
                        features,
                        n_splits=8,
                        iterations=iterations,
                        learning_rate=learning_rate,
                        depth=depth,
                        see_folds=False,
                    )
                else:
                    X_test = pd.read_csv("data/X_test.csv", index_col="ROW_ID")
                    _, scores_catboost = catboost_cv(
                        X_train,
                        y_train,
                        X_test,
                        features,
                        iterations=iterations,
                        learning_rate=learning_rate,
                        depth=depth,
                    )

                mean_score = float(np.mean(scores_catboost))
                std_score = float(np.std(scores_catboost))

                results.append(
                    {
                        "iterations": iterations,
                        "learning_rate": learning_rate,
                        "depth": depth,
                        "mean_score": mean_score,
                        "std_score": std_score,
                    }
                )

                print(f"Mean CV Accuracy: {mean_score:.4f} " f"(std={std_score:.4f})")

                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {
                        "iterations": iterations,
                        "learning_rate": learning_rate,
                        "depth": depth,
                    }
                    print(
                        f"🟢 New best: {best_params} "
                        f"with CV Accuracy: {best_score:.4f}"
                    )

    print(f"✅✅✅ Best parameters: {best_params} with CV Accuracy: {best_score:.4f}")
    return best_params, results

# ------------------------- TRAINING & SAVE FUNCTION ----------------------

def train_lgbm(
    X_train,
    y_train,
    features,
    num_boost_round=500,
    learning_rate=1e-2,
    max_depth=3,
    saving_csv=False,
):

    train_data = lgbm.Dataset(X_train[features], label=y_train)

    lgbm_params = {
        "objective": "mse",
        "metric": "mse",
        "num_threads": 50,
        "seed": 42,
        "verbosity": -1,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
    }

    NUM_BOOST_ROUND = num_boost_round

    model_lgbm = lgbm.train(lgbm_params, train_data, num_boost_round=NUM_BOOST_ROUND)

    X_test = pd.read_csv("data/X_test.csv", index_col="ROW_ID")
    preds_lgbm = model_lgbm.predict(X_test[features])

    sample_submission = pd.read_csv("data/sample_submission.csv", index_col="ROW_ID")
    preds_lgbm = pd.DataFrame(
        preds_lgbm, index=sample_submission.index, columns=["target"]
    )

    if saving_csv:
        filename = input(
            'Choose the name of the csv file to save the predictions (e.g. "preds_lgbm.csv") and press enter: '
        )
        (preds_lgbm > 0).astype(int).to_csv(f"submissions/{filename}.csv")
        print(f"Predictions saved to submissions/{filename}.csv")


def train_catboost(
    X_train,
    y_train,
    X_test,
    features,
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    saving_csv=False,
):
    catboost_params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": iterations,
        "learning_rate": learning_rate,
        "depth": depth,
        "verbose": False,
    }
    
    train_data = catboost.Pool(X_train[features], label=y_train)
    model_catboost = catboost.CatBoostRegressor(**catboost_params)
    model_catboost.fit(train_data)

    preds_catboost = model_catboost.predict(X_test[features])

    sample_submission = pd.read_csv("data/sample_submission.csv", index_col="ROW_ID")
    preds_catboost = pd.DataFrame(
        preds_catboost, index=sample_submission.index, columns=["target"]
    )

    if saving_csv:
        filename = input(
            'Choose the name of the csv file to save the predictions (e.g. "preds_catboost.csv") and press enter: '
        )
        (preds_catboost > 0).astype(int).to_csv(f"submissions/{filename}.csv")
        print(f"Predictions saved to submissions/{filename}.csv")




# ---------------------- GROUPKFOLD-BY-TS CV (LEAKAGE-FREE) ----------------------


def _ts_group_splits(X_train, n_splits=5):
    """
    Returns (train_mask, val_mask) pairs for GroupKFold where each
    unique timestamp goes to exactly one fold.  This guarantees that
    no timestamp leaks across train/validation.
    """
    ts_values = X_train["TS"].values
    unique_ts = np.unique(ts_values)

    gkf = GroupKFold(n_splits=n_splits)
    # groups: one integer per unique_ts index, broadcast to all rows
    ts_to_idx = {ts: i for i, ts in enumerate(unique_ts)}
    row_groups = np.array([ts_to_idx[ts] for ts in ts_values])

    for train_idx, val_idx in gkf.split(X_train, groups=row_groups):
        yield train_idx, val_idx


def lgbm_cv_grouped(
    X_train,
    y_train,
    features,
    n_splits=5,
    num_boost_round=2000,
    learning_rate=0.03,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    early_stopping_rounds=200,
    see_folds=True,
):
    """
    LightGBM cross-validation using GroupKFold by timestamp.
    Uses binary cross-entropy loss for better probability calibration.
    Returns models, OOF predictions, and per-fold accuracy scores.
    """
    features_list = list(features)

    lgbm_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_threads": 50,
        "seed": 42,
        "verbosity": -1,
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "min_child_samples": min_child_samples,
    }

    y_bin = (y_train["target"].values > 0).astype(int)
    oof_preds = np.zeros(len(X_train))
    scores = []
    models = []

    for fold_i, (train_idx, val_idx) in enumerate(
        _ts_group_splits(X_train, n_splits=n_splits)
    ):
        X_tr = X_train.iloc[train_idx][features_list]
        y_tr = y_bin[train_idx]
        X_val = X_train.iloc[val_idx][features_list]
        y_val = y_bin[val_idx]

        dtrain = lgbm.Dataset(X_tr, label=y_tr)
        dval = lgbm.Dataset(X_val, label=y_val, reference=dtrain)

        callbacks = [lgbm.early_stopping(early_stopping_rounds, verbose=False),
                     lgbm.log_evaluation(-1)]

        model = lgbm.train(
            lgbm_params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dval],
            callbacks=callbacks,
        )

        pred_proba = model.predict(X_val)
        oof_preds[val_idx] = pred_proba

        score = accuracy_score(y_val, (pred_proba > 0.5).astype(int))
        scores.append(score)
        models.append(model)

        if see_folds:
            print(f"  Fold {fold_i + 1}/{n_splits} | acc={score * 100:.2f}% | "
                  f"best_iter={model.best_iteration}")

    mean_acc = np.mean(scores) * 100
    std_acc = np.std(scores) * 100
    oof_acc = accuracy_score(y_bin, (oof_preds > 0.5).astype(int)) * 100
    print(f"  LGBM GroupKFold | mean={mean_acc:.2f}% +/-{std_acc:.2f}% | "
          f"OOF={oof_acc:.2f}%")

    return models, oof_preds, scores


def catboost_cv_grouped(
    X_train,
    y_train,
    features,
    n_splits=5,
    iterations=1800,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=8.0,
    early_stopping_rounds=200,
    see_folds=True,
):
    """
    CatBoost cross-validation using GroupKFold by timestamp.
    Uses Logloss for better probability calibration.
    Returns models, OOF predictions, and per-fold accuracy scores.
    """
    features_list = list(features)

    y_bin = (y_train["target"].values > 0).astype(int)
    oof_preds = np.zeros(len(X_train))
    scores = []
    models = []

    for fold_i, (train_idx, val_idx) in enumerate(
        _ts_group_splits(X_train, n_splits=n_splits)
    ):
        X_tr = X_train.iloc[train_idx][features_list]
        y_tr = y_bin[train_idx]
        X_val = X_train.iloc[val_idx][features_list]
        y_val = y_bin[val_idx]

        train_pool = catboost.Pool(X_tr, label=y_tr)
        val_pool = catboost.Pool(X_val, label=y_val)

        params = {
            "loss_function": "Logloss",
            "eval_metric": "Accuracy",
            "iterations": iterations,
            "learning_rate": learning_rate,
            "depth": depth,
            "l2_leaf_reg": l2_leaf_reg,
            "random_seed": 42,
            "verbose": False,
        }

        model = catboost.CatBoostClassifier(**params)
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )

        pred_proba = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = pred_proba

        score = accuracy_score(y_val, (pred_proba > 0.5).astype(int))
        scores.append(score)
        models.append(model)

        if see_folds:
            print(f"  Fold {fold_i + 1}/{n_splits} | acc={score * 100:.2f}% | "
                  f"best_iter={model.get_best_iteration()}")

    mean_acc = np.mean(scores) * 100
    std_acc = np.std(scores) * 100
    oof_acc = accuracy_score(y_bin, (oof_preds > 0.5).astype(int)) * 100
    print(f"  CatBoost GroupKFold | mean={mean_acc:.2f}% +/-{std_acc:.2f}% | "
          f"OOF={oof_acc:.2f}%")

    return models, oof_preds, scores


def xgb_cv_grouped(
    X_train,
    y_train,
    features,
    n_splits=5,
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=200,
    see_folds=True,
):
    """
    XGBoost cross-validation using GroupKFold by timestamp.
    Returns models, OOF predictions, and per-fold accuracy scores.
    """
    if not _XGB_AVAILABLE:
        raise ImportError("xgboost is not installed. Run: pip install xgboost")

    features_list = list(features)
    y_bin = (y_train["target"].values > 0).astype(int)
    oof_preds = np.zeros(len(X_train))
    scores = []
    models = []

    for fold_i, (train_idx, val_idx) in enumerate(
        _ts_group_splits(X_train, n_splits=n_splits)
    ):
        X_tr = X_train.iloc[train_idx][features_list]
        y_tr = y_bin[train_idx]
        X_val = X_train.iloc[val_idx][features_list]
        y_val = y_bin[val_idx]

        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            early_stopping_rounds=early_stopping_rounds,
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        pred_proba = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = pred_proba

        score = accuracy_score(y_val, (pred_proba > 0.5).astype(int))
        scores.append(score)
        models.append(model)

        if see_folds:
            print(f"  Fold {fold_i + 1}/{n_splits} | acc={score * 100:.2f}%")

    mean_acc = np.mean(scores) * 100
    std_acc = np.std(scores) * 100
    oof_acc = accuracy_score(y_bin, (oof_preds > 0.5).astype(int)) * 100
    print(f"  XGBoost GroupKFold | mean={mean_acc:.2f}% +/-{std_acc:.2f}% | "
          f"OOF={oof_acc:.2f}%")

    return models, oof_preds, scores


# ---------------------- ENSEMBLE UTILS ----------------------


def find_optimal_threshold(oof_preds, y_train):
    """
    Grid-search the decision threshold on OOF predictions.
    Returns the threshold with highest OOF accuracy.
    """
    y_bin = (y_train["target"].values > 0).astype(int)
    best_thresh = 0.5
    best_acc = 0.0

    for t in np.arange(0.30, 0.71, 0.01):
        acc = accuracy_score(y_bin, (oof_preds >= t).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    print(f"  Optimal threshold={best_thresh:.2f} | OOF acc={best_acc * 100:.2f}%")
    return best_thresh, best_acc


def find_optimal_blend(oof_list, y_train, model_names=None):
    """
    Grid-search blend weights over OOF probability arrays.
    oof_list: list of 1D arrays of OOF probabilities, one per model.
    Returns optimal weights and the blended OOF accuracy.
    """
    y_bin = (y_train["target"].values > 0).astype(int)
    n_models = len(oof_list)
    best_acc = 0.0
    best_weights = [1.0 / n_models] * n_models

    if n_models == 1:
        return best_weights, accuracy_score(y_bin, (oof_list[0] > 0.5).astype(int))

    # For 2–4 models, exhaustive search over 0.05 increments
    from itertools import product as iproduct

    steps = np.arange(0, 1.05, 0.05)
    for combo in iproduct(steps, repeat=n_models):
        total = sum(combo)
        if abs(total) < 1e-6:
            continue
        w = np.array(combo) / total
        blend = sum(w[i] * oof_list[i] for i in range(n_models))
        acc = accuracy_score(y_bin, (blend > 0.5).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_weights = list(w)

    if model_names is None:
        model_names = [f"model_{i}" for i in range(n_models)]
    weight_str = ", ".join(
        f"{model_names[i]}={best_weights[i]:.2f}" for i in range(n_models)
    )
    print(f"  Best blend: {weight_str} | OOF acc={best_acc * 100:.2f}%")
    return best_weights, best_acc


def predict_ensemble(models_list, X_test, features, weights, use_catboost_proba=True):
    """
    Generate blended test predictions from multiple model lists.

    models_list: list of (model_list, model_type, feat_list) where feat_list
                 is the feature list for that model.  For backward compatibility
                 the 3rd element is optional; if omitted, the shared `features`
                 argument is used.
    weights: blend weights (one per model_list entry).
    Returns binary predictions (0/1) and blended probabilities.
    """
    default_features = list(features)
    blend = np.zeros(len(X_test))

    for entry, w in zip(models_list, weights):
        model_list, model_type = entry[0], entry[1]
        feat = list(entry[2]) if len(entry) > 2 else default_features
        fold_preds = []
        for model in model_list:
            if model_type == "lgbm":
                p = model.predict(X_test[feat])
            elif model_type == "catboost":
                p = model.predict_proba(X_test[feat])[:, 1]
            elif model_type == "xgb":
                p = model.predict_proba(X_test[feat])[:, 1]
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            fold_preds.append(p)
        blend += w * np.mean(fold_preds, axis=0)

    return (blend > 0.5).astype(int), blend


# ---------------------------- MAIN EXECUTION -----------------------------

if __name__ == "__main__":

    temporal =  False #(
    #     input("\n Do you want to perform temporal CV? (y/n): ").strip().lower() == "y"
    # )
    if temporal:
        print("Performing TEMPORAL CV. This may take a while...")
    else:
        print("Performing STANDARD CV. This may take a while...")

    y_train = pd.read_csv("data/y_train.csv", index_col="ROW_ID")
    X_test = pd.read_csv("data/X_test.csv", index_col="ROW_ID")

    if temporal:
        X_train = pd.read_csv("data/X_train_reconstructed.csv", index_col="ROW_ID")
    else:
        X_train = pd.read_csv("data/X_train.csv", index_col="ROW_ID")

    # add benchmark features
    X_train, X_test, features = feature_eng.FE_benchmark(X_train, X_test)
    print("Benchmark features added")

    # now add rowwise features
    X_train, X_test, features = feature_eng.add_rowwise_features(
        X_train, X_test, features
    )
    print("Rowwise features added")

    # now add cross-sectional context features
    X_train, X_test, features = feature_eng.add_cross_sectional_context_features(
    X_train, X_test, features
    )
    print("Cross-sectional context features added")

    if temporal:
        X_train, X_test, features = feature_eng.add_temporal_FE(
            X_train, X_test, features
        )
        print("Added temporal features")

    best_params_lgbm, lgbm_grid_results = find_hyperparameters_lgbm(
        X_train,
        y_train,
        features,
        temporal=temporal,
    )

    best_params_catboost, catboost_grid_results = find_hyperparameters_catboost(
        X_train,
        y_train,
        features,
        temporal=temporal,
    )
    saving_csv = (
        input("\n Do you want to save the predictions of the final models? (y/n): ")
        .strip()
        .lower()
        == "y"
    )

    if saving_csv:
        print("Training final LGBM model and saving predictions...")
        train_lgbm(
            X_train,
            y_train,
            features,
            num_boost_round=best_params_lgbm["num_boost_round"],
            learning_rate=best_params_lgbm["learning_rate"],
            max_depth=best_params_lgbm["max_depth"],
            saving_csv=True,
        )

        print("Training final CatBoost model and saving predictions...")
        train_catboost(
            X_train,
            y_train,
            features,
            iterations=best_params_catboost["iterations"],
            learning_rate=best_params_catboost["learning_rate"],
            depth=best_params_catboost["depth"],
            saving_csv=True,
        )
