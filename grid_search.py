import pandas as pd
import numpy as np
import seaborn as sns

sns.set()
import matplotlib.pyplot as plt

from sklearn import linear_model
import lightgbm as lgbm
import catboost

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import feature_eng

# ---------------------------- LGBM CV FUNCTIONS -----------------------------


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


def find_hyperparameters_lgbm(X_train, y_train, features):
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


def find_hyperparameters_catboost(X_train, y_train, features):
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


# ---------------------------- MAIN EXECUTION -----------------------------

if __name__ == "__main__":

    X_test = pd.read_csv("data/X_test.csv", index_col="ROW_ID")
    y_train = pd.read_csv("data/y_train.csv", index_col="ROW_ID")
    sample_submission = pd.read_csv("data/sample_submission.csv", index_col="ROW_ID")

    X_train_rec = pd.read_csv("data/X_train_reconstructed.csv", index_col="ROW_ID")

    # only benchmark features
    print("Benchmark features only and temporal CV")
    X_train_rec, X_test, features = feature_eng.FE_benchmark(X_train_rec, X_test)

    # now add rowwise features
    print("Add rowwise features and temporal CV")
    X_train_rec, X_test, features = feature_eng.add_rowwise_features(
        X_train_rec, X_test, features
    )

    # add temporal features
    print("Add temporal features and temporal CV")
    X_train_rec, X_test, features = feature_eng.add_temporal_FE(
        X_train_rec, X_test, features
    )

    best_params_lgbm, lgbm_grid_results = find_hyperparameters_lgbm(
        X_train_rec,
        y_train,
        features,
    )

    best_params_catboost, catboost_grid_results = find_hyperparameters_catboost(
        X_train_rec,
        y_train,
        features,
    )
