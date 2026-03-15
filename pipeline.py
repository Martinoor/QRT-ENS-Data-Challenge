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
