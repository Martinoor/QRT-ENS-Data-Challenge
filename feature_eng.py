# ------------------ Benchmark features ------------------

def FE_benchmark(X_train, X_test):
    RET_features = [f"RET_{i}" for i in range(1, 21)]
    SIGNED_VOLUME_features = [f"SIGNED_VOLUME_{i}" for i in range(1, 21)]
    TURNOVER_features = ["MEDIAN_DAILY_TURNOVER"]

    
    for i in [3, 5, 10, 15, 20]:
        X_train[f"AVERAGE_PERF_{i}"] = X_train[RET_features[:i]].mean(1)
        X_train[f"ALLOCATIONS_AVERAGE_PERF_{i}"] = X_train.groupby("TS")[
            f"AVERAGE_PERF_{i}"
        ].transform("mean")

        X_test[f"AVERAGE_PERF_{i}"] = X_test[RET_features[:i]].mean(1)
        X_test[f"ALLOCATIONS_AVERAGE_PERF_{i}"] = X_test.groupby("TS")[
            f"AVERAGE_PERF_{i}"
        ].transform("mean")

    for i in [20]:
        X_train[f"STD_PERF_{i}"] = X_train[RET_features[:i]].std(1)
        X_train[f"ALLOCATIONS_STD_PERF_{i}"] = X_train.groupby("TS")[
            f"STD_PERF_{i}"
        ].transform("mean")

        X_test[f"STD_PERF_{i}"] = X_test[RET_features[:i]].std(1)
        X_test[f"ALLOCATIONS_STD_PERF_{i}"] = X_test.groupby("TS")[
            f"STD_PERF_{i}"
        ].transform("mean")

    features = RET_features + SIGNED_VOLUME_features + TURNOVER_features
    features = features + [f"AVERAGE_PERF_{i}" for i in [3, 5, 10, 15, 20]]
    features = features + [f"ALLOCATIONS_AVERAGE_PERF_{i}" for i in [3, 5, 10, 15, 20]]
    features = features + [f"STD_PERF_{i}" for i in [20]]
    features = features + [f"ALLOCATIONS_STD_PERF_{i}" for i in [20]]

    return X_train, X_test, features


# ------------------ Row-wise features ------------------

def add_rowwise_features(X_train, X_test, features):
    import pandas as pd
    import numpy as np
    X_train = X_train.copy()
    X_test = X_test.copy()
    features = list(features)

    ret_cols = [f"RET_{i}" for i in range(1, 21) if f"RET_{i}" in X_train.columns and f"RET_{i}" in X_test.columns]
    vol_cols = [
        f"SIGNED_VOLUME_{i}"
        for i in range(1, 21)
        if f"SIGNED_VOLUME_{i}" in X_train.columns and f"SIGNED_VOLUME_{i}" in X_test.columns
    ]

    def _safe_divide(numerator, denominator):
        numerator = np.asarray(numerator, dtype=float)
        denominator = np.asarray(denominator, dtype=float)
        out = np.zeros_like(numerator, dtype=float)
        np.divide(numerator, denominator, out=out, where=denominator != 0)
        return out

    def _add_features(df):
        df = df.copy()
        new_features = []

        if ret_cols:
            if "ret_min_20" not in df.columns:
                df["ret_min_20"] = df[ret_cols].min(axis=1)
                new_features.append("ret_min_20")
            if "ret_max_20" not in df.columns:
                df["ret_max_20"] = df[ret_cols].max(axis=1)
                new_features.append("ret_max_20")
            if "ret_pos_count_20" not in df.columns:
                df["ret_pos_count_20"] = (df[ret_cols] > 0).sum(axis=1)
                new_features.append("ret_pos_count_20")

            abs_ret = df[ret_cols].abs()
            if "ret_abs_mean_20" not in df.columns:
                df["ret_abs_mean_20"] = abs_ret.mean(axis=1)
                new_features.append("ret_abs_mean_20")
            if "ret_abs_std_20" not in df.columns:
                df["ret_abs_std_20"] = abs_ret.std(axis=1)
                new_features.append("ret_abs_std_20")
            if "ret_abs_max_20" not in df.columns:
                df["ret_abs_max_20"] = abs_ret.max(axis=1)
                new_features.append("ret_abs_max_20")

            for w in [3, 5, 10]:
                cols = [f"RET_{i}" for i in range(1, w + 1) if f"RET_{i}" in df.columns]
                col_name = f"ret_mean_{w}"
                if cols and col_name not in df.columns:
                    df[col_name] = df[cols].mean(axis=1)
                    new_features.append(col_name)

            if "RET_1" in df.columns and "ret_mean_5" in df.columns and "ret_mom_1_minus_5" not in df.columns:
                df["ret_mom_1_minus_5"] = df["RET_1"] - df["ret_mean_5"]
                new_features.append("ret_mom_1_minus_5")
            if "ret_mean_5" in df.columns and "AVERAGE_PERF_20" in df.columns and "ret_mom_5_minus_20" not in df.columns:
                df["ret_mom_5_minus_20"] = df["ret_mean_5"] - df["AVERAGE_PERF_20"]
                new_features.append("ret_mom_5_minus_20")

        if vol_cols:
            if "vol_mean_20" not in df.columns:
                df["vol_mean_20"] = df[vol_cols].mean(axis=1)
                new_features.append("vol_mean_20")
            if "vol_std_20" not in df.columns:
                df["vol_std_20"] = df[vol_cols].std(axis=1)
                new_features.append("vol_std_20")
            if "vol_min_20" not in df.columns:
                df["vol_min_20"] = df[vol_cols].min(axis=1)
                new_features.append("vol_min_20")
            if "vol_max_20" not in df.columns:
                df["vol_max_20"] = df[vol_cols].max(axis=1)
                new_features.append("vol_max_20")
            if "vol_pos_count_20" not in df.columns:
                df["vol_pos_count_20"] = (df[vol_cols] > 0).sum(axis=1)
                new_features.append("vol_pos_count_20")

            for w in [3, 5, 10]:
                cols = [
                    f"SIGNED_VOLUME_{i}"
                    for i in range(1, w + 1)
                    if f"SIGNED_VOLUME_{i}" in df.columns
                ]
                col_name = f"vol_mean_{w}"
                if cols and col_name not in df.columns:
                    df[col_name] = df[cols].mean(axis=1)
                    new_features.append(col_name)

            if "SIGNED_VOLUME_1" in df.columns and "vol_mean_5" in df.columns and "vol_trend_1_minus_5" not in df.columns:
                df["vol_trend_1_minus_5"] = df["SIGNED_VOLUME_1"] - df["vol_mean_5"]
                new_features.append("vol_trend_1_minus_5")
            if "vol_mean_5" in df.columns and "vol_mean_20" in df.columns and "vol_trend_5_minus_20" not in df.columns:
                df["vol_trend_5_minus_20"] = df["vol_mean_5"] - df["vol_mean_20"]
                new_features.append("vol_trend_5_minus_20")

        if "MEDIAN_DAILY_TURNOVER" in df.columns:
            if "ret_mean_5" in df.columns and "ret5_x_turnover" not in df.columns:
                df["ret5_x_turnover"] = df["ret_mean_5"] * df["MEDIAN_DAILY_TURNOVER"]
                new_features.append("ret5_x_turnover")
            if "vol_mean_5" in df.columns and "vol5_x_turnover" not in df.columns:
                df["vol5_x_turnover"] = df["vol_mean_5"] * df["MEDIAN_DAILY_TURNOVER"]
                new_features.append("vol5_x_turnover")
            if "ret_abs_mean_20" in df.columns and "absret20_to_turnover" not in df.columns:
                df["absret20_to_turnover"] = _safe_divide(
                    df["ret_abs_mean_20"].values,
                    df["MEDIAN_DAILY_TURNOVER"].values,
                )
                new_features.append("absret20_to_turnover")

        return df, new_features

    X_train, new_features_train = _add_features(X_train)
    X_test, new_features_test = _add_features(X_test)

    new_features = [col for col in new_features_train if col in new_features_test]
    features = features + [col for col in new_features if col not in features]

    return X_train, X_test, features

# ------------------ Temporal features ------------------

def add_temporal_FE(X_train, X_test, features):
    import numpy as np

    X_train = X_train.copy()
    X_test = X_test.copy()
    features = list(features)

    ret_cols = [
        f"RET_{i}"
        for i in range(1, 21)
        if f"RET_{i}" in X_train.columns and f"RET_{i}" in X_test.columns
    ]

    if not ret_cols:
        return X_train, X_test, features

    def _safe_add_feature(df, col_name, values, new_features):
        if col_name not in df.columns:
            df[col_name] = values
            new_features.append(col_name)

    def _compute_streaks(ret_matrix):
        n_rows, n_cols = ret_matrix.shape
        pos_streak = np.zeros(n_rows, dtype=int)
        neg_streak = np.zeros(n_rows, dtype=int)

        for row_idx in range(n_rows):
            current_pos = 0
            current_neg = 0
            best_pos = 0
            best_neg = 0

            for value in ret_matrix[row_idx]:
                if value > 0:
                    current_pos += 1
                    current_neg = 0
                elif value < 0:
                    current_neg += 1
                    current_pos = 0
                else:
                    current_pos = 0
                    current_neg = 0

                if current_pos > best_pos:
                    best_pos = current_pos
                if current_neg > best_neg:
                    best_neg = current_neg

            pos_streak[row_idx] = best_pos
            neg_streak[row_idx] = best_neg

        return pos_streak, neg_streak

    def _compute_sign_changes(ret_matrix):
        signs = np.sign(ret_matrix)
        non_zero = signs != 0
        sign_changes = np.zeros(ret_matrix.shape[0], dtype=int)

        for row_idx in range(ret_matrix.shape[0]):
            row_signs = signs[row_idx][non_zero[row_idx]]
            if row_signs.size >= 2:
                sign_changes[row_idx] = np.sum(row_signs[1:] != row_signs[:-1])

        return sign_changes

    def _add_temporal_features(df):
        df = df.copy()
        new_features = []

        ret_matrix = df[ret_cols].to_numpy(dtype=float)

        # Assumption used throughout the repository:
        # RET_1 is the most recent lag and RET_20 the oldest.
        lag_index = np.arange(1, len(ret_cols) + 1, dtype=float)
        recency_weights = 1.0 / lag_index
        recency_weights = recency_weights / recency_weights.sum()

        weighted_mean = ret_matrix @ recency_weights
        _safe_add_feature(
            df,
            "ret_weighted_mean_20",
            weighted_mean,
            new_features,
        )

        x = np.arange(len(ret_cols), dtype=float)
        x_centered = x - x.mean()
        denominator = np.sum(x_centered ** 2)
        slope = ((ret_matrix - ret_matrix.mean(axis=1, keepdims=True)) @ x_centered) / denominator
        _safe_add_feature(
            df,
            "ret_linear_slope_20",
            slope,
            new_features,
        )

        recent_5 = df[[f"RET_{i}" for i in range(1, 6) if f"RET_{i}" in df.columns]].mean(axis=1)
        old_5 = df[[f"RET_{i}" for i in range(16, 21) if f"RET_{i}" in df.columns]].mean(axis=1)
        _safe_add_feature(
            df,
            "ret_recent5_minus_old5",
            recent_5 - old_5,
            new_features,
        )

        recent_10 = df[[f"RET_{i}" for i in range(1, 11) if f"RET_{i}" in df.columns]].mean(axis=1)
        old_10 = df[[f"RET_{i}" for i in range(11, 21) if f"RET_{i}" in df.columns]].mean(axis=1)
        _safe_add_feature(
            df,
            "ret_recent10_minus_old10",
            recent_10 - old_10,
            new_features,
        )

        pos_streak, neg_streak = _compute_streaks(ret_matrix)
        _safe_add_feature(df, "ret_longest_pos_streak_20", pos_streak, new_features)
        _safe_add_feature(df, "ret_longest_neg_streak_20", neg_streak, new_features)

        sign_changes = _compute_sign_changes(ret_matrix)
        _safe_add_feature(df, "ret_sign_change_count_20", sign_changes, new_features)

        return df, new_features

    X_train, new_features_train = _add_temporal_features(X_train)
    X_test, new_features_test = _add_temporal_features(X_test)

    new_features = [col for col in new_features_train if col in new_features_test]
    features = features + [col for col in new_features if col not in features]

    return X_train, X_test, features


# ------------------ Cross-sectional context features ------------------


def add_cross_sectional_context_features(
    X_train,
    X_test,
    features,
    candidate_cols=None,
    add_group_context=True,
):
    import numpy as np
    import pandas as pd

    X_train = X_train.copy()
    X_test = X_test.copy()
    features = list(features)

    if "TS" not in X_train.columns or "TS" not in X_test.columns:
        raise ValueError("Both X_train and X_test must contain a 'TS' column.")

    # Columns on which cross-sectional context is most likely to help.
    # Keep this focused: enough signal, but not a huge feature explosion.
    if candidate_cols is None:
        candidate_cols = [
            "AVERAGE_PERF_3",
            "AVERAGE_PERF_5",
            "AVERAGE_PERF_10",
            "AVERAGE_PERF_15",
            "AVERAGE_PERF_20",
            "STD_PERF_20",
            "ret_mean_3",
            "ret_mean_5",
            "ret_mean_10",
            "ret_abs_mean_20",
            "ret_abs_std_20",
            "ret_mom_1_minus_5",
            "ret_mom_5_minus_20",
            "vol_mean_5",
            "vol_mean_10",
            "vol_mean_20",
            "vol_std_20",
            "vol_trend_1_minus_5",
            "vol_trend_5_minus_20",
            "MEDIAN_DAILY_TURNOVER",
            "ret5_x_turnover",
            "absret20_to_turnover",
        ]

    usable_cols = [
        col
        for col in candidate_cols
        if col in X_train.columns and col in X_test.columns
    ]

    if not usable_cols:
        return X_train, X_test, features

    def _safe_zscore(x, group_mean, group_std):
        denom = group_std.replace(0.0, np.nan)
        z = (x - group_mean) / denom
        return z.fillna(0.0)

    def _add_context(df):
        df = df.copy()
        new_features = []

        # Timestamp-level structural features
        ts_size = df.groupby("TS")["TS"].transform("size")
        if "ts_n_allocations" not in df.columns:
            df["ts_n_allocations"] = ts_size.astype(float)
            new_features.append("ts_n_allocations")

        if add_group_context and "GROUP" in df.columns:
            ts_group_size = df.groupby(["TS", "GROUP"])["TS"].transform("size")
            if "ts_group_size" not in df.columns:
                df["ts_group_size"] = ts_group_size.astype(float)
                new_features.append("ts_group_size")

            if "ts_group_share" not in df.columns:
                df["ts_group_share"] = (ts_group_size / ts_size).astype(float)
                new_features.append("ts_group_share")

        for col in usable_cols:
            # ---------- relative to the whole timestamp ----------
            ts_mean = df.groupby("TS")[col].transform("mean")
            ts_std = df.groupby("TS")[col].transform("std").fillna(0.0)
            ts_rank = df.groupby("TS")[col].rank(method="average", pct=True)

            diff_name = f"{col}_ts_diff"
            z_name = f"{col}_ts_z"
            rank_name = f"{col}_ts_rank"
            disp_name = f"{col}_ts_dispersion"

            if diff_name not in df.columns:
                df[diff_name] = df[col] - ts_mean
                new_features.append(diff_name)

            if z_name not in df.columns:
                df[z_name] = _safe_zscore(df[col], ts_mean, ts_std)
                new_features.append(z_name)

            if rank_name not in df.columns:
                df[rank_name] = ts_rank.astype(float)
                new_features.append(rank_name)

            if disp_name not in df.columns:
                df[disp_name] = ts_std.astype(float)
                new_features.append(disp_name)

            # ---------- relative to the row's group inside the same timestamp ----------
            if add_group_context and "GROUP" in df.columns:
                tsg_mean = df.groupby(["TS", "GROUP"])[col].transform("mean")
                tsg_std = (
                    df.groupby(["TS", "GROUP"])[col].transform("std").fillna(0.0)
                )

                group_diff_name = f"{col}_tsg_diff"
                group_z_name = f"{col}_tsg_z"
                group_gap_name = f"{col}_group_mean_minus_ts_mean"

                if group_diff_name not in df.columns:
                    df[group_diff_name] = df[col] - tsg_mean
                    new_features.append(group_diff_name)

                if group_z_name not in df.columns:
                    df[group_z_name] = _safe_zscore(df[col], tsg_mean, tsg_std)
                    new_features.append(group_z_name)

                if group_gap_name not in df.columns:
                    df[group_gap_name] = tsg_mean - ts_mean
                    new_features.append(group_gap_name)

        return df, new_features

    X_train, train_new_features = _add_context(X_train)
    X_test, test_new_features = _add_context(X_test)

    final_new_features = [
        col for col in train_new_features
        if col in X_test.columns and col in X_train.columns
    ]

    features = features + [col for col in final_new_features if col not in features]

    return X_train, X_test, features