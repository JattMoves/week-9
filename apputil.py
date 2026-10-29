import pandas as pd
import numpy as np

class GroupEstimate:
    def __init__(self, estimate="mean"):
        if estimate not in ("mean", "median"):
            raise ValueError("estimate must be 'mean' or 'median'")

        self.estimate = estimate
        self.feature_names = None
        self.group_estimates = None

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")

        y_values = list(y)
        if pd.Series(y_values).isna().any():
            raise ValueError("y must not contain missing values")

        self.feature_names = list(X.columns)

        data = X.copy()
        data["_estimate_target"] = y_values

        grouped = data.groupby(
            self.feature_names,
            observed=True,
            dropna=False
        )["_estimate_target"].agg(self.estimate)

        self.group_estimates = {}
        for key, value in grouped.items():
            key_tuple = key if isinstance(key, tuple) else (key,)
            self.group_estimates[key_tuple] = value

        return self

    def predict(self, X_):
        if self.group_estimates is None:
            raise RuntimeError("Call fit before predict")

        if isinstance(X_, pd.DataFrame):
            observations = X_.loc[:, self.feature_names]
        else:
            observations = pd.DataFrame(X_, columns=self.feature_names)

        predictions = []
        missing_count = 0

        for row in observations.itertuples(index=False, name=None):
            prediction = self.group_estimates.get(tuple(row), float("nan"))
            if pd.isna(prediction):
                missing_count += 1
            predictions.append(prediction)

        if missing_count:
            print(f"Missing groups: {missing_count}")

        return np.array(predictions)