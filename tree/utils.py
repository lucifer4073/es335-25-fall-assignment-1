import pandas as pd
import numpy as np

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    return pd.get_dummies(X, drop_first=False)


def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    return pd.api.types.is_float_dtype(y) or pd.api.types.is_numeric_dtype(y) and y.nunique() > 15

def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    values, counts = np.unique(Y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-9))


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    values, counts = np.unique(Y, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs**2)


def mse(Y: pd.Series) -> float:
    """
    Mean Squared Error for regression splitting
    """
    if len(Y) == 0:
        return 0
    mean_val = np.mean(Y)
    return np.mean((Y - mean_val) ** 2)


def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """
    if criterion in ["information_gain", "gini_index"]:  # classification
        if criterion == "information_gain":
            parent_impurity = entropy(Y)
        else:
            parent_impurity = gini_index(Y)

        values, counts = np.unique(attr, return_counts=True)
        weighted_impurity = 0
        for v, c in zip(values, counts):
            subset = Y[attr == v]
            if criterion == "information_gain":
                weighted_impurity += (c / len(Y)) * entropy(subset)
            else:
                weighted_impurity += (c / len(Y)) * gini_index(subset)

        return parent_impurity - weighted_impurity

    elif criterion == "mse":
        parent_error = mse(Y)
        values, counts = np.unique(attr, return_counts=True)
        weighted_error = 0
        for v, c in zip(values, counts):
            subset = Y[attr == v]
            weighted_error += (c / len(Y)) * mse(subset)

        return parent_error - weighted_error


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series, min_samples_leaf: int):
    """
    Function to find the optimal attribute to split about.
    """
    best_gain = -1e9
    best_feature = None
    best_threshold = None

    for feature in features:
        X_col = X[feature]
        if check_ifreal(X_col):
            thresholds = np.unique(X_col)
            for t in thresholds:
                left_mask = X_col <= t
                right_mask = X_col > t
                if left_mask.sum() < min_samples_leaf or right_mask.sum() < min_samples_leaf:
                    continue
                attr = pd.Series(["L" if v <= t else "R" for v in X_col], index=X_col.index)
                gain = information_gain(y, attr, criterion if not check_ifreal(y) else "mse")
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = t
        else:
            value_counts = X_col.value_counts()
            if (value_counts < min_samples_leaf).any():
                continue
            gain = information_gain(y, X_col, criterion)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = None

    return best_feature, best_threshold

def split_data(X: pd.DataFrame, y: pd.Series, attribute, threshold=None):
    """
    Split the data based on a particular value of a particular attribute.
    """
    if threshold is not None:  # numeric
        left_mask = X[attribute] <= threshold
        right_mask = X[attribute] > threshold
        return (X[left_mask], y[left_mask]), (X[right_mask], y[right_mask])
    else:
        splits = {}
        for v in X[attribute].unique():
            mask = X[attribute] == v
            splits[v] = (X[mask], y[mask])
        return splits
