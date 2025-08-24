"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd
from tree.utils import *
import matplotlib.pyplot as plt

np.random.seed(42)

@dataclass
class Node:
    feature: str = None
    threshold: float = None
    children: dict = None
    left = None
    right = None
    prediction = None


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]
    max_depth: int

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth if max_depth is not None else 5
        self.root = None
        self.is_regression = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.is_regression = check_ifreal(y)
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        node = Node()

        # Stopping conditions
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(X) == 0:
            node.prediction = np.mean(y) if self.is_regression else y.mode()[0]
            return node

        feature, threshold = opt_split_attribute(X, y, 
                                                 "mse" if self.is_regression else self.criterion,
                                                 X.columns)
        if feature is None:
            node.prediction = np.mean(y) if self.is_regression else y.mode()[0]
            return node

        node.feature = feature
        node.threshold = threshold

        if threshold is not None:  # numeric split
            (X_left, y_left), (X_right, y_right) = split_data(X, y, feature, threshold)
            node.left = self._build_tree(X_left, y_left, depth+1)
            node.right = self._build_tree(X_right, y_right, depth+1)
        else:  # categorical split
            node.children = {}
            splits = split_data(X, y, feature)
            for v, (X_sub, y_sub) in splits.items():
                node.children[v] = self._build_tree(X_sub, y_sub, depth+1)

        return node

    def predict_one(self, x, node):
        if node.prediction is not None:
            return node.prediction

        if node.threshold is not None:  # numeric
            if x[node.feature] <= node.threshold:
                return self.predict_one(x, node.left)
            else:
                return self.predict_one(x, node.right)
        else:  # categorical
            val = x[node.feature]
            if val in node.children:
                return self.predict_one(x, node.children[val])
            else:
                return list(node.children.values())[0].prediction  # fallback

    def predict(self, X: pd.DataFrame) -> pd.Series:
        preds = [self.predict_one(x, self.root) for _, x in X.iterrows()]
        return pd.Series(preds, index=X.index)

    def _print_tree(self, node, indent=""):
        if node.prediction is not None:
            print(indent + f"Predict: {node.prediction}")
            return

        if node.threshold is not None:
            print(indent + f"?({node.feature} <= {node.threshold})")
            print(indent + "-> True:")
            self._print_tree(node.left, indent + "   ")
            print(indent + "-> False:")
            self._print_tree(node.right, indent + "   ")
        else:
            print(indent + f"?({node.feature})")
            for v, child in node.children.items():
                print(indent + f"-> {v}:")
                self._print_tree(child, indent + "   ")

    def plot(self) -> None:
        self._print_tree(self.root)
