import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)

# Write the code for Q2 a) and b) below. Show your results.

# -----------------------------
# Part (a): 70/30 split
# -----------------------------

X = pd.DataFrame(X, columns=["f1", "f2"])
y = pd.Series(y, dtype="category")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

tree = DecisionTree(criterion="information_gain", max_depth=5)
tree.fit(X_train, y_train)
y_hat = tree.predict(X_test)

print("=== Part (a): Train/Test Split ===")
print("Accuracy:", accuracy(y_hat, y_test))
for cls in y.unique():
    print(f"Class {cls}: Precision={precision(y_hat, y_test, cls)}, Recall={recall(y_hat, y_test, cls)}")


# -----------------------------
# Part (b): Nested Cross-Validation
# -----------------------------
print("\n=== Part (b): Nested Cross-Validation ===")
outer_kf = KFold(n_splits=5, shuffle=True, random_state=42)

outer_scores = []
chosen_depths = []

for outer_train_idx, outer_test_idx in outer_kf.split(X):
    X_outer_train, X_outer_test = X.iloc[outer_train_idx], X.iloc[outer_test_idx]
    y_outer_train, y_outer_test = y.iloc[outer_train_idx], y.iloc[outer_test_idx]

    # Inner CV for hyperparameter tuning
    inner_kf = KFold(n_splits=5, shuffle=True, random_state=42)
    depth_scores = {}

    for depth in range(1, 8):  # try depths 1..7
        inner_scores = []
        for inner_train_idx, inner_val_idx in inner_kf.split(X_outer_train):
            X_inner_train, X_val = X_outer_train.iloc[inner_train_idx], X_outer_train.iloc[inner_val_idx]
            y_inner_train, y_val = y_outer_train.iloc[inner_train_idx], y_outer_train.iloc[inner_val_idx]

            tree_inner = DecisionTree(criterion="information_gain", max_depth=depth)
            tree_inner.fit(X_inner_train, y_inner_train)
            y_val_hat = tree_inner.predict(X_val)
            inner_scores.append(accuracy(y_val_hat, y_val))

        depth_scores[depth] = np.mean(inner_scores)

    # pick best depth
    best_depth = max(depth_scores, key=depth_scores.get)
    chosen_depths.append(best_depth)

    # retrain with best depth
    best_tree = DecisionTree(criterion="information_gain", max_depth=best_depth)
    best_tree.fit(X_outer_train, y_outer_train)
    y_outer_hat = best_tree.predict(X_outer_test)

    outer_scores.append(float(accuracy(y_outer_hat, y_outer_test)))

print("Outer fold accuracies:", outer_scores)
print("Mean outer accuracy:", np.mean(outer_scores))
print("Chosen depths per fold:", chosen_depths)
print("Best average depth:", max(set(chosen_depths), key=chosen_depths.count))

