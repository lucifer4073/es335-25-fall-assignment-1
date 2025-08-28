"""
Comparison of Custom DecisionTree vs sklearn DecisionTree
for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
from tree.base import DecisionTree
from metrics import *

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

np.random.seed(42)

# -----------------------------
# Utility function for comparison
# -----------------------------
def compare_models(X, y, task_type, criteria_list):
    """
    task_type: "classification" or "regression"
    """
    for criteria in criteria_list:
        print("\n===============================")
        print(f"Task: {task_type} | Criterion: {criteria}")
        print("===============================")

        # ---- Custom Decision Tree ----
        custom_tree = DecisionTree(criterion=criteria)
        custom_tree.fit(X, y)
        y_hat_custom = custom_tree.predict(X)

        # ---- Sklearn Decision Tree ----
        if task_type == "classification":
            sk_tree = DecisionTreeClassifier(
                criterion="entropy" if criteria == "information_gain" else "gini",
                max_depth=5,
                random_state=42,
            )
        else:  # regression
            sk_tree = DecisionTreeRegressor(
                criterion="squared_error",  # sklearn uses mse-like "squared_error"
                max_depth=5,
                random_state=42,
            )

        sk_tree.fit(X, y)
        y_hat_sklearn = pd.Series(sk_tree.predict(X), index=X.index)

        # ---- Metrics ----
        if task_type == "classification":
            acc_c = accuracy(y_hat_custom, y)
            acc_s = accuracy(y_hat_sklearn, y)
            print(f"Accuracy - Custom: {acc_c:.3f}, Sklearn: {acc_s:.3f}")
            for cls in y.unique():
                p_c = precision(y_hat_custom, y, cls)
                p_s = precision(y_hat_sklearn, y, cls)
                r_c = recall(y_hat_custom, y, cls)
                r_s = recall(y_hat_sklearn, y, cls)
                print(f"Class {cls}:")
                print(f"   Precision - Custom: {p_c:.3f}, Sklearn: {p_s:.3f}")
                print(f"   Recall    - Custom: {r_c:.3f}, Sklearn: {r_s:.3f}")

        else:  # regression
            rmse_c = rmse(y_hat_custom, y)
            rmse_s = rmse(y_hat_sklearn, y)
            mae_c = mae(y_hat_custom, y)
            mae_s = mae(y_hat_sklearn, y)
            print(f"RMSE - Custom: {rmse_c:.3f}, Sklearn: {rmse_s:.3f}")
            print(f"MAE  - Custom: {mae_c:.3f}, Sklearn: {mae_s:.3f}")


# -----------------------------
# Test case 1: Real Input, Real Output
# -----------------------------
N, P = 30, 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))
compare_models(X, y, task_type="regression", criteria_list=["information_gain", "gini_index"])

# -----------------------------
# Test case 2: Real Input, Discrete Output
# -----------------------------
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(P, size=N), dtype="category")
compare_models(X, y, task_type="classification", criteria_list=["information_gain", "gini_index"])

# -----------------------------
# Test case 3: Discrete Input, Discrete Output
# -----------------------------
X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(P)})
y = pd.Series(np.random.randint(P, size=N), dtype="category")
compare_models(X, y, task_type="classification", criteria_list=["information_gain", "gini_index"])

# -----------------------------
# Test case 4: Discrete Input, Real Output
# -----------------------------
X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(P)})
y = pd.Series(np.random.randn(N))
compare_models(X, y, task_type="regression", criteria_list=["information_gain", "gini_index"])
