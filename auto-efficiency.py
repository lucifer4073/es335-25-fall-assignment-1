import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
np.random.seed(22)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

# Clean the above data by removing redundant columns and rows with junk values
# Compare the performance of your model with the decision tree module from scikit learn
# 2) Clean dataset
# Replace '?' with NaN in horsepower, drop NaNs
data = data.replace('?', np.nan).dropna()
# Convert horsepower to float
data['horsepower'] = data['horsepower'].astype(float)

# 3) Split into features (X) and target (y)
y = data['mpg']
X = data.drop(['mpg', 'car name'], axis=1)   # drop target + non-numeric

print("Features:\n", X.shape)
print("Target:\n", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=22, shuffle= True,
)
# 4) Train custom Decision Tree
tree_custom = DecisionTree(criterion="mse", max_depth=5)
tree_custom.fit(X_train, y_train)


y_hat_train_custom = tree_custom.predict(X_train)
y_hat_test_custom = tree_custom.predict(X_test)

rmse_train_c = rmse(y_hat_train_custom, y_train)
mae_train_c = mae(y_hat_train_custom, y_train)
r2_train_c = r2_score(y_train, y_hat_train_custom)

rmse_test_c = rmse(y_hat_test_custom, y_test)
mae_test_c = mae(y_hat_test_custom, y_test)
r2_test_c = r2_score(y_test, y_hat_test_custom)

# 5) Train sklearn Decision Tree
sk_tree = DecisionTreeRegressor(criterion="squared_error", max_depth=5, random_state=22)
sk_tree.fit(X_train, y_train)

y_hat_train_sk = sk_tree.predict(X_train)
y_hat_test_sk = sk_tree.predict(X_test)

rmse_train_s = rmse(y_hat_train_sk, y_train)
mae_train_s = mae(y_hat_train_sk, y_train)
r2_train_s = r2_score(y_train, y_hat_train_sk)

rmse_test_s = rmse(y_hat_test_sk, y_test)
mae_test_s = mae(y_hat_test_sk, y_test)
r2_test_s = r2_score(y_test, y_hat_test_sk)

# 6) Report comparison
print("\n=== Auto MPG Regression: Train vs Test ===")
print("Custom DecisionTree:")
print(f"  Train → RMSE: {rmse_train_c:.3f}, MAE: {mae_train_c:.3f}, R²: {r2_train_c:.3f}")
print(f"  Test  → RMSE: {rmse_test_c:.3f}, MAE: {mae_test_c:.3f}, R²: {r2_test_c:.3f}")

print("\nsklearn DecisionTreeRegressor:")
print(f"  Train → RMSE: {rmse_train_s:.3f}, MAE: {mae_train_s:.3f}, R²: {r2_train_s:.3f}")
print(f"  Test  → RMSE: {rmse_test_s:.3f}, MAE: {mae_test_s:.3f}, R²: {r2_test_s:.3f}")
