import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
import os

np.random.seed(42)
num_average_time = 100  # Number of times to run each experiment to calculate the average values


# ---------------------------
# Function to create fake data (binary features, random labels)
# ---------------------------
def make_binary_data(N, M):
    X = np.random.randint(0, 2, size=(N, M))
    y = np.random.randint(0, 2, size=N)
    return X, y


# ---------------------------
# Function to benchmark training + prediction runtime
# ---------------------------
def benchmark_tree(N, M, criterion="information_gain", max_depth=None, task="classification"):
    X, y = make_binary_data(N, M)
    
    # Switch between regression and classification targets
    if task == "regression":
        y = np.random.rand(N) * 100   # continuous labels

    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(M)])
    y = pd.Series(y, name="target")
    tree = DecisionTree(criterion=criterion, max_depth=max_depth)

    # Train time
    start = time.time()
    tree.fit(X, y)
    train_time = time.time() - start

    # Predict time
    start = time.time()
    _ = tree.predict(X)
    predict_time = time.time() - start

    return train_time, predict_time


# ---------------------------
# Function to run multiple trials for averaging runtimes
# ---------------------------
def average_runtime(N, M, criterion, max_depth, task, trials=3):
    train_times, predict_times = [], []
    for _ in range(trials):
        t_train, t_pred = benchmark_tree(N, M, criterion, max_depth, task)
        train_times.append(t_train)
        predict_times.append(t_pred)
    return (np.mean(train_times), np.std(train_times)), (np.mean(predict_times), np.std(predict_times))


# ---------------------------
# Function to plot results
# ---------------------------
def plot_results(Ns, Ms, results, title, ylabel):
    for i, N in enumerate(Ns):
        plt.errorbar(Ms, [r[0] for r in results[i]], yerr=[r[1] for r in results[i]], label=f"N={N}")
    plt.xlabel("M (features)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    # Ensure results/ folder exists
    os.makedirs("results", exist_ok=True)
    # Save the plot
    plt.savefig(f"results/{title}.png")
    plt.close()


# ---------------------------
# Main Experiment
# ---------------------------
def run_experiments():
    Ns = [100, 500, 1000, 2000]  # vary number of samples
    Ms = [5, 10, 20, 50]         # vary number of features

    cases = {
        "Classification-Unpruned": {"criterion": "information_gain", "max_depth": None, "task": "classification"},
        "Classification-Pruned":   {"criterion": "information_gain", "max_depth": 5,   "task": "classification"},
        "Regression-Unpruned":     {"criterion": "mse",              "max_depth": None, "task": "regression"},
        "Regression-Pruned":       {"criterion": "mse",              "max_depth": 5,   "task": "regression"},
    }

    for case_name, params in cases.items():
        print(f"\n=== Running: {case_name} ===")
        results_train, results_predict = [], []

        for N in Ns:
            train_times, predict_times = [], []
            for M in Ms:
                (mean_t, std_t), (mean_p, std_p) = average_runtime(N, M, **params)
                train_times.append((mean_t, std_t))
                predict_times.append((mean_p, std_p))
            results_train.append(train_times)
            results_predict.append(predict_times)

        # Plot results
        plot_results(Ns, Ms, results_train, f"{case_name} - Training Time", "Training time (s)")
        plot_results(Ns, Ms, results_predict, f"{case_name} - Prediction Time", "Prediction time (s)")


# ---------------------------
# Run everything
# ---------------------------
if __name__ == "__main__":
    run_experiments()