import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
import os

np.random.seed(42)
num_average_time = 100

def make_binary_data(N, M, y_kind="discrete"):
    X = np.random.randint(0, 2, size=(N, M))
    if y_kind == "discrete":
        y = np.random.randint(0, 2, size=N)
    else:
        y = np.random.rand(N) * 100.0
    return X, y

def make_real_data(N, M, y_kind="discrete"):
    X = np.random.randn(N, M)
    if y_kind == "discrete":
        y = np.random.randint(0, 2, size=N)
    else:
        y = np.random.randn(N) * 100.0
    return X, y

def benchmark_tree(N, M, case, criterion="information_gain", max_depth=None):
    if case == "disc_disc":
        X, y = make_binary_data(N, M, "discrete")
        task = "classification"
    elif case == "disc_real":
        X, y = make_binary_data(N, M, "real")
        task = "regression"
    elif case == "real_disc":
        X, y = make_real_data(N, M, "discrete")
        task = "classification"
    elif case == "real_real":
        X, y = make_real_data(N, M, "real")
        task = "regression"
    else:
        raise ValueError("Unknown case")

    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(M)])
    y = pd.Series(y, name="target")
    if task == "classification":
        tree = DecisionTree(criterion=criterion if criterion else "information_gain",
                            max_depth=max_depth)
    else:
        tree = DecisionTree(criterion=criterion if criterion else "mse",
                            max_depth=max_depth)

    start = time.time()
    tree.fit(X, y)
    train_time = time.time() - start

    start = time.time()
    _ = tree.predict(X)
    predict_time = time.time() - start

    return train_time, predict_time

def average_runtime(N, M, case, criterion=None, max_depth=None, trials=3):
    train_times, predict_times = [], []
    for _ in range(trials):
        t_train, t_pred = benchmark_tree(N, M, case, criterion, max_depth)
        train_times.append(t_train)
        predict_times.append(t_pred)
    return (np.mean(train_times), np.std(train_times)), (np.mean(predict_times), np.std(predict_times))

def plot_results(Ns, Ms, results, title, ylabel):
    for i, N in enumerate(Ns):
        means = [r[0] for r in results[i]]
        stds  = [r[1] for r in results[i]]
        plt.errorbar(Ms, means, yerr=stds, marker='o', capsize=3, label=f"N={N}")
    plt.xlabel("M (features)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{title}.png", bbox_inches="tight", dpi=150)
    plt.close()

def run_all():
    base_Ns = [200, 500, 1000]
    base_Ms = [5, 10, 20, 40]
    base_trials = 5

    light_Ns = [200, 500]
    light_Ms = [5, 10, 20]
    light_trials = 3

    cases = [
        ("real_disc", "Real X, Discrete y", light_Ns, light_Ms, light_trials),
        ("real_real", "Real X, Real y", light_Ns, light_Ms, light_trials),
    ]

    summary_rows = []

    for case_key, case_name, Ns, Ms, trials in cases:
        fit_results_by_N = []
        pred_results_by_N = []
        for N in Ns:
            fit_results_M = []
            pred_results_M = []
            for M in Ms:
                (fit_mean, fit_std), (pred_mean, pred_std) = average_runtime(
                    N, M, case=case_key, criterion=None, max_depth=None, trials=trials
                )
                fit_results_M.append((fit_mean, fit_std))
                pred_results_M.append((pred_mean, pred_std))
                summary_rows.append({
                    "case": case_key,
                    "N": N,
                    "M": M,
                    "fit_mean": fit_mean,
                    "fit_std": fit_std,
                    "pred_mean": pred_mean,
                    "pred_std": pred_std,
                    "trials": trials,
                })
            fit_results_by_N.append(fit_results_M)
            pred_results_by_N.append(pred_results_M)

        plot_results(Ns, Ms, fit_results_by_N,
                     title=f"{case_name} - Train Time vs M",
                     ylabel="Train time (s)")
        plot_results(Ns, Ms, pred_results_by_N,
                     title=f"{case_name} - Predict Time vs M",
                     ylabel="Predict time (s)")

    df = pd.DataFrame(summary_rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/runtime_summary.csv", index=False)
    print("Saved results to results/ folder with reduced grids for real-feature cases")

if __name__ == "__main__":
    run_all()
