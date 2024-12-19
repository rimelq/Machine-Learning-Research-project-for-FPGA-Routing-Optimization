import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
)
from sklearn.utils.class_weight import compute_sample_weight

def train_models(input_dir="datasets", output_dir="weighted_results", target_percentages=[0.5, 0.8, 1.0], lambdas=[0.0001, 0.001, 0.01, 0.1, 1.0]):
    """
    Load datasets, train models, and save data for plotting errors and predictions.

    Parameters:
    ----------- 
    input_dir : str
        Path to the directory containing the preprocessed datasets.
    output_dir : str
        Path to the directory where results will be saved.
    target_percentages : list
        List of routing percentages for which models will be trained.
    lambdas : list
        List of alpha values (regularization strength) to iterate over for Ridge Regression.
    """
    os.makedirs(output_dir, exist_ok=True)

    benchmark_ids = [f"{i:02d}" for i in range(1, 13)]
    print(f"Benchmarks: {benchmark_ids}")

    for target_percentage in target_percentages:
        target_label = f"target_{int(target_percentage * 100)}"
        print(f"\n=== Processing target percentage: {target_percentage} ({target_label}) ===")

        X_full = []
        y_full = []

        for bench_id in benchmark_ids:
            X_path = os.path.join(input_dir, f"X_bench_{bench_id}.csv")
            y_path = os.path.join(input_dir, f"y_bench_{bench_id}_{target_label}.csv")

            if os.path.exists(X_path) and os.path.exists(y_path):
                X_bench = pd.read_csv(X_path)
                y_bench = pd.read_csv(y_path)
                X_full.append(X_bench)
                y_full.append(y_bench)

        if len(X_full) == 0 or len(y_full) == 0:
            print(f"No data found for {target_label}. Skipping...")
            continue

        X_full = pd.concat(X_full, ignore_index=True)
        y_full = pd.concat(y_full, ignore_index=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, test_size=0.3, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model_results = []

        # Ridge Regression with multiple lambdas
        for alpha in lambdas:
            print(f"Training Ridge Regression with alpha={alpha}...")
            ridge_mses, ridge_r2s = [], []

            for i in range(y_train.shape[1]):  # Loop over each target
                sample_weights = compute_sample_weight("balanced", y_train.iloc[:, i])
                ridge = Ridge(alpha=alpha, max_iter=10000)
                ridge.fit(X_train_scaled, y_train.iloc[:, i], sample_weight=sample_weights)
                preds = ridge.predict(X_test_scaled)

                ridge_mses.append(mean_squared_error(y_test.iloc[:, i], preds))
                ridge_r2s.append(r2_score(y_test.iloc[:, i], preds))

            # Save results for this alpha
            result = {
                "Metric": ["MSE", "R²"],
                **{f"Target_{i}": [ridge_mses[i], ridge_r2s[i]] for i in range(len(ridge_mses))}
            }
            output_file = os.path.join(output_dir, f"Ridge_alpha_{alpha}_{target_label}_results.csv")
            pd.DataFrame(result).to_csv(output_file, index=False)
            print(f"Results saved for Ridge with alpha={alpha} at {target_label}: {output_file}")

        # Random Forest
        print("Training Random Forest...")
        rf_mses, rf_r2s = [], []
        for i in range(y_train.shape[1]):  # Loop over each target
            rf = RandomForestRegressor(random_state=42)
            rf.fit(X_train, y_train.iloc[:, i])  # Random Forest does not support sample_weight directly
            preds = rf.predict(X_test)

            rf_mses.append(mean_squared_error(y_test.iloc[:, i], preds))
            rf_r2s.append(r2_score(y_test.iloc[:, i], preds))

        model_results.append({
            "Model": "RandomForest",
            "MSE": rf_mses,
            "R²": rf_r2s
        })

        # Linear Regression with sample_weight
        print("Training Linear Regression...")
        lin_mses, lin_r2s = [], []
        for i in range(y_train.shape[1]):  # Loop over each target
            sample_weights = compute_sample_weight("balanced", y_train.iloc[:, i])
            lin = LinearRegression()
            lin.fit(X_train_scaled, y_train.iloc[:, i], sample_weight=sample_weights)
            preds = lin.predict(X_test_scaled)

            lin_mses.append(mean_squared_error(y_test.iloc[:, i], preds))
            lin_r2s.append(r2_score(y_test.iloc[:, i], preds))

        model_results.append({
            "Model": "LinearRegression",
            "MSE": lin_mses,
            "R²": lin_r2s
        })

        # Save results
        for result in model_results:
            model_name = result["Model"]
            output_file = os.path.join(output_dir, f"{model_name}_{target_label}_results.csv")
            pd.DataFrame.from_dict(result, orient="index").to_csv(output_file)
            print(f"Results saved for {model_name} at {target_label}: {output_file}")

    print("Training completed for all targets.")

if __name__ == "__main__":
    train_models()
