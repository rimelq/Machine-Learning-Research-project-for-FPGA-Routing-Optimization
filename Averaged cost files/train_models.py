import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score
)

def train_models(input_dir="datasets", output_dir="results", target_percentages=[0.5, 0.8, 1.0], lambdas=[0.0001, 0.001, 0.01, 0.1, 1.0]):
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
            ridge_model = Ridge(alpha=alpha)
            ridge_model.fit(X_train_scaled, y_train)
            ridge_preds = ridge_model.predict(X_test_scaled)

            ridge_mse = mean_squared_error(y_test, ridge_preds)
            ridge_r2 = r2_score(y_test, ridge_preds)

            model_results.append({
                "Model": f"Ridge_alpha_{alpha}",
                "MSE": ridge_mse.tolist(),
                "R²": ridge_r2.tolist()
            })

        # Random Forest 
        print("Training Random Forest...")
        rf_model = RandomForestRegressor(random_state=42)
        rf_model.fit(X_train, y_train)
        rf_preds = rf_model.predict(X_test)

        rf_mse = mean_squared_error(y_test, rf_preds)
        rf_r2 = r2_score(y_test, rf_preds)

        model_results.append({
            "Model": "RandomForest",
            "MSE": rf_mse.tolist(),
            "R²": rf_r2.tolist()
        })

        # Linear Regression 
        print("Training Linear Regression...")
        lin_model = LinearRegression()
        lin_model.fit(X_train_scaled, y_train)
        lin_preds = lin_model.predict(X_test_scaled)

        lin_mse = mean_squared_error(y_test, lin_preds)
        lin_r2 = r2_score(y_test, lin_preds)

        model_results.append({
            "Model": "LinearRegression",
            "MSE": lin_mse.tolist(),
            "R²": lin_r2.tolist()
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
