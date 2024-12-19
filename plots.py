import os
import pandas as pd
import matplotlib.pyplot as plt
import ast

def visualize_mse_by_target(results_dir, output_file="ridge_results.png"):
    """
    Visualize the Mean Squared Error (MSE) for different models across target percentages.

    Parameters:
    - results_dir (str): Directory containing the result CSV files.
    - output_file (str): File name to save the plot.
    """
    mse_data = []

    for file_name in os.listdir(results_dir):
        if not file_name.endswith(".csv") or "alpha" in file_name:
            continue

        file_path = os.path.join(results_dir, file_name)
        data = pd.read_csv(file_path, header=None)

        parts = file_name.split("_")
        model_name = parts[0]
        target_percentage = int(parts[2])

        mse = float(data.loc[data[0] == "MSE", 1].values[0].strip("[]"))
        mse_data.append((model_name, target_percentage, mse))

    mse_df = pd.DataFrame(mse_data, columns=["Model", "Target", "MSE"])
    mse_df = mse_df.sort_values(by=["Model", "Target"])

    plt.figure(figsize=(10, 6))
    for i, model in enumerate(mse_df["Model"].unique()):
        model_data = mse_df[mse_df["Model"] == model]
        plt.plot(
            model_data["Target"],
            model_data["MSE"],
            marker="o",
            label=model,
            alpha=0.8,
            linewidth=2 - 0.5 * i,
        )

    plt.title("MSE by Target Percentage for Different Models")
    plt.xlabel("Target Percentage (%)")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.legend()
    plt.grid()
    plt.savefig(output_file, format="png")
    print(f"Plot saved as {output_file}")
    plt.show()

def visualize_ridge_results_by_target(results_dir, output_file="ridge_results.png"):
    """
    Visualize the MSE of Ridge regression across different alpha values for each target percentage.

    Parameters:
    - results_dir (str): Directory containing Ridge regression result CSV files.
    - output_file (str): File name to save the plot.
    """
    results = []

    for file_name in os.listdir(results_dir):
        if file_name.startswith("Ridge_alpha_") and file_name.endswith(".csv"):
            file_path = os.path.join(results_dir, file_name)
            data = pd.read_csv(file_path, header=None, index_col=0)

            alpha = float(file_name.split("_")[2])
            target = int(file_name.split("_")[-2])

            mse = float(data.loc["MSE"][1].strip("[]"))
            results.append({"alpha": alpha, "target": target, "MSE": mse})

    results_df = pd.DataFrame(results).sort_values(["target", "alpha"])

    plt.figure(figsize=(10, 5))
    for target in results_df["target"].unique():
        subset = results_df[results_df["target"] == target]
        plt.plot(subset["alpha"], subset["MSE"], marker="o", label=f"Target {target}")
    plt.xscale("log")
    plt.xlabel("Alpha")
    plt.ylabel("Mean Squared Error")
    plt.title("Ridge Regression: MSE vs. Alpha by Target")
    plt.legend()
    plt.grid()
    plt.savefig(output_file, format="png")
    print(f"Plot saved as {output_file}")
    plt.show()

def visualize_mse_by_target_avg(results_dir, output_file="ridge_results.png"):
    """
    Visualize the MSE for different models and multiple targets across target percentages.

    Parameters:
    - results_dir (str): Directory containing result CSV files.
    - output_file (str): File name to save the plot.
    """
    mse_data = []

    for file_name in os.listdir(results_dir):
        if not file_name.endswith(".csv") or "alpha" in file_name:
            continue

        file_path = os.path.join(results_dir, file_name)
        data = pd.read_csv(file_path, header=None)

        parts = file_name.split("_")
        model_name = parts[0]
        target_percentage = int(parts[2])

        mse_raw = data.loc[data[0] == "MSE", 1].values[0]
        mse_list = ast.literal_eval(mse_raw)

        for target_idx, mse in enumerate(mse_list):
            mse_data.append((model_name, target_percentage, f"Target {target_idx + 1}", mse))

    mse_df = pd.DataFrame(mse_data, columns=["Model", "Target Percentage", "Target", "MSE"])
    mse_df = mse_df.sort_values(by=["Model", "Target Percentage", "Target"])

    plt.figure(figsize=(10, 6))
    for model in mse_df["Model"].unique():
        model_data = mse_df[mse_df["Model"] == model]
        for target in model_data["Target"].unique():
            target_data = model_data[model_data["Target"] == target]
            plt.plot(
                target_data["Target Percentage"],
                target_data["MSE"],
                marker="o",
                label=f"{model} - {target}",
                alpha=0.8,
            )

    plt.title("MSE by Target Percentage for Different Models and Targets")
    plt.xlabel("Target Percentage (%)")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.legend()
    plt.grid()
    plt.savefig(output_file, format="png")
    print(f"Plot saved as {output_file}")
    plt.show()

def visualize_ridge_results_by_target_avg(results_dir, output_file="ridge_results.png"):
    """
    Visualize the MSE of Ridge regression for multiple targets across target percentages.

    Parameters:
    - results_dir (str): Directory containing Ridge result CSV files.
    - output_file (str): File name to save the plot.
    """
    mse_data = []

    for file_name in os.listdir(results_dir):
        if not file_name.endswith(".csv") or not file_name.startswith("Ridge") or "alpha" in file_name:
            continue

        file_path = os.path.join(results_dir, file_name)
        data = pd.read_csv(file_path, header=None)

        parts = file_name.split("_")
        target_percentage = int(parts[2])

        mse_raw = data.loc[data[0] == "MSE", 1].values[0]
        mse_list = ast.literal_eval(mse_raw)

        for target_idx, mse in enumerate(mse_list):
            mse_data.append((target_percentage, f"Target {target_idx + 1}", mse))

    mse_df = pd.DataFrame(mse_data, columns=["Target Percentage", "Target", "MSE"])
    mse_df = mse_df.sort_values(by=["Target Percentage", "Target"])

    plt.figure(figsize=(10, 6))
    for target in mse_df["Target"].unique():
        target_data = mse_df[mse_df["Target"] == target]
        plt.plot(
            target_data["Target Percentage"],
            target_data["MSE"],
            marker="o",
            label=target,
            alpha=0.8,
        )

    plt.title("MSE by Target Percentage for Ridge Model")
    plt.xlabel("Target Percentage (%)")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.legend(title="Targets")
    plt.grid()
    plt.savefig(output_file, format="png")
    print(f"Plot saved as {output_file}")
    plt.show()
