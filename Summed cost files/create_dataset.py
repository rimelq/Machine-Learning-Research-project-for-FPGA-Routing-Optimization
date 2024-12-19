import os
import pandas as pd
from functions import *
from sklearn.utils import shuffle

def create_datasets(output_dir="datasets", target_percentages=[0.5, 0.8, 1.0]):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize dictionaries
    all_route_data = {}
    all_hcost_data = {}
    all_total_iter = {}

    # Extract data for all benchmarks (e.g., 01 to 12)
    for i in range(1, 13):  # Adjusted range for benchmarks
        benchmark_number = f'{i:02d}'
        print(f"Extracting benchmark {benchmark_number}...")
        route_data, hcost_data, total_iter = extract_benchmark(benchmark_number)
        all_route_data[benchmark_number] = route_data
        all_hcost_data[benchmark_number] = hcost_data
        all_total_iter[benchmark_number] = total_iter

    print('\nAll benchmarks extracted successfully.\n')

    # Generate feature sets for all benchmarks and targets
    for i in range(1, 13):  # Adjusted range for benchmarks
        benchmark_number = f'{i:02d}'
        print(f"\n--- Generating feature set for benchmark {benchmark_number} ---")
        route_data = all_route_data[benchmark_number]
        hcost_data = all_hcost_data[benchmark_number]
        total_iter = all_total_iter[benchmark_number]

        print(f"Generating feature set (X) and targets for benchmark {benchmark_number}...")
        try:
            X_bench, targets_dict = generate_feature_set(route_data, hcost_data, total_iter, target_percentages)
        except Exception as e:
            print(f"Error during feature generation for benchmark {benchmark_number}: {e}")
            continue

        # Save the feature set (X)
        X_bench.to_csv(os.path.join(output_dir, f"X_bench_{benchmark_number}.csv"), index=False)
        print(f"Feature set (X) saved for benchmark {benchmark_number}.")

        # Save each target percentage
        for target_percentage, y_target in targets_dict.items():
            y_target.to_csv(
                os.path.join(output_dir, f"y_bench_{benchmark_number}_target_{int(target_percentage * 100)}.csv"),
                index=False
            )
            print(f"Target (y) saved for benchmark {benchmark_number} at {int(target_percentage * 100)}%.")

    print(f"\nAll feature sets and targets saved successfully in '{output_dir}'.\n")

if __name__ == "__main__":
    create_datasets()

