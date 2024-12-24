# FPGA Congestion Management with Machine Learning

This project addresses challenges related to FPGA congestion, integrating machine learning to manage resource utilization effectively. 

## Objectives

The main objective of this research project is to explore feature engineering techniques that can effectively monitor the dynamic evolution of constrained zones. This involves leveraging cost-related data to design features capable of capturing these changes. To achieve this, we tested various feature ideas and evaluated them using simple machine learning models to determine whether these features represent viable avenues for future research. Machine learning models are subsequently utilized to assess the viability of these features, determining their potential value for future research directions and practical applications.

Below is a detailed breakdown of the project structure and the methodologies employed, with a particular emphasis on feature creation and FPGA-specific adaptations.

---

## Code Organization

The project structure includes the following main scripts and their functionalities:

1. **`create_dataset.py`**  
   - Generates and prepares datasets.  
   - Master file for creating features and targets required for model training.  
   - Exports data to formats suitable for machine learning pipelines (CSV, etc.).

2. **`functions.py`**  
   - Utility functions for:  
     - Preprocessing datasets (e.g., cleaning and splitting).  
     - Basic feature engineering and transformations.  
     - Creating features and generating the feature set and targets.

3. **`train_models.py`**  
   - Trains machine learning models without accounting for weighted samples.  
   - Provides evaluation metrics for the trained models.  
   - Uses MultiOutput functionality for predictions on the summed cost per channel (2 outputs: CHANX and CHANY).

4. **`train_models_weighted_samples.py`**  
   - Similar to `train_models.py`, but incorporates sample weights.  
   - Addresses class imbalance by training models with weighted samples.

---

## Preprocessing

The preprocessing phase ensures the data is ready for feature generation and model training. Below are the key functions used, organized by steps:

### 1. Data Preparation

#### Parsing Route Files  
- **`extract_benchmark(number)`** parses `.route` files containing routing information for FPGA benchmarks.  
- **`parse_route_trees(filename, only_parse_nets=None)`** transforms `.route` files into `route_tree` structures, represented as directed graphs (`DiGraphs`) for each net.  
- Each node in the graph includes attributes such as:
  - Start and end coordinates (`start_coords`, `end_coords`)
  - Wire length (`wire_length`)
  - Track information (`track`)
  - Node type (e.g., `SINK`, `SOURCE`, `CHANX`, `CHANY`)
- Outputs include dictionaries for both `route_tree` and `history_costs`, serving as inputs for feature generation.

#### Parsing History Cost Files  
- The **`extract_benchmark(number)`** function also parses `.hcost` files to track historical cost data for routing nodes, mapping costs to specific nodes and iterations.

---

### 2. Feature Calculation

- **`calculate_pin_density(route_tree)`**  
  Computes the density of input (`IPIN`) and output (`OPIN`) pins for each tile.  
  Returns a dictionary mapping tile coordinates to pin density values.

- **`calculate_channel_usage(route_tree)`**  
  Calculates the total usage of horizontal (`CHANX`) and vertical (`CHANY`) channels for each tile, separated by fixed wire lengths (e.g., 1, 2, 4, 12).

- **`calculate_tile_density(channel_usage, coords, neighborhood_size)`**  
  Computes the proportion of wires used in a specific tile area for local and global usage patterns.

- **`calculate_channel_bias_for_tile(channel_usage, coords, neighborhood_size)`**  
  Calculates the bias between horizontal (`CHANX`) and vertical (`CHANY`) channel usage for a specific tile area, identifying routing imbalances.

---

### 3. Cost Aggregation

- **`calculate_sum_tile_costs(tile_nodes, hcost_data, relevant_iterations)`**  
  Computes the sum of costs for each channel.  

- **`calculate_weighted_average_cost(tile_nodes, hcost_data, relevant_iterations)`**  
  Calculates weighted average costs for each tile using a logarithmic scaling formula, focusing on infrequent high-cost events.

---

### 4. Feature Generation

- **`generate_feature_set(route_data, hcost_data, total_iterations, target_percentages=[1.0])`**  
  Combines pin density, channel usage, and cost metrics into a structured dataset.  
  Generates features for each tile and extracts targets for specific iteration percentages.  
  Returns a DataFrame of features and corresponding targets.

---

## Workflow

### Requirements  
Install required packages using:  
```bash
pip install -r pip-requirements.txt
```

### Steps to Follow  

#### Dataset Generation  
- Use `create_dataset.py` to generate a dataset with feature and label files for each benchmark.  
- **Note:** Running `create_dataset.py` for all benchmarks at once uses too much memory. We strongly advise running one benchmark at a time.  

#### Model Training  
- Group the benchmark features and targets into one block and shuffle the tile rows.  
- Split the dataset into training and testing sets (70-30%).  
- For standard model training, use `train_models.py`.  
- For training with weighted samples, use `train_models_weighted_samples.py`.  
