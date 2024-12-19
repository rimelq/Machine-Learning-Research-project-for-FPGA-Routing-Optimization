This project addresses challenges related to FPGA congestion, integrating machine learning to manage resource utilization effectively. 

The main objective of this research project is to explore feature engineering techniques that can effectively monitor the dynamic evolution of constrained zones. This involves leveraging cost-related data to design features capable of capturing these changes. To achieve this, we tested various feature ideas and evaluated them using simple machine learning models to determine whether these features represent viable avenues for future research. Machine learning models are subsequently utilized to assess the viability of these features, determining their potential value for future research directions and practical applications.

Below is a detailed breakdown of the project structure and the methodologies employed, with a particular emphasis on feature creation and FPGA-specific adaptations.

Code Organization : 

The project structure includes two main directories, each with specialized functionality:

1. create_dataset.py
This script is responsible for generating and preparing datasets. Key functionalities include:
- Master file for creating features and targets required for model training.
- Handling data export to formats suitable for machine learning pipelines (CSV, ...).

3. functions.py
This script provides utility functions for:
- Preprocessing datasets, such as cleaning and splitting.
- Basic feature engineering and transformations.
- All functions used to create the features and generate the feature set as well as the targets

4. train_models.py
This script implements model training using the prepared dataset. It:
- Trains machine learning models without accounting for weighted samples.
- Provides evaluation metrics for the trained models.
- Using MultiOutput functionality when we do predictions on the summed cost per channel (2 outputs : for CHANX and CHANY)

6. train_models_weighted_samples.py
This script is similar to train_models.py but specifically:
- Incorporates sample weights to address class imbalance.
- Trains models with weighted samples to improve performance on underrepresented classes.

Preprocessing

The preprocessing phase ensures that the data is ready for feature generation and model training. Below are the key functions used, organized by steps:

1. Data Preparation
   
Parsing Route Files : 
The extract_benchmark(number) function parses .route files containing routing information for FPGA benchmarks.
At its core, the parse_route_trees(filename, only_parse_nets=None) function transforms .route files into route_tree structures, represented as directed graphs (DiGraphs) for each net.
Each node in the graph includes detailed attributes such as:
- Start and end coordinates (start_coords, end_coords)
- Wire length (wire_length)
- Track information (track)
- Node type (SINK, SOURCE, CHANX, CHANY, ...)
The parsing process also filters specific nets if needed, providing flexibility for focused analysis.
Parsing route files is one of the most critical steps in this project as it lays the foundation for all information extraction for feature calculations.

Parsing History Cost Files :
The extract_benchmark(number) function also parses .hcost files, which store historical cost data for routing nodes.
These costs are mapped to specific nodes and iterations to track the evolution of routing challenges over time.

Outputs of these two first steps include dictionaries for both route_tree and history_costs, which serve as inputs for feature generation.

2. Feature Calculation
   
- Calculate_pin_density(route_tree) : 

Computes the density of input (IPIN) and output (OPIN) pins for each tile.
Returns a dictionary mapping tile coordinates to pin density values.

- Calculate_channel_usage(route_tree) : 

Calculates the total usage of horizontal (CHANX) and vertical (CHANY) channels for each tile.
Separates usage by fixed wire lengths (e.g., 1, 2, 4, 12).
Outputs a detailed breakdown of channel usage per tile.

- Calculate_tile_density(channel_usage, coords, neighborhood_size) : 

Computes the proportion of wires used in a for a specific tile area.
Allows analysis of local and global usage patterns.

- Calculate_channel_bias_for_tile(channel_usage, coords, neighborhood_size) : 

Calculates the bias between horizontal (CHANX) and vertical (CHANY) channel usage for a specific tile area.
Useful for identifying routing imbalances and sources of congestion.

3. Cost Aggregation
   
Calculate_sum_tile_costs(tile_nodes, hcost_data, relevant_iterations) or calculate_weighted_average_cost(tile_nodes, hcost_data, relevant_iterations) : 
- The calculate_sum_tile_costs function computes a simple sum of costs for each channel.
- The calculate_weighted_average_cost function calculates weighted average costs or each tile by applying a logarithmic scaling formula. Weights are determined by the frequency of cost values, ensuring a focus on infrequent high-cost events.

4. Feature Generation : 
   
Generate_feature_set(route_data, hcost_data, total_iterations, target_percentages=[1.0]) function : 
- Combines pin density, channel usage, and cost metrics into a structured dataset.
- Generates features for each tile and extracts targets for specific iteration percentages.
- Returns a DataFrame of features and corresponding targets.



Workflow

Requirements:
Install required packages using:
pip install -r pip-requirements.txt

Steps to Follow 

Dataset Generation:
Use create_dataset.py to generate a dataset with features and labels files : one file for each benchmark.

NB: running create_dataset.py for all benchmarks at once uses too much memory. We strongly advise to run one benchmark at a time.

Model Training:
- Group the benchmarks features and targets into one block and shuffle the tile rows
- Split the dataset into training and testing sets : 70-30%
- For standard model training, use train_models.py.
- For training with weighted samples, use train_models_weighted_samples.py.
