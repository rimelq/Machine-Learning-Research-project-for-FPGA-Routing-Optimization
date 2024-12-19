""" functions """

import os
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd

from natsort import natsorted

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import math
from collections import defaultdict


# Constants
PIN_COUNTS = {
    'CLB': {'IPIN': 72, 'OPIN': 32},
    'BRAM': {'IPIN': 43, 'OPIN': 33},
    'DSP': {'IPIN': 132, 'OPIN': 81},
    'IO': {'IPIN': 128, 'OPIN': 64},
    # Add default values for unknown block types if necessary
    'UNKNOWN': {'IPIN': 1, 'OPIN': 1},  
}

get_net = lambda line : line.split()[1]
get_node = lambda line : int(line.split()[1])
get_node_type = lambda line : line.split()[2]
get_coords = lambda line : line.split()[3].split(',')
get_x = lambda line : int(get_coords(line)[0][1:])
get_y = lambda line : int(get_coords(line)[-1][:-1])

##########################################################################
def parse_node_users(filename):
    """Parses signals that use each used RRG node.

    Parameters
    ----------
    filename : str
        Name of the file containing the .route to be parsed.

    Returns
    -------
    Dict[int, Set[str]]
        Node-user singal map.
    """

    with open(filename, "r") as inf:
        lines = inf.readlines()

    user_map = {}
    net = None
    for line in lines:
        if line.startswith("Node:"):
            if line.split()[2] in ("SOURCE", "SINK"):
                continue
            node = get_node(line)
            try:
                user_map[node].add(net)
            except:
                user_map.update({node : set([net])})
        elif line.startswith("Net"):
            net = get_net(line)

    return user_map
##########################################################################

##########################################################################
def get_node_coords(filename):
    """Parses the coordinates of all nodes.

    Paramters
    ---------
    filename : str
        Name of the .route file to parse.

    Returns
    -------
    Dict[int, Tuple[int]]
        Coordinates of all nodes.
    """

    with open(filename, "r") as inf:
        lines = inf.readlines()

    coord_dict = {}
    for line in lines:
        if line.startswith("Node:"):
            if not get_node_type(line).startswith("CHAN"):
                continue
            coord_dict.update({get_node(line) : (get_x(line), get_y(line))})


    return coord_dict
##########################################################################

##########################################################################
def get_coords_and_length(line):
    """Parses coordinates and computes wire length for CHAN nodes.

    Parameters
    ----------
    line : str
        A single line from the .route file.

    Returns
    -------
    Tuple[int, int, int, int, int]
        Start X, Start Y, End X, End Y, Wire Length.
    """
    # Split the line into parts
    parts = line.split()

    # Locate the coordinates
    coords_index = next(i for i, part in enumerate(parts) if "(" in part and ")" in part)
    # Extract the part with coordinates
    coords = parts[coords_index]  

    if "to" in parts:  # Check if "to" exists in the line (multi-coordinate case)
        # Multi-coordinate case
        start_coords = coords
        # "to" is at index +1, so end_coords is at index +2
        end_coords = parts[coords_index + 2]  
        start_x, start_y = map(int, start_coords.strip("()").split(','))
        end_x, end_y = map(int, end_coords.strip("()").split(','))
        wire_length = abs(end_x - start_x) + abs(end_y - start_y) + 1
    else:
        # Single-coordinate case
        start_x, start_y = map(int, coords.strip("()").split(','))
        end_x, end_y = start_x, start_y
        wire_length = 1 if get_node_type(line).startswith("CHAN") else None

    return start_x, start_y, end_x, end_y, wire_length



##########################################################################

##########################################################################
def get_track(line):
    """Extracts the track number for CHAN nodes, accounting for multi-coordinate nodes.

    Parameters
    ----------
    line : str
        A single line from the .route file.

    Returns
    -------
    int or None
        Track number if applicable, otherwise None.
    """
    # Check if the node type is CHAN (CHANX or CHANY)
    if get_node_type(line).startswith("CHAN"):
        # Split the line and check where "Track:" appears
        parts = line.split()
        for i, part in enumerate(parts):
            if part == "Track:":
                # Return the number after "Track:"
                return int(parts[i + 1])  
    return None  # Return None if no track info is found or for non-CHAN nodes

##########################################################################
 
##########################################################################

def get_bloc_type(line):
    """Extracts the bloc type from the line, taking everything before the first period.

    Parameters
    ----------
    line : str
        A single line from the .route file.

    Returns
    -------
    str or None
        The extracted bloc type, or None if no valid bloc type is found.
    """
    # Find the position of the first period
    period_index = line.find(".")
    if period_index != -1:  # Ensure there is a period in the line
        # Extract everything before the period
        substring = line[:period_index]
        # Find the last space before the period
        last_space_index = substring.rfind(" ")
        if last_space_index != -1:
            return substring[last_space_index + 1:]  # Return from the last space to the period
        else:
            # If no space is found, return the whole substring
            return substring
        
    # Return None if no period is found
    return None  

##########################################################################
 
##########################################################################

def parse_route_trees(filename, only_parse_nets=None):
    """Parses the entire route trees for all nets.

    Parameters
    ----------
    filename : str
        Name of the file containing the .route to be parsed.
    only_parse_nets : Optional[list[str]], default = None
        Specifies that only a subset of nets should be parsed.
        None means parse all.

    Returns
    -------
    Dict[str, nx.DiGraph]
        Route trees for all nets.
    """

    with open(filename, "r") as inf:
        lines = inf.readlines()

    if only_parse_nets is not None:
        only_parse_nets = set(only_parse_nets)

    rts = {}
    net = None
    rt = nx.DiGraph()
    prev_node = None
    skip = False
    global_skip = False

    for line in lines:
        if line.startswith("Node:"):
            if skip or global_skip:
                continue

            # Extract node details
            node = get_node(line)
            start_x, start_y, end_x, end_y, wire_length = get_coords_and_length(line)
            track = get_track(line)

            # Add node to the graph
            rt.add_node(
                node,
                start_coords=(start_x, start_y),
                end_coords=(end_x, end_y),
                wire_length=wire_length,
                track=track,
                node_type=get_node_type(line),
            )
            if prev_node is not None:
                rt.add_edge(prev_node, node)
            if get_node_type(line) == "SINK":
                prev_node = None
            else:
                prev_node = node
        elif line.startswith("Net"):
            # Check if this is a global net
            if ":" in line:
                # Skip lines after global net until a new valid net
                global_skip = True  
                continue
            else:
                # Reset global skip for valid nets
                global_skip = False  

            # Process the current net
            if not skip and net is not None:
                rts.update({net: rt})
                rt = nx.DiGraph()
            net = get_net(line)
            skip = True
            if only_parse_nets is None or net in only_parse_nets:
                skip = False

    # Update the last net if necessary
    if not skip and net is not None:
        rts.update({net: rt})
        rt = nx.DiGraph()

    return rts
##########################################################################

##########################################################################
# function to read history cost file
def read_history_cost_dict(filename):
    """
    Reads the history cost file and stores the data in a dictionary.

    Parameters:
    -----------
    filename : str
        File containing the history cost information.

    Returns:
    --------
    dict
        A dictionary where keys are node IDs and values are history costs.
    """
    # Read the file and create a dictionary
    history_cost_per_node = {}
    with open(filename, "r") as inf:
        for line in inf:
            node_id, history_cost = line.split()
            history_cost_per_node[int(node_id)] = float(history_cost)
    return history_cost_per_node

##########################################################################


###########################################################################

def print_net_info(route_trees, net_id):
    """
    Prints all information (nodes and edges) of a specific net.

    Parameters:
    -----------
    route_trees : dict
        Dictionary where keys are net IDs and values are NetworkX DiGraphs.
    net_id : str
        The net ID of the graph you want to print.
    """
    if net_id in route_trees:
        print(f"--- Information for Net: {net_id} ---")
        net_graph = route_trees[net_id]
        
        print("\nNodes:")
        for node, attrs in net_graph.nodes(data=True):
            print(f"Node: {node}")
            for key, value in attrs.items():
                print(f"  {key}: {value}")
    else:
        print(f"Net ID '{net_id}' not found in the route_trees.")

###############################################################

###############################################################

def calculate_pin_density(route_tree):
    """
    Calculate input and output pin densities for each tile, adjusting for block type.

    Parameters:
    -----------
    route_tree : dict
        Dictionary containing the routing information for all nets (DiGraphs).

    Returns:
    --------
    dict
        A dictionary where keys are tile coordinates (x, y) and values are
        tuples of (input pin density, output pin density).
    """

    ipin_counts = {}
    opin_counts = {}
    block_types = {}

    # Iterate through all nets 
    for net_id, graph in route_tree.items():
        if not isinstance(graph, nx.DiGraph):
            print(f"Invalid graph for net_id: {net_id}, type: {type(graph)}")
            continue

        for node_id, attrs in graph.nodes(data=True):
            coords = attrs['start_coords']  # Coordinates of the tile
            node_type = attrs['node_type']
            block_type = attrs.get('block_type', 'UNKNOWN')  # Default to 'UNKNOWN'
            
            # Store block type for each coordinate
            block_types[coords] = block_type

            if node_type == 'IPIN':  
                ipin_counts[coords] = ipin_counts.get(coords, 0) + 1
            elif node_type == 'OPIN':  
                opin_counts[coords] = opin_counts.get(coords, 0) + 1

    # Calculate densities
    pin_density = {}
    for coords in set(ipin_counts.keys()).union(set(opin_counts.keys())):
        block_type = block_types.get(coords, 'UNKNOWN')  # Default to 'UNKNOWN'
        if block_type not in PIN_COUNTS:
            # Log or handle unknown block types
            print(f"Warning: Unknown block type at {coords}. Skipping...")
            continue
        total_ipins = PIN_COUNTS[block_type]['IPIN']
        total_opins = PIN_COUNTS[block_type]['OPIN']
        ipin_density = ipin_counts.get(coords, 0) / total_ipins
        opin_density = opin_counts.get(coords, 0) / total_opins
        pin_density[coords] = (ipin_density, opin_density)

    return pin_density



##################################################################

##################################################################

def plot_pin_density(pin_density, density_type='Input'):
    """
    Visualize pin density (input or output) as a heatmap.

    Parameters:
    -----------
    pin_density : dict
        Dictionary with tile coordinates as keys and tuples of
        (input pin density, output pin density) as values.
    density_type : str
        Either 'Input' or 'Output' to specify which density to plot.
    """
    # Extract coordinates and densities
    x_coords = [coords[0] for coords in pin_density.keys()]
    y_coords = [coords[1] for coords in pin_density.keys()]
    if density_type == 'Input':
        densities = [values[0] for values in pin_density.values()]
    elif density_type == 'Output':
        densities = [values[1] for values in pin_density.values()]
    else:
        raise ValueError("density_type must be 'Input' or 'Output'")

    # Create the scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x_coords, y_coords, c=densities, cmap='coolwarm', s=100)
    plt.colorbar(scatter, label=f'{density_type} Pin Density')
    plt.title(f'{density_type} Pin Density per Tile')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.show()

#####################################################################

#####################################################################

def calculate_channel_usage(route_tree):
    """
    Calculate the number of wires (tracks) used in each tile, separated into
    horizontal (CHANX) and vertical (CHANY) channels. Also, calculate the
    number of tracks per fixed wire lengths (1, 2, 4, 12).

    Parameters:
    -----------
    route_tree : dict
        Dictionary containing the routing information for all nets (DiGraphs).

    Returns:
    --------
    dict
        A dictionary with keys 'CHANX' and 'CHANY', each containing:
        - 'total': Total number of tracks used in each tile.
        - 'per_length': Number of tracks used in each tile per fixed wire length.
    """
    # Initialize dictionaries 
    channel_usage = {
        'CHANX': {'total': {}, 'per_length': {}},
        'CHANY': {'total': {}, 'per_length': {}},
    }

    fixed_lengths = [1, 2, 4, 12]

    # Helper function to traverse from start to end
    def traverse_points(start, end, channel_key):
        points = []
        if channel_key == 'CHANX':  # x changes, y is constant
            x1, y = start
            x2, _ = end
            for x in range(min(x1, x2), max(x1, x2) + 1):
                points.append((x, y))
        elif channel_key == 'CHANY':  # y changes, x is constant
            x, y1 = start
            _, y2 = end
            for y in range(min(y1, y2), max(y1, y2) + 1):
                points.append((x, y))
        return points

    # Iterate through all nets 
    for net_id, graph in route_tree.items():
        for node_id, attrs in graph.nodes(data=True):
            node_type = attrs['node_type']
            if node_type not in ['CHANX', 'CHANY']:
                continue

            start_coords = attrs['start_coords']
            end_coords = attrs['end_coords']  # Assuming 'end_coords' exists in attrs
            wire_length = attrs['wire_length']

            # Determine the channel type
            channel_key = 'CHANX' if node_type == 'CHANX' else 'CHANY'

            # Traverse all points from start to end
            points = traverse_points(start_coords, end_coords, channel_key)

            for coords in points:
                # Update total count for the tile
                channel_usage[channel_key]['total'][coords] = (
                    channel_usage[channel_key]['total'].get(coords, 0) + 1
                )

                # Initialize per_length if not present
                if coords not in channel_usage[channel_key]['per_length']:
                    channel_usage[channel_key]['per_length'][coords] = {length: 0 for length in fixed_lengths}

                # Update count for the specific wire length
                if wire_length in fixed_lengths:
                    channel_usage[channel_key]['per_length'][coords][wire_length] += 1

    return channel_usage
##################################################################################

##################################################################################

def plot_channel_usage(channel_usage, channel_type='CHANX', wire_length=None):
    """
    Visualize channel usage (total tracks or tracks per wire length) as a heatmap.

    Parameters:
    -----------
    channel_usage : dict
        Dictionary containing channel usage data (output of calculate_channel_usage).
    channel_type : str
        Either 'CHANX' or 'CHANY' to specify the channel type to visualize.
    wire_length : int or None
        Specific wire length to visualize. If None, visualize total tracks.
    """
    usage_data = channel_usage[channel_type]
    if wire_length:
        # Use tracks for a specific wire length
        data = {coords: lengths.get(wire_length, 0) for coords, lengths in usage_data['per_length'].items()}
        title = f'{channel_type} Tracks (Length {wire_length})'
    else:
        # Use total tracks
        data = usage_data['total']
        title = f'Total {channel_type} Tracks'

    x_coords = [coords[0] for coords in data.keys()]
    y_coords = [coords[1] for coords in data.keys()]
    values = list(data.values())

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x_coords, y_coords, c=values, cmap='viridis', s=100)
    plt.colorbar(scatter, label=title)
    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.show()


####################################################################################

####################################################################################
def calculate_weighted_average_cost(route_tree, history_costs):
    """
    Calculate the weighted average cost of wires (CHANX and CHANY) for each tile 
    using the logarithmic scaling formula with cost counts.

    Parameters:
    -----------
    route_tree : dict
        A dictionary where keys are net IDs and values are DiGraphs for each net.
    history_costs : dict
        Dictionary mapping node IDs to their corresponding history costs.

    Returns:
    --------
    dict
        A dictionary where keys are tile coordinates (x, y) and values are the weighted average costs.
    """
    tile_costs = defaultdict(list)
    cost_counts = defaultdict(int)

    # Count occurrences of each cost
    for cost in history_costs.values():
        cost_counts[cost] += 1
    print('count', cost_counts)
        
    # Iterate through all nets and nodes in the route_tree
    for net_id, graph in route_tree.items():
        for node_id, attrs in graph.nodes(data=True):
            coords = attrs['start_coords']  # Tile coordinates
            node_type = attrs['node_type']
            cost = history_costs.get(node_id, 1)  # Default cost is 1 if missing

            # Only process wire nodes (CHANX or CHANY)
            if node_type in ['CHANX', 'CHANY']:
                # Apply the weight formula: 
                if cost!=0:
                    weight = 1 / (math.log(cost_counts[cost] + 1))**8 if cost_counts[cost] > 0 else 0
                    weighted_cost = weight * cost
                    tile_costs[coords].append(weighted_cost)

    # Calculate the weighted average cost per tile
    weighted_avg_costs = {
        coords: sum(costs) / len(costs) for coords, costs in tile_costs.items()
    }
    print('finished wighted av!!!!')
    return weighted_avg_costs

###################################################################################

###################################################################################

def get_total_iterations_for_benchmark(dir):
    """
    Get the total number of iterations for a single benchmark.

    Parameters:
     -----------
    dir : str
        Path to the benchmark folder containing .hcost files.

    Returns:
    --------
    int
        Total number of iterations for the benchmark.
    """
    # List all history cost files in the directory
    hcost_files = [f for f in os.listdir(dir) if f.endswith('.hcost')]

    # Extract the iteration numbers
    iteration_numbers = [
        int(hcost_file.replace('history_cost_iteration_', '').replace('.hcost', ''))
        for hcost_file in hcost_files
    ]

    # Find the maximum iteration number
    total_iterations = max(iteration_numbers)

    return total_iterations

###################################################################################

###################################################################################

def extract_benchmark(number = '01'):
    # extract .route files
    path = './data/FPGA'+number
    route_files = natsorted([f for f in os.listdir(path) if f.startswith('iteration_') and f.endswith('.route')])

    route_data = {}  # Dictionary to store parsed data for each iteration

    for route_file in route_files:
        full_path = os.path.join(path, route_file)

        # Parse the route tree
        route_trees = parse_route_trees(full_path)

        # Store the parsed data in a dictionary
        iteration_key = route_file.replace('.route', '')
        route_data[iteration_key] = route_trees  # Each iteration contains its DiGraphs

    # read history cost file

    hcost_files = natsorted([f for f in os.listdir(path) if f.startswith('history_cost_iteration_') and f.endswith('.hcost')])
    hcost_data = {}

    for hcost_file in hcost_files:
        full_path = os.path.join(path, hcost_file)

        hcost = read_history_cost_dict(full_path)

        # store in a dictionnary
        iteration_key = hcost_file.replace('history_cost_', '').replace('.hcost', '')
        hcost_data[iteration_key] = hcost

    total_iter = get_total_iterations_for_benchmark(path)

    return route_data, hcost_data, total_iter

####################################################################################

####################################################################################

def calculate_tile_density(channel_usage, coords, neighborhood_size=0):
    """
    Calculate the tile density for a neighborhood around a given tile.

    Parameters:
    -----------
    channel_usage : dict
        Dictionary containing channel usage data.
    coords : tuple
        Coordinates of the central tile.
    neighborhood_size : int
        The radius of the neighborhood to include.

    Returns:
    --------
    float
        Proportion of total usage relative to the total wires in the neighborhood.
    """
    total_wires_per_tile = 56 * 2 
    total_wires_in_neighborhood = 0
    total_usage_in_neighborhood = 0

    x, y = coords
    for dx in range(-neighborhood_size, neighborhood_size + 1):
        for dy in range(-neighborhood_size, neighborhood_size + 1):
            neighbor_coords = (x + dx, y + dy)
            tile_usage = channel_usage['CHANX']['total'].get(neighbor_coords, 0) + \
                         channel_usage['CHANY']['total'].get(neighbor_coords, 0)
            total_usage_in_neighborhood += tile_usage
            total_wires_in_neighborhood += total_wires_per_tile

    if total_wires_in_neighborhood == 0:
        return 0

    return total_usage_in_neighborhood / total_wires_in_neighborhood

####################################################################################

####################################################################################

def calculate_channel_bias_for_tile(channel_usage, coords, neighborhood_size=0):
    """
    Calculate the channel Bias for a specific tile, considering a neighborhood.

    Parameters:
    -----------
    channel_usage : dict
        Dictionary containing channel usage data.
    coords : tuple
        Coordinates of the tile.
    neighborhood_size : int, optional
        The radius of the neighborhood to include (default is 0 for the tile itself).

    Returns:
    --------
    float
        Channel bias for the tile and its neighborhood.
    """
    chanx_used_total = 0
    chany_used_total = 0

    x, y = coords

    # Iterate over the neighborhood
    for dx in range(-neighborhood_size, neighborhood_size + 1):
        for dy in range(-neighborhood_size, neighborhood_size + 1):
            neighbor_coords = (x + dx, y + dy)
            chanx_used_total += channel_usage['CHANX']['total'].get(neighbor_coords, 0)
            chany_used_total += channel_usage['CHANY']['total'].get(neighbor_coords, 0)

    total_usage = chanx_used_total + chany_used_total
    if total_usage == 0:
        return 0  # Avoid division by zero
    return (chanx_used_total - chany_used_total) / total_usage

####################################################################################

####################################################################################

def generate_feature_set(route_data, hcost_data, total_iterations, target_percentages=1.0):
    """
    Generate the feature set for each tile using precomputed inputs and choose a specific target iteration.

    Parameters:
    -----------
    route_data : dict
        Parsed route data for the benchmark.
    hcost_data : dict
        Dictionary of history cost data for each iteration (keyed by iteration ID).
    total_iterations : int
        Total number of iterations available for the benchmark.
    target_iteration_percentage : float, optional
        Percentage of total iterations to use as the target (default is 1.0 for the last iteration).

    Returns:
    --------
    pd.DataFrame, pd.Series
        Features and target for the model.
    """
    # Flatten route_data to pass to feature calculation functions
    all_route_trees = {}
    for iteration_key, nets in route_data.items():
        for net_id, graph in nets.items():
            if isinstance(graph, nx.DiGraph):
                all_route_trees[net_id] = graph
            else:
                print(f"Invalid graph for net_id: {net_id}, type: {type(graph)}")

    # Calculate features using the flattened route_data
    pin_density = calculate_pin_density(all_route_trees)
    channel_usage = calculate_channel_usage(all_route_trees)

    avg_costs = {}
    print('1')
    for iteration_key, route_tree in route_data.items():
        print('2')
        avg_costs_per_iter = calculate_weighted_average_cost(route_tree, hcost_data[iteration_key])
        avg_costs[iteration_key] = avg_costs_per_iter

    # Define relevant iterations
    relevant_iterations = ['iteration_001', 'iteration_002', 'iteration_003']
    if total_iterations > 3:
        relevant_iterations.extend([
            f'iteration_{int(total_iterations * 0.5):03d}',
            f'iteration_{int(total_iterations * 0.8):03d}',
            f'iteration_{total_iterations:03d}',
        ])

    # Collect all unique tile coordinates
    all_coords = set(pin_density.keys()).union(
        channel_usage['CHANX']['total'].keys(), channel_usage['CHANY']['total'].keys()
    ) 

    feature_set = []
    for coords in all_coords:  
        features = {
            'Input Pin Density': pin_density.get(coords, (0, 0))[0],
            'Output Pin Density': pin_density.get(coords, (0, 0))[1],
            'Local tile Density Usage': calculate_tile_density(channel_usage, coords),
            'Global tile Density Usage': calculate_tile_density(channel_usage, coords, 5),
            'Local Channel Bias': calculate_channel_bias_for_tile(channel_usage, coords),
            'Global Channel Bias': calculate_channel_bias_for_tile(channel_usage, coords, 5),
        }
        # Add costs for the relevant iterations
        # Include costs for first three iterations
        for iteration in relevant_iterations:
            features[f'Cost {iteration}'] = avg_costs.get(iteration, {}).get(coords, 1)

        feature_set.append(features)

    feature_set_df = pd.DataFrame(feature_set)

    # Step 6: Extract targets for each percentage
    print("Extracting and renaming targets for all percentages...")
    targets_dict = {}
    for percentage in target_percentages:
        iteration_suffix = f'iteration_{int(total_iterations * percentage):03d}'
        target_columns = [
            f'Cost {iteration_suffix}'
        ]
        renamed_columns = {
            f'Cost {iteration_suffix}': f'Cost {int(percentage * 100)}%'
        }
        if all(col in feature_set_df.columns for col in target_columns):
            targets_dict[percentage] = feature_set_df[target_columns].rename(columns=renamed_columns)
        else:
            raise ValueError(f"Missing target columns for percentage {percentage * 100}%.")

    # Step 7: Define features using first three iterations and other columns
    print("Finalizing feature columns...")
    feature_columns = [col for col in feature_set_df.columns 
                       if not any(col.startswith(f"Cost iteration_") or col.startswith(f"Cost iteration_") 
                                  for percentage in target_percentages)]
    feature_columns.extend([
        'Cost iteration_001', 'Cost iteration_002',
        'Cost iteration_003'
    ])

    features = feature_set_df[feature_columns]

    print(f"Final features shape: {features.shape}")
    print(f"Targets prepared for percentages: {list(targets_dict.keys())}\n")
    return features, targets_dict


###########################################################################################

###########################################################################################
