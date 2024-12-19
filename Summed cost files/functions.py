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
    
    # Check if "to" exists in the line (multi-coordinate case)
    if "to" in parts:  
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

def get_block_type(line):
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
            # Return from the last space to the period
            return substring[last_space_index + 1:]  
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
            node_type = get_node_type(line)
            block_type = get_block_type(line)


            # Add node to the graph
            rt.add_node(
                node,
                start_coords=(start_x, start_y),
                end_coords=(end_x, end_y),
                wire_length=wire_length,
                track=track,
                node_type=node_type,
                block_type=block_type,
            )
            if prev_node is not None:
                rt.add_edge(prev_node, node)
            if get_node_type(line) == "SINK":
                prev_node = None
            else:
                prev_node = node
        elif line.startswith("Net"):
            # Check if this is a global net
            if "global" in line:
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


def get_tile_coordinates(route_data):
    """
    Extract the coordinates of each node and group them into tiles, 
    including intermediate tiles for CHANX and CHANY nodes.
    
    Now we store both node_id and node_type in the set.
    """

    tile_nodes = defaultdict(set)

    def traverse_wire_nodes(start_coords, end_coords, node_id):
        points = set()
        x1, y1 = start_coords
        x2, y2 = end_coords

        # Horizontal wires (CHANX): y remains constant, x varies
        if y1 == y2:
            for x in range(min(x1, x2), max(x1, x2) + 1):
                points.add((x, y1))
        # Vertical wires (CHANY): x remains constant, y varies
        elif x1 == x2:
            for y in range(min(y1, y2), max(y1, y2) + 1):
                points.add((x1, y))

        # Add the start and end points explicitly
        points.add(start_coords)
        points.add(end_coords)

        return points

    # Iterate through all nets in the routing data
    for _, nets in route_data.items():
        for _, graph in nets.items():
            for node_id, attrs in graph.nodes(data=True):
                start_coords = attrs['start_coords']
                end_coords = attrs.get('end_coords', start_coords)
                node_type = attrs['node_type']

                # For CHAN nodes, include all intermediate tiles
                if node_type in ['CHANX', 'CHANY']:
                    wire_tiles = traverse_wire_nodes(start_coords, end_coords, node_id)
                    for tile in wire_tiles:
                        tile_nodes[tile].add((node_id, node_type))

    print(f"Total tiles identified: {len(tile_nodes)}")
    return tile_nodes


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
        
 #       print("\nEdges:")
 #       for u, v, attrs in net_graph.edges(data=True):
 #           print(f"Edge: {u} -> {v}")
 #           for key, value in attrs.items():
 #               print(f"  {key}: {value}")
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

    # Use defaultdict to initialize counts to 0 for new coordinates
    ipin_counts = defaultdict(int)
    opin_counts = defaultdict(int)
    block_types = defaultdict(str)
    coord=[]

    PIN_COUNTS = {
        'CLB': {'IPIN': 72, 'OPIN': 32},
        'BRAM': {'IPIN': 43, 'OPIN': 33},
        'DSP': {'IPIN': 132, 'OPIN': 81},
        'IO': {'IPIN': 128, 'OPIN': 64},
        'UNKNOWN': {'IPIN': 1, 'OPIN': 1},  # Default to avoid division by zero
    }

    # Iterate through all nets 
    for net_id, graph in route_tree.items():
        if not isinstance(graph, nx.DiGraph):
            print(f"Invalid graph for net_id: {net_id}, type: {type(graph)}")
            continue

        for _, attrs in graph.nodes(data=True):
            coords = attrs['start_coords']  # Coordinates of the tile
            node_type = attrs['node_type']
            block_type = attrs['block_type'] # Default to 'UNKNOWN'
            
            # Store the block type for the tile

            # Increment counts for IPIN or OPIN
            if node_type == 'IPIN': 
                block_types[coords] = block_type
                coord.append(coords)
                ipin_counts[coords] += 1
            elif node_type == 'OPIN': 
                block_types[coords] = block_type
                coord.append(coords)
                opin_counts[coords] += 1

    # Calculate densities
    pin_density = {}
    for c in coord:
        block_type = block_types[c]
        if block_type not in PIN_COUNTS:
            print(f"Warning: Unknown block type at {c}. Skipping...")
            continue
        
        total_ipins = PIN_COUNTS[block_type]['IPIN']
        total_opins = PIN_COUNTS[block_type]['OPIN']

        # Compute densities
        ipin_density = ipin_counts[c] / total_ipins
        opin_density = opin_counts[c] / total_opins

        # Store densities
        pin_density[c] = (ipin_density, opin_density)

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


def calculate_sum_tile_costs(tile_nodes, hcost_data, relevant_iterations):
    """
    Calculate the summed CHANX and CHANY costs for each tile using history cost data.

    Parameters:
    -----------
    tile_nodes : dict
        Dictionary mapping tile coordinates to sets of (node_id, node_type) tuples.
    hcost_data : dict
        Dictionary of history cost data for each iteration: {iteration_key: {node_id: cost, ...}, ...}.
    relevant_iterations : list
        List of relevant iterations to calculate costs for.

    Returns:
    --------
    dict
        Dictionary with tile coordinates as keys and summed costs {'CHANX', 'CHANY'} as values.
    """
    print("Calculating average costs per tile...")

    # Initialize the cost dictionary for tiles
    tile_costs = {coords: {'CHANX': [], 'CHANY': []} for coords in tile_nodes.keys()}

    # Step 1: Aggregate costs per tile
    for iteration_key in relevant_iterations:
        if iteration_key not in hcost_data:
            print(f"Warning: Missing cost data for {iteration_key}")
            continue

        history_costs = hcost_data[iteration_key]
        for tile_coords, nodes in tile_nodes.items():
            for node in nodes:
                node = list(node)  
                node_id = node[0]
                node_type = node[1]

                if node_id in history_costs:
                    cost = history_costs[node_id]
                    if node_type == "CHANX":
                        tile_costs[tile_coords]["CHANX"].append(cost)
                    elif node_type == "CHANY":
                        tile_costs[tile_coords]["CHANY"].append(cost)

    # Step 2: Calculate the average cost for each tile
    avg_costs = {}
    for coords, costs in tile_costs.items():
        avg_chanx = sum(costs["CHANX"])  if costs["CHANX"] else 1
        avg_chany = sum(costs["CHANY"])  if costs["CHANY"] else 1

        avg_costs[coords] = {
            "CHANX": avg_chanx,
            "CHANY": avg_chany
        }

    print("Average costs per tile calculated.")
    return avg_costs



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

def generate_feature_set(route_data, hcost_data, total_iterations, target_percentages=[1.0]):
    """
    Generate the feature set for each tile and targets for multiple iteration percentages.

    Parameters:
    -----------
    route_data : dict
        Parsed route data for the benchmark.
    hcost_data : dict
        Dictionary of history cost data for each iteration (keyed by iteration ID).
    total_iterations : int
        Total number of iterations available for the benchmark.
    target_percentages : list of float
        List of percentages of total iterations to use as targets.

    Returns:
    --------
    pd.DataFrame, dict of pd.DataFrame
        Features and a dictionary of targets for each percentage.
    """
    print("\n--- Generating Feature Set ---\n")

    # Step 1: Extract tile coordinates
    print("Extracting tile coordinates...")
    tile_nodes = get_tile_coordinates(route_data)

    # Step 2: Identify relevant iterations
    print("Identifying relevant iterations...")
    relevant_iterations = ['iteration_001', 'iteration_002', 'iteration_003']
    for percentage in target_percentages:
        iteration_key = f'iteration_{int(total_iterations * percentage):03d}'
        if iteration_key not in relevant_iterations:
            relevant_iterations.append(iteration_key)
    print(f"Relevant iterations: {relevant_iterations}\n")

    # Step 3: Calculate tile costs
    print("Calculating tile costs...")
    tile_costs = calculate_sum_tile_costs(tile_nodes, hcost_data, relevant_iterations)

    # Step 4: Calculate pin and channel features
    print("Flattening route trees...")
    all_route_trees = {net_id: graph for nets in route_data.values() for net_id, graph in nets.items()}
    pin_density = calculate_pin_density(all_route_trees)
    channel_usage = calculate_channel_usage(all_route_trees)

    # Step 5: Compute features
    print("Computing features...")
    feature_set = []
    for coords in tile_nodes.keys():
        features = {
            'Input Pin Density': pin_density.get(coords, (0, 0))[0],
            'Output Pin Density': pin_density.get(coords, (0, 0))[1],
            'Local Tile Density Usage': calculate_tile_density(channel_usage, coords),
            'Global Tile Density Usage': calculate_tile_density(channel_usage, coords, 5),
            'Local Channel Bias': calculate_channel_bias_for_tile(channel_usage, coords),
            'Global Channel Bias': calculate_channel_bias_for_tile(channel_usage, coords, 5),
        }
        # Include costs for first three iterations
        for iteration in relevant_iterations:
            features[f'Cost CHANX {iteration}'] = tile_costs[coords]['CHANX']
            features[f'Cost CHANY {iteration}'] = tile_costs[coords]['CHANY']

        feature_set.append(features)

    feature_set_df = pd.DataFrame(feature_set)

    # Step 6: Extract targets for each percentage
    print("Extracting and renaming targets for all percentages...")
    targets_dict = {}
    for percentage in target_percentages:
        iteration_suffix = f'iteration_{int(total_iterations * percentage):03d}'
        target_columns = [
            f'Cost CHANX {iteration_suffix}',
            f'Cost CHANY {iteration_suffix}'
        ]
        renamed_columns = {
            f'Cost CHANX {iteration_suffix}': f'Cost CHANX {int(percentage * 100)}%',
            f'Cost CHANY {iteration_suffix}': f'Cost CHANY {int(percentage * 100)}%'
        }
        if all(col in feature_set_df.columns for col in target_columns):
            targets_dict[percentage] = feature_set_df[target_columns].rename(columns=renamed_columns)
        else:
            raise ValueError(f"Missing target columns for percentage {percentage * 100}%.")

    # Step 7: Define features using first three iterations and other columns
    print("Finalizing feature columns...")
    feature_columns = [col for col in feature_set_df.columns 
                       if not any(col.startswith(f"Cost CHANX iteration_") or col.startswith(f"Cost CHANY iteration_") 
                                  for percentage in target_percentages)]
    feature_columns.extend([
        'Cost CHANX iteration_001', 'Cost CHANY iteration_001',
        'Cost CHANX iteration_002', 'Cost CHANY iteration_002',
        'Cost CHANX iteration_003', 'Cost CHANY iteration_003'
    ])

    features = feature_set_df[feature_columns]

    print(f"Final features shape: {features.shape}")
    print(f"Targets prepared for percentages: {list(targets_dict.keys())}\n")
    return features, targets_dict

#####################################################################################

#####################################################################################
