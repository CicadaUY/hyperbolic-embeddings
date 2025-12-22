"""
General Hyperbolic KNN Classification Script

This script implements K-Nearest Neighbors classification using hyperbolic distance
on various datasets including PoincareMaps datasets, OGB datasets (ogbn-arxiv), etc.
"""

import argparse
import json
import os
import pickle
import random
from typing import Tuple

import networkx as nx
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from hyperbolic_embeddings import HyperbolicEmbeddings
from neuroseed_loader import load_neuroseed_dataset
from poincare_maps_networkx_loader import PoincareMapsLoader
from utils.geometric_conversions import HyperbolicConversions

# Patch torch.load for PyTorch 2.6+ compatibility with OGB datasets
# This needs to happen at module level before OGB imports torch.load
try:
    import torch

    # Store original torch.load
    _original_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        """Patched torch.load that defaults to weights_only=False for OGB compatibility."""
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)

    # Apply patch
    torch.load = _patched_torch_load

    # Add safe globals for torch_geometric classes if available
    try:
        from torch_geometric.data import Data
        from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, Data])
    except (ImportError, AttributeError):
        pass

except ImportError:
    # torch not available, will fail later if OGB is used
    pass


def hyperbolic_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Poincaré distance between two points using the function from geometric_conversions.py.

    This wrapper function is compatible with sklearn's metric interface.

    Parameters:
    -----------
    x : np.ndarray
        First point in Poincaré disk (1D array)
    y : np.ndarray
        Second point in Poincaré disk (1D array)

    Returns:
    --------
    float
        Hyperbolic distance between x and y
    """
    # Use compute_distances from geometric_conversions.py
    # Stack the two points into a 2-row array
    points = np.vstack([x, y])
    distance_matrix = HyperbolicConversions.compute_distances(points, space="poincare")
    # Extract the distance between the two points (off-diagonal element)
    return float(distance_matrix[0, 1])


def extract_labels_from_graph(graph: nx.Graph) -> np.ndarray:
    """
    Extract labels from graph node attributes and convert to numeric.

    Parameters:
    -----------
    graph : nx.Graph
        NetworkX graph with node labels

    Returns:
    --------
    np.ndarray
        Numeric labels array
    """
    labels = [graph.nodes[n].get("label", None) for n in graph.nodes()]
    if all(label is None for label in labels):
        raise ValueError("Graph does not contain node labels")
    # Convert string labels to numeric using LabelEncoder
    encoder = LabelEncoder()
    numeric_labels = encoder.fit_transform(labels)
    return numeric_labels


def create_subgraph(
    graph: nx.Graph,
    labels: np.ndarray,
    node_indices: np.ndarray,
    target_nodes: int = 1000,
    random_seed: int = None,
    min_nodes_per_class: int = 5,
    max_classes: int = None,
) -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
    """
    Create a class-aware connected subgraph with approximately target_nodes nodes.

    This function ensures that each class has at least min_nodes_per_class nodes,
    then fills up to target_nodes using BFS while trying to maintain class balance.
    If max_classes is specified, only nodes from the top max_classes (by count) are included.

    Parameters:
    -----------
    graph : nx.Graph
        NetworkX graph to sample from
    labels : np.ndarray
        Original labels array
    node_indices : np.ndarray
        Original node indices array
    target_nodes : int
        Target number of nodes (default: 1000)
    random_seed : int
        Random seed for reproducibility (default: None)
    min_nodes_per_class : int
        Minimum number of nodes per class (default: 5)
    max_classes : int
        Maximum number of classes to include (default: None, includes all classes)

    Returns:
    --------
    subgraph : nx.Graph
        NetworkX Graph subgraph with up to target_nodes nodes
    subgraph_labels : np.ndarray
        Labels for nodes in the subgraph
    subgraph_node_indices : np.ndarray
        Node indices for nodes in the subgraph
    """
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    num_nodes = len(graph.nodes())
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)

    # Step 0: Filter to max_classes if specified
    if max_classes is not None and max_classes < num_classes:
        print(f"Filtering to top {max_classes} classes from {num_classes} classes...")
        # Count nodes per class
        class_counts_dict = {cls: np.sum(labels == cls) for cls in unique_classes}
        # Sort classes by count (descending) and take top max_classes
        sorted_classes = sorted(class_counts_dict.items(), key=lambda x: x[1], reverse=True)
        selected_classes = [cls for cls, count in sorted_classes[:max_classes]]
        selected_classes_set = set(selected_classes)

        # Filter graph, labels, and node_indices to only include selected classes
        nodes_to_keep = [node for node in graph.nodes() if labels[node] in selected_classes_set]
        nodes_to_keep_sorted = sorted(nodes_to_keep)  # Sort for consistent mapping

        # Create mapping from old node indices to new consecutive indices
        node_mapping = {old_node: new_idx for new_idx, old_node in enumerate(nodes_to_keep_sorted)}

        # Create new graph with remapped node indices (0-based consecutive)
        graph_filtered = nx.Graph()
        for old_node in nodes_to_keep_sorted:
            new_node = node_mapping[old_node]
            graph_filtered.add_node(new_node)

        # Add edges with remapped node indices
        for old_u, old_v in graph.edges():
            if old_u in node_mapping and old_v in node_mapping:
                new_u = node_mapping[old_u]
                new_v = node_mapping[old_v]
                graph_filtered.add_edge(new_u, new_v)

        graph = graph_filtered

        # Reindex labels to be 0, 1, 2, 3 for the 4 classes
        label_mapping = {old_cls: new_cls for new_cls, old_cls in enumerate(selected_classes)}
        # Create new arrays with only selected nodes, maintaining order
        filtered_labels = []
        filtered_node_indices = []
        for old_node in nodes_to_keep_sorted:
            filtered_labels.append(label_mapping[labels[old_node]])
            filtered_node_indices.append(node_indices[old_node])
        labels = np.array(filtered_labels)
        node_indices = np.array(filtered_node_indices)

        # Update unique_classes
        unique_classes = np.unique(labels)
        num_classes = len(unique_classes)
        num_nodes = len(graph.nodes())

        # Verify node indices are 0-based consecutive
        graph_nodes = sorted(graph.nodes())
        expected_nodes = list(range(num_nodes))
        if graph_nodes != expected_nodes:
            print("Warning: Graph nodes are not 0-based consecutive. Remapping...")
            # Create a new graph with proper 0-based consecutive indices
            graph_remapped = nx.Graph()
            node_remap = {old_node: new_idx for new_idx, old_node in enumerate(graph_nodes)}
            for old_node in graph_nodes:
                graph_remapped.add_node(node_remap[old_node])
            for old_u, old_v in graph.edges():
                if old_u in node_remap and old_v in node_remap:
                    graph_remapped.add_edge(node_remap[old_u], node_remap[old_v])
            graph = graph_remapped

        print(f"After filtering: {num_nodes} nodes, {num_classes} classes")
        print(f"Selected classes (original labels): {selected_classes}")
        print(f"Class distribution: {np.bincount(labels)}")

        # Final verification: ensure labels array size matches graph size
        if len(labels) != num_nodes:
            raise ValueError(f"Labels array size ({len(labels)}) doesn't match graph size ({num_nodes})")

    # Calculate minimum required nodes
    min_required_nodes = num_classes * min_nodes_per_class

    # If target is too small, adjust it
    if target_nodes < min_required_nodes:
        print(f"Warning: target_nodes ({target_nodes}) is less than minimum required ({min_required_nodes}).")
        print(f"Adjusting target_nodes to {min_required_nodes}.")
        target_nodes = min_required_nodes

    # If graph is smaller than target, return entire graph
    if num_nodes <= target_nodes:
        return graph.copy(), labels.copy(), node_indices.copy()

    print(f"Creating class-aware subgraph with {target_nodes} nodes from {num_nodes} nodes...")
    print(f"Ensuring at least {min_nodes_per_class} nodes per class ({num_classes} classes)...")

    # Step 1: For each class, ensure we have at least min_nodes_per_class nodes
    visited = set()
    class_counts = {cls: 0 for cls in unique_classes}

    # Get nodes by class
    nodes_by_class = {cls: [] for cls in unique_classes}
    for node in graph.nodes():
        nodes_by_class[labels[node]].append(node)

    # First, ensure minimum nodes per class
    for cls in unique_classes:
        class_nodes = nodes_by_class[cls]
        if len(class_nodes) < min_nodes_per_class:
            print(f"Warning: Class {cls} has only {len(class_nodes)} nodes, less than minimum {min_nodes_per_class}.")
            # Add all nodes from this class
            for node in class_nodes:
                visited.add(node)
                class_counts[cls] += 1
        else:
            # Randomly select min_nodes_per_class nodes from this class
            selected = random.sample(class_nodes, min_nodes_per_class)
            for node in selected:
                visited.add(node)
                class_counts[cls] += 1

    print(f"After ensuring minimum per class: {len(visited)} nodes selected")
    print(f"Class distribution: {class_counts}")

    # Step 2: Fill up to target_nodes using BFS, prioritizing underrepresented classes
    if len(visited) < target_nodes:
        print(f"Filling subgraph from {len(visited)} to {target_nodes} nodes using BFS...")
        # Create a queue starting from visited nodes
        queue = list(visited)
        iterations_without_progress = 0
        max_iterations_without_progress = 1000

        # Continue BFS until we have enough nodes
        while queue and len(visited) < target_nodes:
            current = queue.pop(0)
            previous_visited_size = len(visited)

            # Get neighbors and sort by class representation (prioritize underrepresented classes)
            neighbors = list(graph.neighbors(current))
            # Calculate class proportions for prioritization
            if len(visited) > 0:
                class_props = {cls: class_counts[cls] / len(visited) for cls in unique_classes}
                target_prop = 1.0 / num_classes
                # Sort neighbors: underrepresented classes first, then others
                neighbors.sort(
                    key=lambda n: (
                        class_props.get(labels[n], 0) < target_prop,  # True (underrepresented) comes first
                        -class_props.get(labels[n], 0),  # Then by how underrepresented (more negative = more underrepresented)
                    ),
                    reverse=True,
                )
            else:
                random.shuffle(neighbors)  # Randomize if no visited nodes yet

            for neighbor in neighbors:
                if neighbor not in visited and len(visited) < target_nodes:
                    # Always add nodes when below target
                    neighbor_class = labels[neighbor]
                    visited.add(neighbor)
                    class_counts[neighbor_class] += 1
                    queue.append(neighbor)

            # Check if we made progress
            if len(visited) == previous_visited_size:
                iterations_without_progress += 1
                if iterations_without_progress >= max_iterations_without_progress:
                    print(f"Warning: BFS stopped making progress. Reached {len(visited)} nodes (target: {target_nodes})")
                    # Try to add random nodes from underrepresented classes
                    if len(visited) < target_nodes:
                        all_nodes = set(graph.nodes())
                        remaining_nodes = all_nodes - visited
                        if remaining_nodes:
                            # Add nodes from underrepresented classes first
                            for cls in unique_classes:
                                if len(visited) >= target_nodes:
                                    break
                                class_prop = class_counts[cls] / max(len(visited), 1)
                                target_prop = 1.0 / num_classes
                                if class_prop < target_prop:
                                    class_candidates = [n for n in remaining_nodes if labels[n] == cls]
                                    if class_candidates:
                                        node_to_add = random.choice(class_candidates)
                                        visited.add(node_to_add)
                                        class_counts[cls] += 1
                                        remaining_nodes.remove(node_to_add)
                            # Fill remaining with any available nodes
                            while len(visited) < target_nodes and remaining_nodes:
                                node_to_add = random.choice(list(remaining_nodes))
                                visited.add(node_to_add)
                                class_counts[labels[node_to_add]] += 1
                                remaining_nodes.remove(node_to_add)
                    break
            else:
                iterations_without_progress = 0

        print(f"After BFS filling: {len(visited)} nodes selected (target: {target_nodes})")

    # Create subgraph from collected nodes
    subgraph = graph.subgraph(visited).copy()

    # Filter labels and node_indices to match subgraph nodes
    visited_list = sorted(visited)  # Sort for consistent indexing
    subgraph_labels = labels[visited_list]
    subgraph_node_indices = node_indices[visited_list]

    print(f"Subgraph created: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
    print(f"Final class distribution: {class_counts}")
    print(f"Classes in subgraph: {len(np.unique(subgraph_labels))}")

    # Verify all classes have at least min_nodes_per_class
    final_class_counts = {cls: np.sum(subgraph_labels == cls) for cls in unique_classes}
    for cls, count in final_class_counts.items():
        if count < min_nodes_per_class:
            print(f"Warning: Class {cls} has only {count} nodes in subgraph (minimum: {min_nodes_per_class})")

    return subgraph, subgraph_labels, subgraph_node_indices


def load_ogb_dataset(
    dataset_name: str = "ogbn-arxiv",
    subgraph_size: int = None,
    random_seed: int = None,
    min_nodes_per_class: int = 5,
    max_classes: int = None,
) -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
    """
    Load OGB dataset and convert to NetworkX format.

    Parameters:
    -----------
    dataset_name : str
        Name of the OGB dataset (e.g., "ogbn-arxiv")

    Returns:
    --------
    graph : nx.Graph
        NetworkX graph representation (undirected)
    labels : np.ndarray
        Node labels (numeric)
    node_indices : np.ndarray
        Array of node indices
    """
    print(f"Loading OGB dataset: {dataset_name}...")

    try:
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError:
        raise ImportError("ogb package is required. Install it with: pip install ogb")

    dataset = PygNodePropPredDataset(name=dataset_name)
    graph_pyg = dataset[0]  # PyTorch Geometric graph

    print(f"Dataset loaded: {graph_pyg.num_nodes} nodes, {graph_pyg.num_edges} edges")

    # Convert to NetworkX (undirected)
    graph = nx.Graph()
    edge_index = graph_pyg.edge_index.numpy()
    edges = [(int(edge_index[0, i]), int(edge_index[1, i])) for i in range(edge_index.shape[1])]
    graph.add_edges_from(edges)
    graph.add_nodes_from(range(graph_pyg.num_nodes))

    # Extract labels (already numeric)
    labels = graph_pyg.y.numpy().flatten()  # Flatten from (N, 1) to (N,)
    node_indices = np.arange(graph_pyg.num_nodes)

    print(f"Graph created: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"Number of classes: {len(np.unique(labels))}")
    print(f"Label distribution: {np.bincount(labels)}")

    # Create subgraph if requested
    if subgraph_size is not None and subgraph_size > 0:
        graph, labels, node_indices = create_subgraph(
            graph,
            labels,
            node_indices,
            target_nodes=subgraph_size,
            random_seed=random_seed,
            min_nodes_per_class=min_nodes_per_class,
            max_classes=max_classes,
        )
        print(f"After subgraph creation: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        print(f"Number of classes: {len(np.unique(labels))}")
        print(f"Label distribution: {np.bincount(labels)}")

    return graph, labels, node_indices


def load_polblogs_dataset(
    subgraph_size: int = None,
    random_seed: int = None,
    min_nodes_per_class: int = 5,
    max_classes: int = None,
) -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
    """
    Load PolBlogs dataset from PyTorch Geometric and convert to NetworkX format.

    Parameters:
    -----------
    subgraph_size : int
        If specified, create a connected subgraph with this many nodes
    random_seed : int
        Random seed for reproducibility
    min_nodes_per_class : int
        Minimum number of nodes per class when creating subgraph
    max_classes : int
        Maximum number of classes to include in subgraph

    Returns:
    --------
    graph : nx.Graph
        NetworkX graph representation
    labels : np.ndarray
        Node labels (numeric)
    node_indices : np.ndarray
        Array of node indices
    """
    print("Loading PolBlogs dataset...")

    try:
        from torch_geometric.datasets import PolBlogs
    except ImportError:
        raise ImportError("torch_geometric package is required. Install it with: pip install torch-geometric")

    # Load dataset (it will be downloaded if not present)
    dataset = PolBlogs(root="./data/polblogs")
    data = dataset[0]

    print(f"Dataset loaded: {data.num_nodes} nodes, {data.num_edges} edges")
    print(f"Number of classes: {dataset.num_classes}")

    # Convert to NetworkX graph
    edge_index = data.edge_index.numpy()
    edges = [(int(edge_index[0, i]), int(edge_index[1, i])) for i in range(edge_index.shape[1])]

    graph = nx.Graph()
    graph.add_edges_from(edges)
    graph.add_nodes_from(range(data.num_nodes))

    # Get labels
    labels = data.y.numpy()

    # Get node indices
    node_indices = np.arange(data.num_nodes)

    print(f"Graph created: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"Number of classes: {len(np.unique(labels))}")
    print(f"Label distribution: {np.bincount(labels)}")

    # Extract largest connected component (needed for some embedding methods like dmercator)
    if not nx.is_connected(graph):
        print("\nGraph has multiple components. Extracting largest connected component...")
        # Get the largest connected component
        largest_cc = max(nx.connected_components(graph), key=len)

        # Create subgraph with only the largest component
        graph = graph.subgraph(largest_cc).copy()

        # Create mapping from old to new node indices
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(largest_cc))}

        # Relabel nodes to be sequential starting from 0
        graph = nx.relabel_nodes(graph, old_to_new)

        # Filter labels and node indices to match the largest component
        labels = labels[sorted(largest_cc)]
        node_indices = np.arange(len(largest_cc))

        print(
            f"Largest component: {graph.number_of_nodes()} nodes ({100*len(largest_cc)/data.num_nodes:.1f}% of original), {graph.number_of_edges()} edges"
        )
        print(f"Number of classes: {len(np.unique(labels))}")
        print(f"Label distribution: {np.bincount(labels)}")

    # Create subgraph if requested
    if subgraph_size is not None and subgraph_size > 0:
        graph, labels, node_indices = create_subgraph(
            graph,
            labels,
            node_indices,
            target_nodes=subgraph_size,
            random_seed=random_seed,
            min_nodes_per_class=min_nodes_per_class,
            max_classes=max_classes,
        )
        print(f"After subgraph creation: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        print(f"Number of classes: {len(np.unique(labels))}")
        print(f"Label distribution: {np.bincount(labels)}")

    return graph, labels, node_indices


def load_cora_dataset(
    subgraph_size: int = None,
    random_seed: int = None,
    min_nodes_per_class: int = 5,
    max_classes: int = None,
) -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
    """
    Load CORA dataset and convert to NetworkX format.

    Parameters:
    -----------
    subgraph_size : int
        If specified, create a connected subgraph with this many nodes
    random_seed : int
        Random seed for reproducibility
    min_nodes_per_class : int
        Minimum number of nodes per class when creating subgraph
    max_classes : int
        Maximum number of classes to include in subgraph

    Returns:
    --------
    graph : nx.Graph
        NetworkX graph representation
    labels : np.ndarray
        Node labels (numeric)
    node_indices : np.ndarray
        Array of node indices
    """
    print("Loading CORA dataset...")

    with open("./data/Cora/cora_graph.pkl", "rb") as f:
        edge_list = pickle.load(f)
    with open("./data/Cora/cora_graph.json", "r") as f:
        graph_data = json.load(f)

    # Build networkx graph from edge list
    graph = nx.Graph()
    graph.add_edges_from(edge_list)

    # Get labels
    labels = np.array(graph_data.get("y", []))

    # Get node indices
    node_indices = np.arange(len(graph.nodes()))

    print(f"Graph created: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"Number of classes: {len(np.unique(labels))}")
    print(f"Label distribution: {np.bincount(labels)}")

    # Create subgraph if requested
    if subgraph_size is not None and subgraph_size > 0:
        graph, labels, node_indices = create_subgraph(
            graph,
            labels,
            node_indices,
            target_nodes=subgraph_size,
            random_seed=random_seed,
            min_nodes_per_class=min_nodes_per_class,
            max_classes=max_classes,
        )
        print(f"After subgraph creation: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        print(f"Number of classes: {len(np.unique(labels))}")
        print(f"Label distribution: {np.bincount(labels)}")

    return graph, labels, node_indices


def load_dataset(
    dataset_name: str,
    dataset_type: str,
    k_neighbors: int = 15,
    datasets_path: str = None,
    subgraph_size: int = None,
    random_seed: int = None,
    min_nodes_per_class: int = 5,
    max_classes: int = None,
    neuroseed_task: str = "edit_distance",
    neuroseed_split: str = "train",
) -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
    """
    Load dataset based on type and return graph, labels, and node indices.

    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
    dataset_type : str
        Type of dataset ("AS", "ogbn-arxiv", "neuroseed", or PoincareMaps dataset name)
    k_neighbors : int
        Number of neighbors for PoincareMaps/NeuroSEED KNN graph construction
    datasets_path : str
        Path to PoincareMaps datasets directory
    neuroseed_task : str
        Task name for NeuroSEED dataset (e.g., "edit_distance", "closest_string")
    neuroseed_split : str
        Split for NeuroSEED dataset ("train", "val", "test")

    Returns:
    --------
    graph : nx.Graph
        NetworkX graph representation
    labels : np.ndarray
        Node labels (numeric)
    node_indices : np.ndarray
        Array of node indices
    """
    if dataset_type == "AS":
        raise ValueError("AS dataset does not contain node labels for classification")
    elif dataset_type == "ogbn-arxiv":
        return load_ogb_dataset(
            "ogbn-arxiv",
            subgraph_size=subgraph_size,
            random_seed=random_seed,
            min_nodes_per_class=min_nodes_per_class,
            max_classes=max_classes,
        )
    elif dataset_type == "polblogs":
        return load_polblogs_dataset(
            subgraph_size=subgraph_size,
            random_seed=random_seed,
            min_nodes_per_class=min_nodes_per_class,
            max_classes=max_classes,
        )
    elif dataset_type == "Cora":
        return load_cora_dataset(
            subgraph_size=subgraph_size,
            random_seed=random_seed,
            min_nodes_per_class=min_nodes_per_class,
            max_classes=max_classes,
        )
    elif dataset_type == "neuroseed":
        # For NeuroSEED, we DON'T use load_neuroseed_dataset here
        # because we need to handle train/test splits properly in main()
        # This is a placeholder - actual loading happens in main()
        return None, None, None
    else:
        # PoincareMaps dataset
        if datasets_path is None:
            datasets_path = "models/PoincareMaps/datasets/"

        print(f"Loading PoincareMaps dataset: {dataset_name}...")
        loader = PoincareMapsLoader(datasets_path)
        graph, metadata = loader.load_as_networkx(dataset_name, k_neighbors=k_neighbors)
        labels = extract_labels_from_graph(graph)
        node_indices = np.arange(len(graph.nodes()))

        print(f"Graph created: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        print(f"Number of classes: {len(np.unique(labels))}")
        print(f"Label distribution: {np.bincount(labels)}")

        return graph, labels, node_indices


def train_hyperbolic_embeddings(
    graph: nx.Graph,
    embedding_type: str = "poincare_embeddings",
    model_dir: str = "saved_models/default",
    dim: int = 2,
) -> Tuple[np.ndarray, str]:
    """
    Train hyperbolic embeddings for the graph with dataset size-based configurations.

    Parameters:
    -----------
    graph : nx.Graph
        Input graph
    embedding_type : str
        Type of embedding model to use
    model_dir : str
        Directory to save the model
    dim : int
        Embedding dimension

    Returns:
    --------
    embeddings : np.ndarray
        Node embeddings
    embedding_space : str
        Native embedding space of the model
    """
    print(f"\nTraining {embedding_type} embeddings...")

    # Ensure graph nodes are 0-based consecutive (critical for adjacency matrix creation)
    graph_nodes = sorted(graph.nodes())
    expected_nodes = list(range(len(graph_nodes)))
    if graph_nodes != expected_nodes:
        print("Remapping graph nodes to 0-based consecutive indices...")
        # Create node mapping
        node_mapping = {old_node: new_idx for new_idx, old_node in enumerate(graph_nodes)}
        # Create new graph with remapped nodes
        graph_remapped = nx.Graph()
        for old_node in graph_nodes:
            graph_remapped.add_node(node_mapping[old_node])
        for old_u, old_v in graph.edges():
            if old_u in node_mapping and old_v in node_mapping:
                graph_remapped.add_edge(node_mapping[old_u], node_mapping[old_v])
        graph = graph_remapped
        print(f"Graph remapped: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Prepare edge list
    edge_list = list(graph.edges())
    num_nodes = graph.number_of_nodes()

    # Determine dataset size category
    is_large_dataset = num_nodes >= 100000  # ogbn-arxiv scale
    is_medium_dataset = num_nodes >= 1000

    # Embedding configurations based on dataset size
    if is_large_dataset:
        # Large datasets (>= 100K nodes, e.g., ogbn-arxiv)
        configurations = {
            "poincare_embeddings": {"dim": dim, "negs": 10, "epochs": 500, "batch_size": 512, "dimension": 1},
            "lorentz": {"dim": dim, "epochs": 5000, "batch_size": 2048, "num_nodes": num_nodes},
            "dmercator": {"dim": dim},
            "hydra": {"dim": 3},
            "poincare_maps": {"dim": dim, "epochs": 500},
            "hypermap": {"dim": 3},
            "hydra_plus": {"dim": 3},
        }
    elif is_medium_dataset:
        # Medium datasets (1K-100K nodes)
        configurations = {
            "poincare_embeddings": {"dim": dim, "negs": 5, "epochs": 1000, "batch_size": 256, "dimension": 1},
            "lorentz": {"dim": dim, "epochs": 10000, "batch_size": 1024, "num_nodes": num_nodes},
            "dmercator": {"dim": dim},
            "hydra": {"dim": 2},
            "poincare_maps": {"dim": dim, "epochs": 1000},
            "hypermap": {"dim": 3},
            "hydra_plus": {"dim": 2},
        }
    else:
        # Small datasets (< 1K nodes)
        configurations = {
            "poincare_embeddings": {"dim": dim, "negs": 5, "epochs": 1000, "batch_size": 256, "dimension": 1},
            "lorentz": {"dim": dim, "epochs": 10000, "batch_size": 1024, "num_nodes": num_nodes},
            "dmercator": {"dim": dim},
            "hydra": {"dim": 2},
            "poincare_maps": {"dim": dim, "epochs": 1000},
            "hypermap": {"dim": 3},
            "hydra_plus": {"dim": 2},
        }

    config = configurations.get(embedding_type, configurations["poincare_embeddings"])

    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{embedding_type}_embeddings.bin")

    # Initialize and train embeddings
    embedding_runner = HyperbolicEmbeddings(embedding_type=embedding_type, config=config)

    # Get adjacency matrix for models that need it
    if embedding_type in ["hydra", "poincare_maps", "lorentz", "hydra_plus"]:
        A = nx.to_numpy_array(graph)
        embedding_runner.train(adjacency_matrix=A, model_path=model_path)
    else:
        embedding_runner.train(edge_list=edge_list, model_path=model_path)

    # Get embeddings
    embeddings = embedding_runner.get_all_embeddings(model_path)
    embedding_space = embedding_runner.model.native_space

    print(f"Embeddings trained: shape {embeddings.shape}, space: {embedding_space}")

    # Convert embeddings to Poincaré if needed
    if embedding_space != "poincare":
        print(f"Converting embeddings from {embedding_space} to Poincaré...")
        embeddings = HyperbolicConversions.convert_coordinates(embeddings, embedding_space, "poincare")
        embedding_space = "poincare"

    return embeddings, embedding_space


def evaluate_knn(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    k_values: list = [3, 5, 7, 10],
    n_iterations: int = 1,
) -> dict:
    """
    Evaluate KNN classification with different k values using hyperbolic distance.

    Parameters:
    -----------
    X_train : np.ndarray
        Training embeddings
    X_test : np.ndarray
        Test embeddings
    y_train : np.ndarray
        Training labels
    y_test : np.ndarray
        Test labels
    k_values : list
        List of k values to test
    n_iterations : int
        Number of iterations to run for computing mean and std (default: 1)

    Returns:
    --------
    results : dict
        Dictionary containing results for each k value with mean and std
    """
    results = {}

    for k in k_values:
        print(f"\nEvaluating KNN with k={k}...")

        # Store metrics for all iterations
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []

        for iteration in range(n_iterations):
            if n_iterations > 1:
                print(f"  Iteration {iteration + 1}/{n_iterations}...", end="\r")

            # Create KNN classifier with hyperbolic distance
            knn = KNeighborsClassifier(n_neighbors=k, metric=hyperbolic_distance, algorithm="brute")

            # Fit and predict
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)

            # Compute metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        if n_iterations > 1:
            print(f"  Completed {n_iterations} iterations")

        # Calculate mean and std
        results[k] = {
            "accuracy_mean": np.mean(accuracies),
            "accuracy_std": np.std(accuracies),
            "precision_mean": np.mean(precisions),
            "precision_std": np.std(precisions),
            "recall_mean": np.mean(recalls),
            "recall_std": np.std(recalls),
            "f1_mean": np.mean(f1_scores),
            "f1_std": np.std(f1_scores),
            "accuracies": accuracies,
            "precisions": precisions,
            "recalls": recalls,
            "f1_scores": f1_scores,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="General Hyperbolic KNN Classification")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ToggleSwitch",
        choices=[
            "AS",
            "ToggleSwitch",
            "Olsson",
            "MyeloidProgenitors",
            "krumsiek11_blobs",
            "Paul",
            "ogbn-arxiv",
            "polblogs",
            "Cora",
            "neuroseed",
        ],
        help="Dataset to use",
    )
    parser.add_argument(
        "--embedding_type",
        type=str,
        default="hydra_plus",
        choices=["poincare_embeddings", "lorentz", "dmercator", "hydra", "poincare_maps", "hypermap", "hydra_plus"],
        help="Type of embedding model to use.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=2,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Directory to save the trained model (default: saved_models/{dataset})",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing.",
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[3, 5, 7, 10],
        help="List of k values to test for KNN.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for train/test split.",
    )
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=15,
        help="Number of neighbors for PoincareMaps KNN graph construction (ignored for OGB datasets).",
    )
    parser.add_argument(
        "--datasets_path",
        type=str,
        default="models/PoincareMaps/datasets/",
        help="Path to PoincareMaps datasets directory.",
    )
    parser.add_argument(
        "--subgraph_size",
        type=int,
        default=None,
        help="If specified, create a connected subgraph with this many nodes (useful for large datasets like ogbn-arxiv).",
    )
    parser.add_argument(
        "--min_nodes_per_class",
        type=int,
        default=5,
        help="Minimum number of nodes per class when creating subgraph (default: 5).",
    )
    parser.add_argument(
        "--max_classes",
        type=int,
        default=4,
        help="Maximum number of classes to include in subgraph (default: 4). Set to None to include all classes.",
    )
    parser.add_argument(
        "--neuroseed_task",
        type=str,
        default="edit_distance",
        choices=["edit_distance", "closest_string", "hierarchical_clustering", "multiple_alignment"],
        help="Task name for NeuroSEED dataset (only used when dataset=neuroseed).",
    )
    parser.add_argument(
        "--use_predefined_splits",
        action="store_true",
        help="Use pre-defined train/test splits for NeuroSEED (requires real data). If not set, uses synthetic data with random split.",
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=10,
        help="Number of iterations to run for computing mean and std of metrics (default: 10).",
    )

    args = parser.parse_args()

    # Set default model directory if not provided
    if args.model_dir is None:
        args.model_dir = f"saved_models/{args.dataset}"

    # Special handling for NeuroSEED with pre-defined splits
    if args.dataset == "neuroseed" and args.use_predefined_splits:
        print("\n" + "=" * 70)
        print("Using NeuroSEED with Pre-defined Train/Test Splits")
        print("=" * 70)

        try:
            # Load pre-defined splits
            result = load_neuroseed_dataset(
                task=args.neuroseed_task,
                k_neighbors=args.k_neighbors,
                seed=args.random_state,
                use_predefined_splits=True,
            )

            # Unpack the results
            train_graph, test_graph, train_labels, test_labels, train_indices, test_indices = result

            # Train embeddings on training graph
            print("\nTraining embeddings on TRAINING set...")
            train_embeddings, embedding_space = train_hyperbolic_embeddings(
                graph=train_graph,
                embedding_type=args.embedding_type,
                model_dir=os.path.join(args.model_dir, "train"),
                dim=args.dim,
            )

            # Train embeddings on test graph (same model, different graph)
            print("\nTraining embeddings on TEST set...")
            test_embeddings, _ = train_hyperbolic_embeddings(
                graph=test_graph,
                embedding_type=args.embedding_type,
                model_dir=os.path.join(args.model_dir, "test"),
                dim=args.dim,
            )

            # Use the full train/test sets (no re-splitting!)
            X_train, y_train = train_embeddings, train_labels
            X_test, y_test = test_embeddings, test_labels

            print("\n✓ Using pre-defined splits:")
            print(f"  Training set: {len(X_train)} samples")
            print(f"  Test set: {len(X_test)} samples")

        except Exception as e:
            print(f"Error loading NeuroSEED with pre-defined splits: {e}")
            print("Make sure real NeuroSEED data is available in data/neuroseed/")
            print("Falling back to synthetic data with random split...")
            args.use_predefined_splits = False

    # Standard loading for all other cases
    if not (args.dataset == "neuroseed" and args.use_predefined_splits):
        # Load dataset
        try:
            if args.dataset == "neuroseed":
                # Use synthetic data with single graph
                graph, labels, node_indices = load_neuroseed_dataset(
                    task=args.neuroseed_task,
                    num_samples=args.subgraph_size or 1000,
                    k_neighbors=args.k_neighbors,
                    seed=args.random_state,
                    use_predefined_splits=False,
                )
            else:
                graph, labels, node_indices = load_dataset(
                    dataset_name=args.dataset,
                    dataset_type=args.dataset,
                    k_neighbors=args.k_neighbors,
                    datasets_path=args.datasets_path,
                    subgraph_size=args.subgraph_size,
                    random_seed=args.random_state,
                    min_nodes_per_class=args.min_nodes_per_class,
                    max_classes=args.max_classes if args.max_classes > 0 else None,
                    neuroseed_task=args.neuroseed_task,
                    neuroseed_split="train",  # Unused for neuroseed now
                )
        except ValueError as e:
            print(f"Error: {e}")
            return

        # Train embeddings
        embeddings, embedding_space = train_hyperbolic_embeddings(
            graph=graph,
            embedding_type=args.embedding_type,
            model_dir=args.model_dir,
            dim=args.dim,
        )

        # Split data into train and test sets
        print(f"\nSplitting data into train/test sets (test_size={args.test_size})...")
        X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
            embeddings,
            labels,
            node_indices,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=labels,
        )

        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

    # Evaluate KNN with different k values
    results = evaluate_knn(X_train, X_test, y_train, y_test, k_values=args.k_values, n_iterations=args.n_iterations)

    # Print summary
    print("\n" + "=" * 90)
    print(f"SUMMARY OF RESULTS (Mean ± Std over {args.n_iterations} iterations)")
    print("=" * 90)
    if args.n_iterations > 1:
        print(f"{'k':<5} {'Accuracy':<20} {'Precision':<20} {'Recall':<20} {'F1-Score':<20}")
        print("-" * 90)
        for k in sorted(results.keys()):
            r = results[k]
            print(
                f"{k:<5} "
                f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}  "
                f"{r['precision_mean']:.4f} ± {r['precision_std']:.4f}  "
                f"{r['recall_mean']:.4f} ± {r['recall_std']:.4f}  "
                f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}"
            )
    else:
        print(f"{'k':<5} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 90)
        for k in sorted(results.keys()):
            r = results[k]
            print(
                f"{k:<5} "
                f"{r['accuracy_mean']:<12.4f} "
                f"{r['precision_mean']:<12.4f} "
                f"{r['recall_mean']:<12.4f} "
                f"{r['f1_mean']:<12.4f}"
            )
    print("=" * 90)

    # Find best k
    best_k = max(results.keys(), key=lambda k: results[k]["accuracy_mean"])
    print(f"\nBest k value: {best_k} (Accuracy: {results[best_k]['accuracy_mean']:.4f} ± {results[best_k]['accuracy_std']:.4f})")


if __name__ == "__main__":
    main()
