import argparse
import os
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from as_dataset_loader import ASDatasetLoader
from hyperbolic_embeddings import HyperbolicEmbeddings
from poincare_maps_networkx_loader import PoincareMapsLoader
from utils.geometric_conversions import compute_distances, convert_coordinates


def create_test_tree_graph(branching_factor: int = 2, depth: int = 4) -> nx.Graph:
    """Create a balanced tree graph for testing."""
    return nx.balanced_tree(branching_factor, depth)


def simulate_link_removal(graph: nx.Graph, q: float, seed: int = None) -> Dict[str, List[Tuple[int, int]]]:
    """
    Simulate random link removal experiment.

    Args:
        graph: Input NetworkX graph
        q: Probability of keeping a link (removal probability is 1-q)
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing:
        - 'omega_E': List of remaining links (existing edges after removal)
        - 'omega_R': List of removed links (missing links)
        - 'omega_N': List of true non-links (pairs that were never connected)
        - 'omega_R_plus_N': Combined list of removed links and true non-links
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Get all edges in the original graph
    original_edges = list(graph.edges())
    n_nodes = len(graph.nodes())

    # Generate all possible node pairs (excluding self-loops)
    all_possible_pairs = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            all_possible_pairs.append((i, j))

    # Convert original edges to set for faster lookup
    original_edges_set = set(original_edges)
    original_edges_set.update([(v, u) for u, v in original_edges])  # Add both directions

    # Identify true non-links (pairs that were never connected)
    omega_N = []
    for pair in all_possible_pairs:
        if pair not in original_edges_set and (pair[1], pair[0]) not in original_edges_set:
            omega_N.append(pair)

    # Simulate link removal with probability 1-q
    omega_E = []  # Remaining links
    omega_R = []  # Removed links

    for edge in original_edges:
        if random.random() < q:
            # Keep the link
            omega_E.append(edge)
        else:
            # Remove the link
            omega_R.append(edge)

    # Combine removed links and true non-links
    omega_R_plus_N = omega_R + omega_N

    return {"omega_E": omega_E, "omega_R": omega_R, "omega_N": omega_N, "omega_R_plus_N": omega_R_plus_N}


def predict_links(distances: np.ndarray, n_links: int = 10, candidate_edges: list = None) -> list:
    """
    Predict links based on hyperbolic distances.

    Args:
        distances: Distance matrix between all nodes
        n_links: Number of links to predict
        candidate_edges: List of candidate edge pairs to consider for prediction (e.g., Ω_R + Ω_N)
                        If provided, only these pairs will be considered for prediction

    Returns:
        List of predicted links as (u, v, distance) tuples, sorted by distance (smallest first)
    """

    if candidate_edges is not None:
        # Use specific candidate edges (e.g., Ω_R + Ω_N)
        candidate_distances = []

        for u, v in candidate_edges:
            dist = distances[u, v]
            candidate_distances.append((u, v, dist))

        # Sort by distance and take the n_links with smallest distances
        candidate_distances.sort(key=lambda x: x[2])
        predicted_links = candidate_distances[:n_links]

    else:
        # Original behavior: consider all possible pairs
        # Create a copy of the distance matrix
        dist_matrix = distances.copy()

        # Set diagonal to infinity to exclude self-loops
        np.fill_diagonal(dist_matrix, np.inf)

        # Only consider upper triangle to avoid duplicates (i < j)
        upper_triangle_indices = np.triu_indices_from(dist_matrix, k=1)
        upper_distances = dist_matrix[upper_triangle_indices]

        # Find the N smallest distances in upper triangle
        sorted_indices = np.argsort(upper_distances)[:n_links]
        row_indices = upper_triangle_indices[0][sorted_indices]
        col_indices = upper_triangle_indices[1][sorted_indices]
        predicted_distances = upper_distances[sorted_indices]

        # Create list of predicted links
        predicted_links = [(int(row), int(col), float(dist)) for row, col, dist in zip(row_indices, col_indices, predicted_distances)]

    return predicted_links


def evaluate_predictions(predicted_links: list, omega_R: list, omega_N: list) -> dict:
    """Evaluate link prediction performance."""

    # Extract predicted node pairs
    predicted_pairs = [(u, v) for u, v, _ in predicted_links]

    # Calculate metrics

    true_positives = sum(1 for pair in predicted_pairs if pair in omega_R)
    false_positives = sum(1 for pair in predicted_pairs if pair in omega_N)
    false_negatives = sum(1 for pair in omega_R if pair not in predicted_pairs)
    true_negatives = sum(1 for pair in omega_N if pair not in predicted_pairs)

    accuracy = (true_positives + true_negatives) / (true_positives + false_positives + false_negatives + true_negatives)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
    }

    return metrics


def compute_lift_curve(all_candidates_with_distances: list, omega_R: list) -> dict:
    """
    Compute lift curve analysis by dividing candidates into deciles.

    Args:
        all_candidates_with_distances: List of (u, v, distance) tuples
        omega_R: List of true removed links (u, v) tuples

    Returns:
        Dictionary containing decile statistics
    """
    # Convert omega_R to set for faster lookup
    omega_R_set = set(omega_R)
    omega_R_set.update([(v, u) for u, v in omega_R])  # Add both directions

    # Sort candidates by distance (ascending - closest first)
    sorted_candidates = sorted(all_candidates_with_distances, key=lambda x: x[2])

    total_candidates = len(sorted_candidates)
    decile_size = total_candidates // 10

    # Calculate overall baseline TP rate
    total_true_positives = sum(1 for u, v, _ in sorted_candidates if (u, v) in omega_R_set)
    baseline_tp_rate = total_true_positives / total_candidates if total_candidates > 0 else 0

    decile_stats = []
    cumulative_tps = 0

    for decile in range(10):
        start_idx = decile * decile_size
        if decile == 9:  # Last decile gets remaining candidates
            end_idx = total_candidates
        else:
            end_idx = (decile + 1) * decile_size

        decile_candidates = sorted_candidates[start_idx:end_idx]
        decile_count = len(decile_candidates)

        # Count true positives in this decile
        decile_tps = sum(1 for u, v, _ in decile_candidates if (u, v) in omega_R_set)
        cumulative_tps += decile_tps

        # Calculate rates
        decile_tp_rate = decile_tps / decile_count if decile_count > 0 else 0
        cumulative_tp_rate = cumulative_tps / (end_idx) if end_idx > 0 else 0
        lift = decile_tp_rate / baseline_tp_rate if baseline_tp_rate > 0 else 0

        decile_stats.append(
            {
                "decile": decile + 1,
                "count": decile_count,
                "true_positives": decile_tps,
                "tp_rate": decile_tp_rate,
                "cumulative_tps": cumulative_tps,
                "cumulative_tp_rate": cumulative_tp_rate,
                "lift": lift,
            }
        )

    return {
        "decile_stats": decile_stats,
        "total_candidates": total_candidates,
        "total_true_positives": total_true_positives,
        "baseline_tp_rate": baseline_tp_rate,
    }


def plot_lift_curve(lift_data: dict, args, save_path: str):
    """
    Create lift curve visualization with bar chart and cumulative curve.

    Args:
        lift_data: Dictionary from compute_lift_curve()
        args: Command line arguments
        save_path: Path to save the plot
    """
    decile_stats = lift_data["decile_stats"]
    baseline_tp_rate = lift_data["baseline_tp_rate"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Top plot: Bar chart of true positives per decile
    deciles = [d["decile"] for d in decile_stats]
    tps = [d["true_positives"] for d in decile_stats]
    counts = [d["count"] for d in decile_stats]

    bars = ax1.bar(deciles, tps, alpha=0.7, color="skyblue", edgecolor="black")
    ax1.set_xlabel("Decile")
    ax1.set_ylabel("True Positives")
    ax1.set_title(f"True Positives per Decile - {args.embedding_type} " f"({args.dataset})")
    ax1.set_xticks(deciles)
    ax1.grid(True, alpha=0.3)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, height + 0.1, f"n={count}", ha="center", va="bottom", fontsize=8)

    # Bottom plot: Cumulative lift curve
    cumulative_tp_rates = [d["cumulative_tp_rate"] for d in decile_stats]
    lifts = [d["lift"] for d in decile_stats]

    ax2.plot(deciles, cumulative_tp_rates, "o-", linewidth=2, markersize=6, label="Cumulative TP Rate", color="blue")
    ax2.axhline(y=baseline_tp_rate, color="red", linestyle="--", label=f"Baseline TP Rate ({baseline_tp_rate:.4f})")

    ax2.set_xlabel("Decile")
    ax2.set_ylabel("Cumulative True Positive Rate")
    ax2.set_title("Cumulative Lift Curve")
    ax2.set_xticks(deciles)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Add lift values as text annotations
    for i, (decile, lift) in enumerate(zip(deciles, lifts)):
        ax2.text(decile, cumulative_tp_rates[i] + 0.01, f"Lift: {lift:.2f}", ha="center", va="bottom", fontsize=8, rotation=45)

    plt.tight_layout()

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def save_lift_table(lift_data: dict, args, save_path: str):
    """
    Save lift curve analysis as a detailed text table.

    Args:
        lift_data: Dictionary from compute_lift_curve()
        args: Command line arguments
        save_path: Path to save the table
    """
    decile_stats = lift_data["decile_stats"]
    total_candidates = lift_data["total_candidates"]
    total_true_positives = lift_data["total_true_positives"]
    baseline_tp_rate = lift_data["baseline_tp_rate"]

    with open(save_path, "w") as f:
        f.write("Lift Curve Analysis\n")
        f.write("=" * 50 + "\n\n")

        # Summary information
        f.write("Summary:\n")
        f.write("-" * 10 + "\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Embedding Type: {args.embedding_type}\n")
        f.write(f"Total Candidates: {total_candidates}\n")
        f.write(f"Total True Positives: {total_true_positives}\n")
        f.write(f"Baseline TP Rate: {baseline_tp_rate:.4f}\n\n")

        # Detailed table
        f.write("Decile Analysis:\n")
        f.write("-" * 15 + "\n")
        f.write(f"{'Decile':<6} {'Count':<8} {'TPs':<6} " f"{'TP Rate':<10} {'Cum TPs':<8} " f"{'Cum TP Rate':<12} {'Lift':<8}\n")
        f.write("-" * 70 + "\n")

        for stat in decile_stats:
            f.write(
                f"{stat['decile']:<6} {stat['count']:<8} "
                f"{stat['true_positives']:<6} {stat['tp_rate']:<10.4f} "
                f"{stat['cumulative_tps']:<8} "
                f"{stat['cumulative_tp_rate']:<12.4f} "
                f"{stat['lift']:<8.2f}\n"
            )

        f.write("\n")

        # Interpretation
        f.write("Interpretation:\n")
        f.write("-" * 15 + "\n")
        f.write("- Deciles are ordered by distance (ascending): " "Decile 1 = closest pairs\n")
        f.write("- TP Rate: True positive rate within the decile\n")
        f.write("- Cum TP Rate: Cumulative true positive rate " "up to this decile\n")
        f.write("- Lift: Ratio of decile TP rate to baseline TP rate\n")
        f.write("- Lift > 1.0 indicates better than random performance\n")


def plot_embeddings(
    poincare_embeddings: np.ndarray,
    embedding_runner: HyperbolicEmbeddings,
    edges: list,
    true_positives: list,
    false_positives: list,
    plot_title: str,
    save_path: str,
    max_edges_to_plot: int = 1000,  # Limit edges for large graphs
):

    # Embeddings with predictions (in Poincaré disk)
    x, y = poincare_embeddings[:, 0], poincare_embeddings[:, 1]
    num_nodes = len(x)

    fig, ax = plt.subplots(figsize=(12, 12))  # Larger figure for large graphs
    ax.set_title(plot_title)

    # For large graphs, limit the number of edges plotted for readability
    edges_list = list(edges)  # Convert EdgeView to list if needed
    edges_to_plot = edges_list[:max_edges_to_plot] if len(edges_list) > max_edges_to_plot else edges_list
    if len(edges_list) > max_edges_to_plot:
        print(f"Plotting {max_edges_to_plot} out of {len(edges_list)} edges for better visualization")

    # Plot omega_E edges
    for u, v in edges_to_plot:
        if u < len(x) and v < len(x):
            p1 = (x[u], y[u])
            p2 = (x[v], y[v])

            embedding_runner.plot_geodesic_arc(p1, p2, ax)

    # Plot recovered links
    for u, v, _ in true_positives:
        if u < len(x) and v < len(x):
            p1 = (x[u], y[u])
            p2 = (x[v], y[v])

            embedding_runner.plot_geodesic_arc(p1, p2, ax, color="green", linestyle="solid", linewidth=2)

    # Plot false positives
    for u, v, _ in false_positives:
        if u < len(x) and v < len(x):
            p1 = (x[u], y[u])
            p2 = (x[v], y[v])

            embedding_runner.plot_geodesic_arc(p1, p2, ax, color="red", linestyle="solid", linewidth=2)

    # Adjust node size and labels based on graph size
    node_size = 150 if num_nodes < 100 else (50 if num_nodes < 1000 else 10)
    show_labels = num_nodes < 100  # Only show labels for small graphs

    ax.scatter(x, y, s=node_size, edgecolor="black", color="skyblue", zorder=2, alpha=0.7)

    if show_labels:
        for i in range(len(x)):
            ax.text(x[i], y[i], str(i), fontsize=8, ha="center", va="center", zorder=3)
    else:
        print(f"Skipping node labels for large graph ({num_nodes} nodes)")

    # Draw Poincaré disk boundary
    ax.add_artist(plt.Circle((0, 0), 1, fill=False, color="black", linestyle="--"))
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def save_metrics_to_file(metrics: dict, args, results: dict, graph_info: dict, save_path: str):
    """Save evaluation metrics and experiment details to a text file."""
    with open(save_path, "w") as f:
        f.write("Link Prediction Results\n")
        f.write("=" * 50 + "\n\n")

        # Experiment parameters
        f.write("Experiment Parameters:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Embedding Type: {args.embedding_type}\n")
        f.write(f"Link Removal Probability (1-q): {1-args.q:.2f}\n")
        f.write(f"Link Retention Probability (q): {args.q:.2f}\n")
        f.write(f"Number of Links to Predict: {args.n_links if args.n_links else 'Same as removed links'}\n\n")

        # Graph information
        f.write("Graph Information:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Original Graph Nodes: {graph_info['original_nodes']}\n")
        f.write(f"Original Graph Edges: {graph_info['original_edges']}\n")
        f.write(f"Remaining Edges (Ω_E): {len(results['omega_E'])}\n")
        f.write(f"Removed Edges (Ω_R): {len(results['omega_R'])}\n")
        f.write(f"True Non-links (Ω_N): {len(results['omega_N'])}\n")
        f.write(f"Candidate Links (Ω_R + Ω_N): {len(results['omega_R_plus_N'])}\n\n")

        # Performance metrics
        f.write("Performance Metrics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1-Score: {metrics['f1_score']:.4f}\n\n")

        # Confusion matrix details
        f.write("Confusion Matrix:\n")
        f.write("-" * 17 + "\n")
        f.write(f"True Positives: {metrics['true_positives']}\n")
        f.write(f"False Positives: {metrics['false_positives']}\n")
        f.write(f"False Negatives: {metrics['false_negatives']}\n")
        f.write(f"True Negatives: {metrics['true_negatives']}\n\n")

        # Additional analysis
        total_predictions = len(results["omega_R"]) if args.n_links is None else args.n_links
        f.write("Additional Analysis:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Predictions Made: {total_predictions}\n")
        f.write(f"Percentage of Removed Links Recovered: {(metrics['true_positives'] / len(results['omega_R']) * 100):.2f}%\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Link prediction pipeline example")
    parser.add_argument(
        "--embedding_type",
        type=str,
        default="poincare_maps",
        choices=["poincare_embeddings", "lorentz", "poincare_maps", "dmercator", "hydra", "hypermap", "hydra_plus"],
        help="Type of hyperbolic embedding to use",
    )
    parser.add_argument(
        "--n_links", type=int, default=None, help="Number of links to predict (default: same as number of removed edges in graph)"
    )
    parser.add_argument("--q", type=float, default=0.5, help="Probability of keeping a link (default: 0.5)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ToggleSwitch",
        choices=["ToggleSwitch", "AS", "Olsson", "MyeloidProgenitors", "krumsiek11_blobs", "Paul"],
        help="Dataset to use: ToggleSwitch (PoincareMaps) or AS (Stanford SNAP Autonomous Systems)",
    )

    return parser.parse_args()


def main():
    """Run a simple example of the link prediction pipeline."""
    args = parse_args()
    PATH = f"test/other/plots/link_prediction/{args.dataset}"
    RESULTS_PATH = f"test/other/results/{args.dataset}"

    # Create output directories if they don't exist
    os.makedirs(PATH, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)

    print("Link Prediction Pipeline Example")
    print("=" * 40)
    print(f"Embedding type: {args.embedding_type}")
    print(f"Output directory: {PATH}")
    print(f"Results directory: {RESULTS_PATH}")
    print()

    # Load dataset based on type
    if args.dataset == "AS":
        print("Loading Stanford SNAP AS dataset...")
        as_loader = ASDatasetLoader()
        graph, metadata = as_loader.load_as_networkx()
    else:
        print(f"Loading PoincareMaps dataset: {args.dataset}")
        loader = PoincareMapsLoader("models/PoincareMaps/datasets/")
        graph, metadata = loader.load_as_networkx(args.dataset, k_neighbors=15)

    # Store original graph information for results file
    graph_info = {"original_nodes": len(graph.nodes()), "original_edges": len(graph.edges())}

    # Train hyperbolic embeddings
    print("Training hyperbolic embeddings for original graph...")

    # Adjust configurations based on dataset size
    num_nodes = len(graph.nodes())
    is_large_dataset = num_nodes > 1000  # AS dataset has 6474 nodes

    if is_large_dataset:
        print(f"Large dataset detected ({num_nodes} nodes), adjusting parameters...")
        configurations = {
            "poincare_embeddings": {"dim": 2, "negs": 10, "epochs": 500, "batch_size": 512, "dimension": 1},
            "lorentz": {"dim": 2, "epochs": 5000, "batch_size": 2048, "num_nodes": num_nodes},
            "dmercator": {"dim": 2},  # Increase dimension for larger graphs
            "hydra": {"dim": 3},  # Increase dimension for larger graphs
            "poincare_maps": {"dim": 2, "epochs": 500},
            "hypermap": {"dim": 3},
            "hydra_plus": {"dim": 3},  # Increase dimension for larger graphs
        }
    else:
        configurations = {
            "poincare_embeddings": {"dim": 2, "negs": 5, "epochs": 1000, "batch_size": 256, "dimension": 1},
            "lorentz": {"dim": 2, "epochs": 10000, "batch_size": 1024, "num_nodes": num_nodes},
            "dmercator": {"dim": 1},
            "hydra": {"dim": 2},
            "poincare_maps": {"dim": 2, "epochs": 1000},
            "hypermap": {"dim": 3},
            "hydra_plus": {"dim": 2},
        }
    config = configurations[args.embedding_type]
    embedding_runner = HyperbolicEmbeddings(embedding_type=args.embedding_type, config=config)

    # Prepare training data
    adjacency_matrix = nx.to_numpy_array(graph)

    # Train the model
    model_path = f"saved_models/{args.dataset}/link_prediction/{args.embedding_type}_original_graph_model.bin"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    embedding_runner.train(adjacency_matrix=adjacency_matrix, model_path=model_path)

    # Get embeddings
    original_graph_embeddings = embedding_runner.get_all_embeddings(model_path)
    print(f"Embeddings shape: {original_graph_embeddings.shape}")

    native_space = embedding_runner.model.native_space

    # Convert embeddings to Poincaré for better visualization
    if native_space != "poincare":
        original_graph_poincare_embeddings = convert_coordinates(original_graph_embeddings, native_space, "poincare")
    else:
        original_graph_poincare_embeddings = original_graph_embeddings

    plot_title = "Original Graph Embeddings in Poincaré Disk"
    save_path = f"{PATH}/{args.embedding_type}_original_graph_embeddings.pdf"
    max_edges = 2000 if is_large_dataset else 1000
    plot_embeddings(original_graph_poincare_embeddings, embedding_runner, graph.edges(), [], [], plot_title, save_path, max_edges)

    # Generating graph from Distance Matrix
    print("Generating graph from distance matrix...")

    # Convert to hyperboloid coordinates if needed
    if native_space != "hyperboloid":
        print(f"Converting embeddings from {native_space} to hyperboloid coordinates")
        orginal_graph_hyperboloid_embeddings = convert_coordinates(original_graph_embeddings, native_space, "hyperboloid")
    else:
        orginal_graph_hyperboloid_embeddings = original_graph_embeddings

    orginal_graph_distance_matrix = compute_distances(orginal_graph_hyperboloid_embeddings, space="hyperboloid")

    distance_matrix_predicted_links = predict_links(orginal_graph_distance_matrix, n_links=len(graph.edges()))

    # print(f"Distance matrix predicted links: {distance_matrix_predicted_links}")

    # Analyze predictions: recovered links vs false positives
    recovered_links = []  # Correctly predicted removed links (from Ω_R)
    false_positives = []  # Incorrectly predicted non-links (from Ω_N)
    true_edges_set = set(graph.edges())
    true_edges_set.update([(v, u) for u, v in graph.edges()])  # Add both directions

    for u, v, dist in distance_matrix_predicted_links:
        if (u, v) in true_edges_set:
            recovered_links.append((u, v, dist))
        else:
            false_positives.append((u, v, dist))

    print("\nCreating visualization...")

    # Convert embeddings to Poincaré for better visualization
    if native_space != "poincare":
        original_graph_poincare_embeddings = convert_coordinates(orginal_graph_hyperboloid_embeddings, "hyperboloid", "poincare")
    else:
        original_graph_poincare_embeddings = original_graph_embeddings

    plot_title = "Distance Matrix Predicted Links in Poincaré Disk"
    save_path = f"{PATH}/{args.embedding_type}_distance_matrix_predicted_links.pdf"
    plot_embeddings(
        original_graph_poincare_embeddings,
        embedding_runner,
        graph.edges(),
        recovered_links,
        false_positives,
        plot_title,
        save_path,
        max_edges,
    )

    ########################################################################################################################################################

    # Simulate link removal
    print("Simulating link removal...")
    results = simulate_link_removal(graph, q=args.q)
    omega_E = results["omega_E"]  # List of remaining links

    # Create graph with same nodes as original graph and edges from omega_E
    graph_from_omega_E = nx.Graph()
    graph_from_omega_E.add_nodes_from(graph.nodes())  # Preserve all original nodes
    graph_from_omega_E.add_edges_from(omega_E)  # Add only remaining edges

    print(f"Graph created from omega_E: {len(graph_from_omega_E.nodes)} nodes, {len(graph_from_omega_E.edges)} edges")

    # Train hyperbolic embeddings
    print("Training hyperbolic embeddings...")

    # Use the same configurations as before (already adjusted for dataset size)
    if is_large_dataset:
        configurations = {
            "poincare_embeddings": {"dim": 2, "negs": 10, "epochs": 500, "batch_size": 512, "dimension": 1},
            "lorentz": {"dim": 2, "epochs": 5000, "batch_size": 2048, "num_nodes": len(graph.nodes)},
            "dmercator": {"dim": 2},  # Increase dimension for larger graphs
            "hydra": {"dim": 3},  # Increase dimension for larger graphs
            "poincare_maps": {"dim": 2, "epochs": 500},
            "hypermap": {"dim": 3},
            "hydra_plus": {"dim": 3},  # Increase dimension for larger graphs
        }
    else:
        configurations = {
            "poincare_embeddings": {"dim": 2, "negs": 5, "epochs": 1000, "batch_size": 256, "dimension": 1},
            "lorentz": {"dim": 2, "epochs": 10000, "batch_size": 1024, "num_nodes": len(graph.nodes)},
            "dmercator": {"dim": 1},
            "hydra": {"dim": 2},
            "poincare_maps": {"dim": 2, "epochs": 1000},
            "hypermap": {"dim": 3},
            "hydra_plus": {"dim": 2},
        }
    config = configurations[args.embedding_type]
    embedding_runner = HyperbolicEmbeddings(embedding_type=args.embedding_type, config=config)

    # Prepare training data
    adjacency_matrix = nx.to_numpy_array(graph_from_omega_E)

    # Train the model
    model_path = f"saved_models/{args.dataset}/link_prediction/{args.embedding_type}_model.bin"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    embedding_runner.train(adjacency_matrix=adjacency_matrix, model_path=model_path)

    # Get embeddings
    embeddings = embedding_runner.get_all_embeddings(model_path)
    print(f"Embeddings shape: {embeddings.shape}")

    # Convert embeddings to Poincaré for better visualization
    if native_space != "poincare":
        poincare_embeddings = convert_coordinates(embeddings, native_space, "poincare")
    else:
        poincare_embeddings = embeddings

    plot_title = "Omega E Graph Embeddings in Poincaré Disk"
    save_path = f"{PATH}/{args.embedding_type}_omega_E_graph_embeddings.pdf"
    plot_embeddings(poincare_embeddings, embedding_runner, omega_E, [], [], plot_title, save_path, max_edges)

    # Convert to hyperboloid coordinates if needed
    native_space = embedding_runner.model.native_space
    if native_space != "hyperboloid":
        print(f"Converting embeddings from {native_space} to hyperboloid coordinates")
        hyperboloid_embeddings = convert_coordinates(embeddings, native_space, "hyperboloid")
    else:
        hyperboloid_embeddings = embeddings

    # Compute hyperbolic distances
    print("Computing hyperbolic distances...")
    distances = compute_distances(hyperboloid_embeddings, space="hyperboloid")
    print(f"Distance matrix shape: {distances.shape}")
    print(f"Distance range: {distances[distances > 0].min():.4f} to {distances.max():.4f}")

    # Predict links from Ω_R + Ω_N (removed links + true non-links)
    omega_R = results["omega_R"]
    omega_N = results["omega_N"]
    omega_R_plus_N = results["omega_R_plus_N"]
    n_links = args.n_links if args.n_links else len(omega_R)
    print(f"Number of links to predict: {n_links}")

    predicted_links = predict_links(distances, n_links=n_links, candidate_edges=omega_R_plus_N)

    # Evaluate predictions against removed edges (Ω_R) - the true positive targets
    print()
    print("Evaluating predictions...")
    metrics = evaluate_predictions(predicted_links, omega_R, omega_N)

    # Perform lift curve analysis
    print("\nPerforming lift curve analysis...")

    # Collect all candidates with their distances
    all_candidates_with_distances = []
    for u, v in omega_R_plus_N:
        dist = distances[u, v]
        all_candidates_with_distances.append((u, v, dist))

    # Compute lift curve statistics
    lift_data = compute_lift_curve(all_candidates_with_distances, omega_R)

    # Generate lift curve visualization
    lift_plot_path = f"{RESULTS_PATH}/{args.embedding_type}_lift_curve.pdf"
    plot_lift_curve(lift_data, args, lift_plot_path)

    # Save detailed lift analysis table
    lift_table_path = f"{RESULTS_PATH}/{args.embedding_type}_lift_analysis.txt"
    save_lift_table(lift_data, args, lift_table_path)

    print(f"Lift curve plot saved to: {lift_plot_path}")
    print(f"Lift analysis table saved to: {lift_table_path}")

    # Print summary of lift analysis
    print("\nLift Curve Summary:")
    print(f"Total candidates analyzed: {lift_data['total_candidates']}")
    print(f"Baseline TP rate: {lift_data['baseline_tp_rate']:.4f}")
    print(f"Top decile lift: {lift_data['decile_stats'][0]['lift']:.2f}")
    print(f"Top 3 deciles cumulative TP rate: " f"{lift_data['decile_stats'][2]['cumulative_tp_rate']:.4f}")

    # Print results
    print("\nLink Prediction Results:")
    print(f"Predicted {len(predicted_links)} links from {len(omega_R_plus_N)} candidates (Ω_R + Ω_N)")
    print(f"Target removed edges (Ω_R): {len(omega_R)} links")
    print(f"True non-links (Ω_N): {len(results['omega_N'])} pairs")
    print("\nMetrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Save metrics to file
    results_file_path = f"{RESULTS_PATH}/{args.embedding_type}_results.txt"
    save_metrics_to_file(metrics, args, results, graph_info, results_file_path)
    print(f"\nResults saved to: {results_file_path}")

    # Analyze predictions: recovered links vs false positives
    recovered_links = []  # Correctly predicted removed links (from Ω_R)
    false_positives = []  # Incorrectly predicted non-links (from Ω_N)
    omega_R_set = set(omega_R)
    omega_R_set.update([(v, u) for u, v in omega_R])  # Add both directions

    for u, v, dist in predicted_links:
        if (u, v) in omega_R_set:
            recovered_links.append((u, v, dist))
        else:
            false_positives.append((u, v, dist))

    # print(f"\nPrediction Analysis:")

    # print("\nRecovered links (node1, node2, distance):")
    # for u, v, dist in recovered_links:
    #     print(f"  ({u}, {v}): {dist:.4f}")

    # print("\nFalse positives (node1, node2, distance):")
    # for u, v, dist in false_positives:
    #     print(f"  ({u}, {v}): {dist:.4f}")

    true_links = []
    for u, v in omega_R:
        dist = distances[u, v]
        true_links.append((u, v, dist))

    # print("\nTrue links (Omega R) (node1, node2, distance):")
    # for u, v, dist in true_links:
    #     print(f"  ({u}, {v}): {dist:.4f}")

    # Create a visualization showing predicted links
    print("\nCreating visualization...")

    # Convert embeddings to Poincaré for better visualization
    if native_space != "poincare":
        poincare_embeddings = convert_coordinates(hyperboloid_embeddings, "hyperboloid", "poincare")
    else:
        poincare_embeddings = embeddings

    plot_title = "Predicted Links in Poincaré Disk"
    save_path = f"{PATH}/{args.embedding_type}_omega_E_predicted_links.pdf"
    plot_embeddings(poincare_embeddings, embedding_runner, omega_E, recovered_links, false_positives, plot_title, save_path, max_edges)

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
