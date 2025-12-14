import argparse
import json
import os
import pickle
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy
from graspologic.embed import AdjacencySpectralEmbed

from hyperbolic_embeddings import HyperbolicEmbeddings
from poincare_maps_networkx_loader import PoincareMapsLoader
from snap_dataset_loader import SNAPDatasetLoader
from utils.geometric_conversions import compute_distances, convert_coordinates

try:
    from torch_geometric.datasets import Airports
except ImportError:
    Airports = None


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


def predict_links(distances: np.ndarray, n_links: int = 10, candidate_edges: list = None, return_sorted_candidates: bool = False):
    """
    Predict links based on hyperbolic distances.

    Args:
        distances: Distance matrix between all nodes
        n_links: Number of links to predict
        candidate_edges: List of candidate edge pairs to consider for prediction (e.g., Ω_R + Ω_N)
                        If provided, only these pairs will be considered for prediction
        return_sorted_candidates: If True, also return the sorted list of all candidates with distances

    Returns:
        If return_sorted_candidates is False:
            List of predicted links as (u, v, distance) tuples, sorted by distance (smallest first)
        If return_sorted_candidates is True:
            Tuple of (predicted_links, sorted_all_candidates) where:
            - predicted_links: List of top n_links as (u, v, distance) tuples
            - sorted_all_candidates: List of all candidates sorted by distance (smallest first)
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

        if return_sorted_candidates:
            return predicted_links, candidate_distances
        else:
            return predicted_links

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


def predict_links_from_probabilities(
    probabilities: np.ndarray, n_links: int = 10, candidate_edges: list = None, return_sorted_candidates: bool = False
):
    """
    Predict links based on RDPG probability matrix.

    Args:
        probabilities: Probability matrix between all nodes (X @ X^T)
        n_links: Number of links to predict
        candidate_edges: List of candidate edge pairs to consider for prediction (e.g., Ω_R + Ω_N)
                        If provided, only these pairs will be considered for prediction
        return_sorted_candidates: If True, also return the sorted list of all candidates with probabilities

    Returns:
        If return_sorted_candidates is False:
            List of predicted links as (u, v, probability) tuples, sorted by probability (highest first)
        If return_sorted_candidates is True:
            Tuple of (predicted_links, sorted_all_candidates) where:
            - predicted_links: List of top n_links as (u, v, probability) tuples
            - sorted_all_candidates: List of all candidates sorted by probability (highest first)
    """

    if candidate_edges is not None:
        # Use specific candidate edges (e.g., Ω_R + Ω_N)
        candidate_probs = []

        for u, v in candidate_edges:
            prob = probabilities[u, v]
            candidate_probs.append((u, v, prob))

        # Sort by probability (descending - highest first) and take the n_links with highest probabilities
        candidate_probs.sort(key=lambda x: x[2], reverse=True)
        predicted_links = candidate_probs[:n_links]

        if return_sorted_candidates:
            return predicted_links, candidate_probs
        else:
            return predicted_links

    else:
        # Original behavior: consider all possible pairs
        # Create a copy of the probability matrix
        prob_matrix = probabilities.copy()

        # Set diagonal to negative infinity to exclude self-loops
        np.fill_diagonal(prob_matrix, -np.inf)

        # Only consider upper triangle to avoid duplicates (i < j)
        upper_triangle_indices = np.triu_indices_from(prob_matrix, k=1)
        upper_probs = prob_matrix[upper_triangle_indices]

        # Find the N highest probabilities in upper triangle
        sorted_indices = np.argsort(upper_probs)[::-1][:n_links]  # Reverse sort for descending
        row_indices = upper_triangle_indices[0][sorted_indices]
        col_indices = upper_triangle_indices[1][sorted_indices]
        predicted_probs = upper_probs[sorted_indices]

        # Create list of predicted links
        predicted_links = [(int(row), int(col), float(prob)) for row, col, prob in zip(row_indices, col_indices, predicted_probs)]

    return predicted_links


def evaluate_predictions(predicted_links: list, omega_R: list, omega_N: list) -> dict:
    """Evaluate link prediction performance."""

    # Extract predicted node pairs
    predicted_pairs = [(u, v) for u, v, _ in predicted_links]

    # Convert to sets with both directions for consistent matching
    omega_R_set = set(omega_R)
    omega_R_set.update([(v, u) for u, v in omega_R])  # Add both directions

    omega_N_set = set(omega_N)
    omega_N_set.update([(v, u) for u, v in omega_N])  # Add both directions

    predicted_pairs_set = set(predicted_pairs)
    predicted_pairs_set.update([(v, u) for u, v in predicted_pairs])  # Add both directions

    # Calculate metrics (using sets for consistent bidirectional matching)
    true_positives = sum(1 for pair in predicted_pairs if pair in omega_R_set)
    false_positives = sum(1 for pair in predicted_pairs if pair in omega_N_set)
    false_negatives = sum(1 for pair in omega_R if pair not in predicted_pairs_set)
    true_negatives = sum(1 for pair in omega_N if pair not in predicted_pairs_set)

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


def compute_lift_curve(all_candidates_with_distances: list, omega_R: list, n_bins: int = 10, descending: bool = False) -> dict:
    """
    Compute lift curve analysis by dividing candidates into bins (deciles, centiles, etc.).

    Args:
        all_candidates_with_distances: List of (u, v, distance/probability) tuples
        omega_R: List of true removed links (u, v) tuples
        n_bins: Number of bins to divide candidates into (10 for deciles, 100 for centiles, etc.)
        descending: If True, sort by value descending (for probabilities). If False, sort ascending (for distances).

    Returns:
        Dictionary containing bin statistics
    """
    # Convert omega_R to set for faster lookup
    omega_R_set = set(omega_R)
    omega_R_set.update([(v, u) for u, v in omega_R])  # Add both directions

    # Sort candidates by distance/probability
    if descending:
        # Sort by probability (descending - highest first)
        sorted_candidates = sorted(all_candidates_with_distances, key=lambda x: x[2], reverse=True)
    else:
        # Sort by distance (ascending - closest first)
        sorted_candidates = sorted(all_candidates_with_distances, key=lambda x: x[2])

    total_candidates = len(sorted_candidates)
    bin_size = total_candidates // n_bins

    # Calculate overall baseline TP rate
    total_true_positives = sum(1 for u, v, _ in sorted_candidates if (u, v) in omega_R_set)
    baseline_tp_rate = total_true_positives / total_candidates if total_candidates > 0 else 0

    bin_stats = []
    cumulative_tps = 0

    for bin_idx in range(n_bins):
        start_idx = bin_idx * bin_size
        if bin_idx == n_bins - 1:  # Last bin gets remaining candidates
            end_idx = total_candidates
        else:
            end_idx = (bin_idx + 1) * bin_size

        bin_candidates = sorted_candidates[start_idx:end_idx]
        bin_count = len(bin_candidates)

        # Count true positives in this bin
        bin_tps = sum(1 for u, v, _ in bin_candidates if (u, v) in omega_R_set)
        cumulative_tps += bin_tps

        # Calculate rates
        bin_tp_rate = bin_tps / bin_count if bin_count > 0 else 0
        cumulative_tp_rate = cumulative_tps / (end_idx) if end_idx > 0 else 0
        lift = bin_tp_rate / baseline_tp_rate if baseline_tp_rate > 0 else 0

        bin_stats.append(
            {
                "bin": bin_idx + 1,
                "count": bin_count,
                "true_positives": bin_tps,
                "tp_rate": bin_tp_rate,
                "cumulative_tps": cumulative_tps,
                "cumulative_tp_rate": cumulative_tp_rate,
                "lift": lift,
            }
        )

    return {
        "bin_stats": bin_stats,
        "n_bins": n_bins,
        "total_candidates": total_candidates,
        "total_true_positives": total_true_positives,
        "baseline_tp_rate": baseline_tp_rate,
    }


def plot_lift_curve(lift_data: dict, dataset: str, embedding_type: str, save_path: str):
    """
    Create lift curve visualization with bar chart and cumulative curve.

    Args:
        lift_data: Dictionary from compute_lift_curve()
        dataset: Dataset name for plot title
        embedding_type: Embedding type name for plot title
        save_path: Path to save the plot
    """
    bin_stats = lift_data["bin_stats"]
    n_bins = lift_data["n_bins"]
    baseline_tp_rate = lift_data["baseline_tp_rate"]

    # Determine bin type name for labeling
    if n_bins == 10:
        bin_type = "Decile"
    elif n_bins == 100:
        bin_type = "Centile"
    elif n_bins == 20:
        bin_type = "Ventile"
    elif n_bins == 4:
        bin_type = "Quartile"
    else:
        bin_type = f"Bin (1/{n_bins})"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Top plot: Bar chart of true positives per bin
    bins = [d["bin"] for d in bin_stats]
    tps = [d["true_positives"] for d in bin_stats]
    counts = [d["count"] for d in bin_stats]

    bars = ax1.bar(bins, tps, alpha=0.7, color="skyblue", edgecolor="black")
    ax1.set_xlabel(bin_type)
    ax1.set_ylabel("True Positives")
    ax1.set_title(f"True Positives per {bin_type} - {embedding_type} ({dataset})")

    # Adjust x-axis ticks for readability
    if n_bins <= 20:
        ax1.set_xticks(bins)
    else:
        # For large number of bins, show every 10th tick
        tick_step = max(1, n_bins // 10)
        ax1.set_xticks(bins[::tick_step])

    ax1.grid(True, alpha=0.3)

    # Add count labels on bars (only for smaller number of bins to avoid clutter)
    if n_bins <= 20:
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, height + 0.1, f"n={count}", ha="center", va="bottom", fontsize=8)

    # Bottom plot: Cumulative lift curve
    cumulative_tp_rates = [d["cumulative_tp_rate"] for d in bin_stats]
    lifts = [d["lift"] for d in bin_stats]

    ax2.plot(bins, cumulative_tp_rates, "o-", linewidth=2, markersize=6, label="Cumulative TP Rate", color="blue")
    ax2.axhline(y=baseline_tp_rate, color="red", linestyle="--", label=f"Baseline TP Rate ({baseline_tp_rate:.4f})")

    ax2.set_xlabel(bin_type)
    ax2.set_ylabel("Cumulative True Positive Rate")
    ax2.set_title("Cumulative Lift Curve")

    # Adjust x-axis ticks for readability
    if n_bins <= 20:
        ax2.set_xticks(bins)
    else:
        # For large number of bins, show every 10th tick
        tick_step = max(1, n_bins // 10)
        ax2.set_xticks(bins[::tick_step])

    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Add lift values as text annotations (only for smaller number of bins to avoid clutter)
    if n_bins <= 20:
        for i, (bin_num, lift) in enumerate(zip(bins, lifts)):
            ax2.text(bin_num, cumulative_tp_rates[i] + 0.01, f"Lift: {lift:.2f}", ha="center", va="bottom", fontsize=8, rotation=45)

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
    bin_stats = lift_data["bin_stats"]
    n_bins = lift_data["n_bins"]

    # Determine bin type name for labeling
    if n_bins == 10:
        bin_type = "Decile"
    elif n_bins == 100:
        bin_type = "Centile"
    elif n_bins == 20:
        bin_type = "Ventile"
    elif n_bins == 4:
        bin_type = "Quartile"
    else:
        bin_type = f"Bin (1/{n_bins})"

    with open(save_path, "w") as f:
        f.write("Lift Curve Analysis\n")
        f.write("=" * 50 + "\n\n")

        # Detailed table
        f.write(f"{bin_type} Analysis:\n")
        f.write("-" * (len(bin_type) + 10) + "\n")
        f.write(f"{bin_type:<8} {'Count':<8} {'TPs':<6} " f"{'TP Rate':<10} {'Cum TPs':<8} " f"{'Cum TP Rate':<12} {'Lift':<8}\n")
        f.write("-" * 70 + "\n")

        for stat in bin_stats:
            f.write(
                f"{stat['bin']:<8} {stat['count']:<8} "
                f"{stat['true_positives']:<6} {stat['tp_rate']:<10.4f} "
                f"{stat['cumulative_tps']:<8} "
                f"{stat['cumulative_tp_rate']:<12.4f} "
                f"{stat['lift']:<8.2f}\n"
            )

        f.write("\n")


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


def save_results_analysis(
    metrics: dict,
    args,
    results: dict,
    graph_info: dict,
    lift_data_deciles: dict,
    lift_data_centiles: dict,
    save_path: str,
):
    """Save comprehensive analysis including link prediction results and lift curves for both deciles and centiles."""
    with open(save_path, "w") as f:
        f.write("Link Prediction Results\n")
        f.write("=" * 80 + "\n\n")

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

        # LIFT CURVE ANALYSIS - DECILES
        f.write("LIFT CURVE ANALYSIS - DECILES (10 BINS)\n")
        f.write("=" * 50 + "\n\n")

        _write_lift_analysis_section(f, lift_data_deciles, args, "Decile", "Deciles")

        # LIFT CURVE ANALYSIS - CENTILES
        f.write("\n\nLIFT CURVE ANALYSIS - CENTILES (100 BINS)\n")
        f.write("=" * 50 + "\n\n")

        _write_lift_analysis_section(f, lift_data_centiles, args, "Centile", "Centiles")


def _write_lift_analysis_section(f, lift_data: dict, args, bin_type: str, bin_type_plural: str):
    """Helper function to write lift analysis section for a specific bin type."""
    bin_stats = lift_data["bin_stats"]
    n_bins = lift_data["n_bins"]

    # Detailed table - show all for deciles, top 20 for centiles
    f.write(f"{bin_type} Analysis:\n")
    f.write("-" * (len(bin_type) + 10) + "\n")
    f.write(f"{bin_type:<8} {'Count':<8} {'TPs':<6} {'TP Rate':<10} {'Cum TPs':<8} {'Cum TP Rate':<12} {'Lift':<8}\n")
    f.write("-" * 70 + "\n")

    # For centiles, show top 20 bins, then summary stats
    stats_to_show = bin_stats if n_bins <= 20 else bin_stats[:20]

    for stat in stats_to_show:
        f.write(
            f"{stat['bin']:<8} {stat['count']:<8} "
            f"{stat['true_positives']:<6} {stat['tp_rate']:<10.4f} "
            f"{stat['cumulative_tps']:<8} "
            f"{stat['cumulative_tp_rate']:<12.4f} "
            f"{stat['lift']:<8.2f}\n"
        )

    if n_bins > 20:
        f.write("... (showing top 20 bins only)\n")
        f.write(f"\nKey Statistics for All {n_bins} {bin_type_plural}:\n")
        f.write("-" * 30 + "\n")

        # Find top 5 performing bins
        top_5_bins = sorted(bin_stats, key=lambda x: x["lift"], reverse=True)[:5]
        f.write("Top 5 Performing Bins (by lift):\n")
        for i, stat in enumerate(top_5_bins, 1):
            lift_val = stat["lift"]
            tp_rate_val = stat["tp_rate"]
            f.write(f"  {i}. {bin_type} {stat['bin']}: " f"Lift {lift_val:.2f}, TP Rate {tp_rate_val:.4f}\n")

        f.write(f"\nBottom 5 {bin_type_plural} (lowest lift):\n")
        bottom_5_bins = sorted(bin_stats, key=lambda x: x["lift"])[:5]
        for i, stat in enumerate(bottom_5_bins, 1):
            lift_val = stat["lift"]
            tp_rate_val = stat["tp_rate"]
            f.write(f"  {i}. {bin_type} {stat['bin']}: " f"Lift {lift_val:.2f}, TP Rate {tp_rate_val:.4f}\n")

    f.write("\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Link prediction pipeline example")
    parser.add_argument(
        "--embedding_type",
        type=str,
        default="poincare_maps",
        choices=[
            "poincare_embeddings",
            "lorentz",
            "poincare_maps",
            "dmercator",
            "hydra",
            "hypermap",
            "hydra_plus",
            "rdpg",
        ],
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
        choices=[
            "ToggleSwitch",
            "AS",
            "Facebook",
            "Olsson",
            "MyeloidProgenitors",
            "krumsiek11_blobs",
            "Paul",
            "Cora",
            "AirportsUSA",
            "AirportsBrazil",
            "AirportsEurope",
        ],
        help="Dataset to use: ToggleSwitch (PoincareMaps), AS (Stanford SNAP Autonomous Systems), Facebook (Stanford SNAP), Cora, or Airports (USA/Brazil/Europe)",
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=10,
        help="Number of bins for lift curve analysis (10 for deciles, 100 for centiles, etc.)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible edge removal (default: None, non-deterministic)",
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
        snap_loader = SNAPDatasetLoader()
        graph, metadata = snap_loader.load_networkx("as")
    elif args.dataset == "Facebook":
        print("Loading Stanford SNAP Facebook dataset...")
        snap_loader = SNAPDatasetLoader()
        graph, metadata = snap_loader.load_networkx("facebook")
    elif args.dataset == "Cora":
        print("Loading CORA dataset...")
        with open("./data/Cora/cora_graph.pkl", "rb") as f:
            edge_list = pickle.load(f)
        with open("./data/Cora/cora_graph.json", "r") as f:
            graph_data = json.load(f)

        # Build networkx graph from edge list
        graph = nx.Graph()
        graph.add_edges_from(edge_list)
        metadata = {"labels": graph_data.get("y", None)}
    elif args.dataset == "AirportsUSA":
        print("Loading Airports USA dataset...")
        if Airports is None:
            raise ImportError("torch_geometric package is required. Install it with: pip install torch-geometric")
        dataset = Airports(root="./data/airports", name="USA")
        data = dataset[0]

        # Convert to NetworkX graph
        edge_index = data.edge_index.numpy()
        edges = [(int(edge_index[0, i]), int(edge_index[1, i])) for i in range(edge_index.shape[1])]

        graph = nx.Graph()
        graph.add_edges_from(edges)
        graph.add_nodes_from(range(data.num_nodes))

        # Extract labels if available
        labels = data.y.numpy() if data.y is not None else None
        metadata = {"labels": labels, "dataset_name": "AirportsUSA", "num_nodes": data.num_nodes, "num_edges": data.num_edges}
    elif args.dataset == "AirportsBrazil":
        print("Loading Airports Brazil dataset...")
        if Airports is None:
            raise ImportError("torch_geometric package is required. Install it with: pip install torch-geometric")
        dataset = Airports(root="./data/airports", name="Brazil")
        data = dataset[0]

        # Convert to NetworkX graph
        edge_index = data.edge_index.numpy()
        edges = [(int(edge_index[0, i]), int(edge_index[1, i])) for i in range(edge_index.shape[1])]

        graph = nx.Graph()
        graph.add_edges_from(edges)
        graph.add_nodes_from(range(data.num_nodes))

        # Extract labels if available
        labels = data.y.numpy() if data.y is not None else None
        metadata = {"labels": labels, "dataset_name": "AirportsBrazil", "num_nodes": data.num_nodes, "num_edges": data.num_edges}
    elif args.dataset == "AirportsEurope":
        print("Loading Airports Europe dataset...")
        if Airports is None:
            raise ImportError("torch_geometric package is required. Install it with: pip install torch-geometric")
        dataset = Airports(root="./data/airports", name="Europe")
        data = dataset[0]

        # Convert to NetworkX graph
        edge_index = data.edge_index.numpy()
        edges = [(int(edge_index[0, i]), int(edge_index[1, i])) for i in range(edge_index.shape[1])]

        graph = nx.Graph()
        graph.add_edges_from(edges)
        graph.add_nodes_from(range(data.num_nodes))

        # Extract labels if available
        labels = data.y.numpy() if data.y is not None else None
        metadata = {"labels": labels, "dataset_name": "AirportsEurope", "num_nodes": data.num_nodes, "num_edges": data.num_edges}
    else:
        print(f"Loading PoincareMaps dataset: {args.dataset}")
        loader = PoincareMapsLoader("models/PoincareMaps/datasets/")
        graph, metadata = loader.load_as_networkx(args.dataset, k_neighbors=15)

    # Store original graph information for results file
    graph_info = {"original_nodes": len(graph.nodes()), "original_edges": len(graph.edges())}

    # Adjust configurations based on dataset size
    num_nodes = len(graph.nodes())
    is_large_dataset = num_nodes > 1000  # AS dataset has 6474 nodes

    # Simulate link removal
    print("Simulating link removal...")
    results = simulate_link_removal(graph, q=args.q, seed=args.seed)
    omega_E = results["omega_E"]  # List of remaining links

    # Create graph with same nodes as original graph and edges from omega_E
    graph_from_omega_E = nx.Graph()
    graph_from_omega_E.add_nodes_from(graph.nodes())  # Preserve all original nodes
    graph_from_omega_E.add_edges_from(omega_E)  # Add only remaining edges

    print(f"Graph created from omega_E: {len(graph_from_omega_E.nodes)} nodes, {len(graph_from_omega_E.edges)} edges")

    # Train embeddings (hyperbolic or RDPG)
    if args.embedding_type == "rdpg":
        print("Computing RDPG embeddings...")
        adjacency_matrix = nx.to_numpy_array(graph_from_omega_E)

        (w, v) = scipy.sparse.linalg.eigs(adjacency_matrix, k=10, which="LM")

        wabs = np.array(list(zip(-np.abs(w), -np.sign(np.real(w)))), dtype=[("abs", "f4"), ("sign", "i4")])
        w = w[np.argsort(wabs, order=["abs", "sign"])]
        print(f"Eigenvalues: {w}")

        d = 2  # Default dimension for RDPG
        ase = AdjacencySpectralEmbed(n_components=d, diag_aug=True, algorithm="full")
        embeddings = ase.fit_transform(adjacency_matrix)
        print(f"RDPG embeddings shape: {embeddings.shape}")
        native_space = "euclidean"
        embedding_runner = None
        poincare_embeddings = None  # Skip Poincaré visualization for RDPG
        print("Skipping Poincaré visualization for RDPG embeddings")
    else:
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
                "hydra_plus": {
                    "dim": 2,
                    "device": None,
                    "use_gpu": False,
                    "seed": args.seed,
                },
            }
        else:
            configurations = {
                "poincare_embeddings": {"dim": 2, "negs": 5, "epochs": 1000, "batch_size": 256, "dimension": 1},
                "lorentz": {"dim": 2, "epochs": 10000, "batch_size": 1024, "num_nodes": len(graph.nodes)},
                "dmercator": {"dim": 1},
                "hydra": {"dim": 2},
                "poincare_maps": {"dim": 2, "epochs": 1000},
                "hypermap": {"dim": 3},
                "hydra_plus": {
                    "dim": 2,
                    "device": None,
                    "use_gpu": False,
                    "seed": args.seed,
                },
            }
        config = configurations[args.embedding_type]
        embedding_runner = HyperbolicEmbeddings(embedding_type=args.embedding_type, config=config)

        native_space = embedding_runner.model.native_space
        max_edges = 2000 if is_large_dataset else 1000

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

    if args.embedding_type == "rdpg":
        # Compute probability matrix for RDPG
        print("Computing RDPG probability matrix...")
        probabilities = np.dot(embeddings, embeddings.T)
        print(f"Probability matrix shape: {probabilities.shape}")
        print(f"Probability range: {probabilities[probabilities > 0].min():.4f} to {probabilities.max():.4f}")
        predicted_links, sorted_all_candidates = predict_links_from_probabilities(
            probabilities, n_links=n_links, candidate_edges=omega_R_plus_N, return_sorted_candidates=True
        )
    else:
        predicted_links, sorted_all_candidates = predict_links(
            distances, n_links=n_links, candidate_edges=omega_R_plus_N, return_sorted_candidates=True
        )
    print(f"Top 10 sorted links: {sorted_all_candidates[:10]}")

    # Evaluate predictions against removed edges (Ω_R) - the true positive targets
    print()
    print("Evaluating predictions...")
    metrics = evaluate_predictions(predicted_links, omega_R, omega_N)

    # Perform lift curve analysis
    print("\nPerforming lift curve analysis...")

    # Compute lift curve statistics for both deciles and centiles
    print("Computing decile analysis (10 bins)...")
    is_rdpg = args.embedding_type == "rdpg"
    lift_data_deciles = compute_lift_curve(sorted_all_candidates, omega_R, n_bins=10, descending=is_rdpg)

    print("Computing centile analysis (100 bins)...")
    lift_data_centiles = compute_lift_curve(sorted_all_candidates, omega_R, n_bins=100, descending=is_rdpg)

    # Generate lift curve visualizations for both
    print("Generating visualizations...")

    lift_plot_deciles_path = f"{RESULTS_PATH}/{args.embedding_type}_lift_curve_deciles.pdf"
    plot_lift_curve(lift_data_deciles, args.dataset, args.embedding_type, lift_plot_deciles_path)

    lift_plot_centiles_path = f"{RESULTS_PATH}/{args.embedding_type}_lift_curve_centiles.pdf"
    plot_lift_curve(lift_data_centiles, args.dataset, args.embedding_type, lift_plot_centiles_path)

    print(f"Decile lift curve plot saved to: {lift_plot_deciles_path}")
    print(f"Centile lift curve plot saved to: {lift_plot_centiles_path}")

    # Print comprehensive lift analysis summary
    print("\nLIFT CURVE SUMMARY:")
    decile_stats = lift_data_deciles["bin_stats"]
    centile_stats = lift_data_centiles["bin_stats"]
    baseline_tp = lift_data_deciles["baseline_tp_rate"]

    print(
        f"  Deciles:  Lift {decile_stats[0]['lift']:.2f}, TP rate {decile_stats[0]['tp_rate']:.4f}, "
        f"Top 3 cum: {decile_stats[2]['cumulative_tp_rate']:.4f}"
    )
    print(
        f"  Centiles: Lift {centile_stats[0]['lift']:.2f}, TP rate {centile_stats[0]['tp_rate']:.4f}, "
        f"Top 3 cum: {centile_stats[2]['cumulative_tp_rate']:.4f}"
    )
    print(f"  Baseline TP rate: {baseline_tp:.4f} " f"({lift_data_deciles['total_candidates']} candidates)")

    # Print results
    print("\nLink Prediction Results:")
    print(f"Predicted {len(predicted_links)} links from {len(omega_R_plus_N)} candidates (Ω_R + Ω_N)")
    print(f"Target removed edges (Ω_R): {len(omega_R)} links")
    print(f"True non-links (Ω_N): {len(results['omega_N'])} pairs")
    print("\nMetrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Save results analysis to file
    results_path = f"{RESULTS_PATH}/{args.embedding_type}_result_analysis.txt"
    save_results_analysis(metrics, args, results, graph_info, lift_data_deciles, lift_data_centiles, results_path)
    print(f"\nResults analysis saved to: {results_path}")

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

    true_links = []
    if args.embedding_type == "rdpg":
        for u, v in omega_R:
            prob = probabilities[u, v]
            true_links.append((u, v, prob))
    else:
        for u, v in omega_R:
            dist = distances[u, v]
            true_links.append((u, v, dist))

    # Create a visualization showing predicted links
    print("\nCreating visualization...")

    # Convert embeddings to Poincaré for better visualization (skip for RDPG)
    if args.embedding_type != "rdpg":
        if native_space != "poincare":
            poincare_embeddings = convert_coordinates(hyperboloid_embeddings, "hyperboloid", "poincare")
        else:
            poincare_embeddings = embeddings

        plot_title = "Predicted Links in Poincaré Disk"
        save_path = f"{PATH}/{args.embedding_type}_omega_E_predicted_links.pdf"
        plot_embeddings(poincare_embeddings, embedding_runner, omega_E, recovered_links, false_positives, plot_title, save_path, max_edges)
    else:
        print("Skipping Poincaré visualization for RDPG embeddings")

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
