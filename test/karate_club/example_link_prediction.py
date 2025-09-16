#!/usr/bin/env python3
"""
Example script demonstrating the link prediction pipeline.

This script shows how to use the HyperbolicEmbeddings class
to perform link prediction on a test tree graph.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from hyperbolic_embeddings import HyperbolicEmbeddings
from utils.geometric_conversions import compute_distances, convert_coordinates


def create_test_tree_graph(branching_factor: int = 2, depth: int = 4) -> nx.Graph:
    """Create a balanced tree graph for testing."""
    return nx.balanced_tree(branching_factor, depth)


def predict_links(distances: np.ndarray, n_links: int = 10, exclude_existing: bool = True, graph: nx.Graph = None) -> list:
    """Predict links based on hyperbolic distances."""
    # Create a copy of the distance matrix
    dist_matrix = distances.copy()

    # Set diagonal to infinity to exclude self-loops
    np.fill_diagonal(dist_matrix, np.inf)

    # Exclude existing edges if requested
    if exclude_existing and graph is not None:
        for u, v in graph.edges():
            dist_matrix[u, v] = np.inf
            dist_matrix[v, u] = np.inf

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


def evaluate_predictions(predicted_links: list, test_edges: list) -> dict:
    """Evaluate link prediction performance."""
    # Create sets for faster lookup
    test_edges_set = set(test_edges)
    test_edges_set.update([(v, u) for u, v in test_edges])  # Add both directions

    # Extract predicted node pairs
    predicted_pairs = [(u, v) for u, v, _ in predicted_links]

    # Calculate metrics
    true_positives = sum(1 for pair in predicted_pairs if pair in test_edges_set)
    false_positives = len(predicted_pairs) - true_positives
    false_negatives = len(test_edges) - true_positives

    precision = true_positives / len(predicted_pairs) if predicted_pairs else 0
    recall = true_positives / len(test_edges) if test_edges else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

    return metrics


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
    parser.add_argument("--n_links", type=int, default=None, help="Number of links to predict (default: same as number of edges in graph)")
    return parser.parse_args()


def main():
    """Run a simple example of the link prediction pipeline."""
    args = parse_args()

    print("Link Prediction Pipeline Example")
    print("=" * 40)
    print(f"Embedding type: {args.embedding_type}")
    print(f"Number of links to predict: {args.n_links if args.n_links else 'same as graph edges'}")
    print()

    # Load Karate Club graph
    graph = nx.karate_club_graph()
    edge_list = list(graph.edges())
    A = nx.to_numpy_array(graph)
    labels = [graph.nodes[n]["club"] for n in sorted(graph.nodes())]

    # Visualize the original graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph)

    # Color nodes by club membership
    node_colors = ["lightblue" if label == "Mr. Hi" else "orange" for label in labels]
    nx.draw(graph, pos, with_labels=True, node_color=node_colors, node_size=500, font_size=10, font_weight="bold")
    plt.title("Original Karate Club Graph")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [Patch(facecolor="lightblue", label="Mr. Hi"), Patch(facecolor="orange", label="Officer")]
    plt.legend(handles=legend_elements)
    plt.show()

    # Train hyperbolic embeddings using the real library
    print("Training hyperbolic embeddings...")
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
    edge_list = list(graph.edges())
    adjacency_matrix = nx.to_numpy_array(graph)

    # Train the model
    model_path = f"saved_models/{args.embedding_type}_model.bin"
    os.makedirs("saved_models", exist_ok=True)
    embedding_runner.train(adjacency_matrix=adjacency_matrix, model_path=model_path)

    # Get embeddings
    embeddings = embedding_runner.get_all_embeddings(model_path)
    print(f"Embeddings shape: {embeddings.shape}")

    # Convert to hyperboloid coordinates if needed
    native_space = embedding_runner.model.native_space
    if native_space != "hyperboloid":
        print(f"Converting embeddings from {native_space} to hyperboloid coordinates")
        embeddings = convert_coordinates(embeddings, native_space, "hyperboloid")

    # Compute hyperbolic distances
    print("Computing hyperbolic distances...")
    distances = compute_distances(embeddings, space="hyperboloid")
    print(f"Distance matrix shape: {distances.shape}")
    print(f"Distance range: {distances[distances > 0].min():.4f} to {distances.max():.4f}")
    # print(f"Distance matrix: {distances}")

    # Get all edges in the graph for comparison
    all_edges = list(graph.edges())
    print(f"Total edges in graph: {len(all_edges)}")

    # Determine number of links to predict
    n_links = args.n_links if args.n_links is not None else len(all_edges)
    print(f"Predicting {n_links} links...")

    # Predict links
    predicted_links = predict_links(distances, n_links=n_links, exclude_existing=False, graph=graph)

    # Evaluate predictions against all existing edges
    print("Evaluating predictions...")
    metrics = evaluate_predictions(predicted_links, all_edges)

    # Print results
    print("\nResults Metrics:")

    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    print(f"\nAccuracy: {metrics['precision']:.4f} ({metrics['true_positives']}/{len(predicted_links)} predicted links were correct)")

    # Count correct vs incorrect predictions
    correct_predictions = []
    incorrect_predictions = []
    all_edges_set = set(all_edges)
    all_edges_set.update([(v, u) for u, v in all_edges])  # Add both directions

    for u, v, dist in predicted_links:
        if (u, v) in all_edges_set:
            correct_predictions.append((u, v, dist))
        else:
            incorrect_predictions.append((u, v, dist))

    print(f"\nCorrect predictions: {len(correct_predictions)}")
    print(f"Incorrect predictions: {len(incorrect_predictions)}")

    # print("\nCorrect predictions (node1, node2, distance):")
    # for u, v, dist in correct_predictions:
    #     print(f"  ({u}, {v}): {dist:.4f}")

    # print("\nIncorrect predictions (node1, node2, distance):")
    # for u, v, dist in incorrect_predictions:
    #     print(f"  ({u}, {v}): {dist:.4f}")

    # Create a visualization showing predicted links
    print("\nCreating visualization...")

    # Convert embeddings to Poincaré for better visualization
    poincare_embeddings = convert_coordinates(embeddings, "hyperboloid", "poincare")

    # Create a comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Original graph
    pos = nx.spring_layout(graph)
    node_colors = ["lightblue" if label == "Mr. Hi" else "orange" for label in labels]
    nx.draw(graph, pos, ax=axes[0], with_labels=True, node_color=node_colors, node_size=500, font_size=10, font_weight="bold")
    axes[0].set_title("Original Karate Club Graph")

    # Embeddings with predictions (in Poincaré disk)
    x, y = poincare_embeddings[:, 0], poincare_embeddings[:, 1]
    scatter_colors = ["lightblue" if label == "Mr. Hi" else "orange" for label in labels]
    axes[1].scatter(x, y, s=100, alpha=0.7, c=scatter_colors)

    # Add node labels
    for i, (xi, yi) in enumerate(zip(x, y)):
        axes[1].annotate(str(i), (xi, yi), xytext=(5, 5), textcoords="offset points", fontsize=8)

    # Plot original edges in gray
    for u, v in graph.edges():
        if u < len(x) and v < len(x):
            axes[1].plot([x[u], x[v]], [y[u], y[v]], "gray", alpha=0.3, linewidth=0.5)

    # Create sets for faster lookup
    all_edges_set = set(all_edges)
    all_edges_set.update([(v, u) for u, v in all_edges])  # Add both directions

    # Highlight predicted links - green if correct, red if incorrect
    pred_edges = [(u, v) for u, v, _ in predicted_links]
    for u, v in pred_edges:
        if u < len(x) and v < len(x):
            # Check if this predicted link actually exists in the original graph
            if (u, v) in all_edges_set:
                # Correct prediction - green dashed
                axes[1].plot([x[u], x[v]], [y[u], y[v]], "green", alpha=0.8, linewidth=2)
            else:
                # Incorrect prediction - red dashed
                axes[1].plot([x[u], x[v]], [y[u], y[v]], "red", alpha=0.8, linewidth=2)

    # Draw Poincaré disk boundary
    circle = plt.Circle((0, 0), 1, fill=False, color="black", linestyle="--", alpha=0.5)
    axes[1].add_patch(circle)
    axes[1].set_xlim(-1.1, 1.1)
    axes[1].set_ylim(-1.1, 1.1)
    axes[1].set_aspect("equal")

    axes[1].set_title("Hyperbolic Embeddings (Poincaré Disk) and Link Predictions")
    axes[1].set_xlabel("X (first spatial coordinate)")
    axes[1].set_ylabel("Y (second spatial coordinate)")
    axes[1].grid(True, alpha=0.3)

    # Add legend for both predictions and node colors
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_elements = [
        Line2D([0], [0], color="green", linewidth=2, label="Correct predictions"),
        Line2D([0], [0], color="red", linewidth=2, label="Incorrect predictions"),
        Patch(facecolor="lightblue", label="Mr. Hi"),
        Patch(facecolor="orange", label="Officer"),
    ]
    axes[1].legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(f"test/karate_club/plots/{args.embedding_type}_link_prediction_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.show()

    # Also create a separate embedding plot
    plt.figure(figsize=(10, 10))
    scatter_colors = ["lightblue" if label == "Mr. Hi" else "orange" for label in labels]
    plt.scatter(x, y, s=100, alpha=0.7, c=scatter_colors)

    # Add node labels
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.annotate(str(i), (xi, yi), xytext=(5, 5), textcoords="offset points", fontsize=8)

    # Plot original edges in gray
    for u, v in graph.edges():
        if u < len(x) and v < len(x):
            plt.plot([x[u], x[v]], [y[u], y[v]], "gray", alpha=0.3, linewidth=0.5)

    # Create sets for faster lookup
    all_edges_set = set(all_edges)
    all_edges_set.update([(v, u) for u, v in all_edges])  # Add both directions

    # Highlight predicted links - green if correct, red if incorrect
    for u, v in pred_edges:
        if u < len(x) and v < len(x):
            # Check if this predicted link actually exists in the original graph
            if (u, v) in all_edges_set:
                # Correct prediction - green solid
                plt.plot([x[u], x[v]], [y[u], y[v]], "green", alpha=0.8, linewidth=2)
            else:
                # Incorrect prediction - red solid
                plt.plot([x[u], x[v]], [y[u], y[v]], "red", alpha=0.8, linewidth=2)

    # Draw Poincaré disk boundary
    circle = plt.Circle((0, 0), 1, fill=False, color="black", linestyle="--", alpha=0.5)
    plt.gca().add_patch(circle)
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.gca().set_aspect("equal")

    plt.title("Hyperbolic Embeddings and Link Predictions")
    plt.xlabel("X (first spatial coordinate)")
    plt.ylabel("Y (second spatial coordinate)")
    plt.grid(True, alpha=0.3)

    # Add legend for both predictions and node colors
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_elements = [
        Line2D([0], [0], color="green", linewidth=2, label="Correct predictions"),
        Line2D([0], [0], color="red", linewidth=2, label="Incorrect predictions"),
        Patch(facecolor="lightblue", label="Mr. Hi"),
        Patch(facecolor="orange", label="Officer"),
    ]
    plt.legend(handles=legend_elements)

    plt.savefig(f"test/karate_club/plots/{args.embedding_type}_link_prediction_embeddings.pdf", dpi=300, bbox_inches="tight")
    plt.show()

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
