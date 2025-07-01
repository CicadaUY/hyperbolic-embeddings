import argparse
import os

import networkx as nx
import numpy as np

from hyperbolic_embeddings import HyperbolicEmbeddings


def main():
    parser = argparse.ArgumentParser(description="Train and plot hyperbolic embeddings for the Karate Club graph.")
    parser.add_argument(
        "--embedding_type",
        type=str,
        required=True,
        choices=["poincare_embeddings", "lorentz", "dmercator", "hydra", "poincare_maps"],
        help="Type of embedding model to use.",
    )
    parser.add_argument("--model_dir", type=str, default="saved_models/tree_test", help="Directory to save the trained model.")
    parser.add_argument("--plot_dir", type=str, default="test/tree_test/plots", help="Directory to save the plots.")

    args = parser.parse_args()

    # Build tree graph
    G = nx.balanced_tree(2, 4)

    lengths = dict(nx.all_pairs_shortest_path_length(G))
    n = G.number_of_nodes()
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = lengths[i][j]
    A = nx.to_numpy_array(G)

    edge_list = list(G.edges())
    labels = list(range(n))

    # Embedding configs
    configurations = {
        "poincare_embeddings": {"dim": 2, "negs": 5, "epochs": 1000, "batch_size": 256, "dimension": 1},
        "lorentz": {"dim": 2, "epochs": 50000, "batch_size": 256},
        "dmercator": {"dim": 1},
        "hydra": {"dim": 2},
        "poincare_maps": {"dim": 2, "epochs": 1000},
    }

    embedding_type = args.embedding_type
    if embedding_type not in configurations:
        raise ValueError(f"Unknown embedding type: {embedding_type}")

    config = configurations[embedding_type]
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    model_path = os.path.join(args.model_dir, f"{embedding_type}_embeddings.bin")
    plot_path = os.path.join(args.plot_dir, f"{embedding_type}_embeddings.pdf")

    print(f"Training {embedding_type} embeddings...")
    embedding_runner = HyperbolicEmbeddings(embedding_type=embedding_type, config=config)

    if embedding_type == "hydra" or "poincare_maps":
        embedding_runner.train(adjacency_matrix=A, model_path=model_path)
    else:
        embedding_runner.train(edge_list=edge_list, model_path=model_path)

    embedding_runner.plot_embeddings(labels=labels, edge_list=edge_list, save_path=plot_path)


if __name__ == "__main__":
    main()
