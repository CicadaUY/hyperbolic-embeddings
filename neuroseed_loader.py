"""
NeuroSEED Dataset Loader

This module provides a loader for the NeuroSEED dataset, which contains
pre-computed hyperbolic embeddings and labels for various biological datasets.

The NeuroSEED dataset is from the paper:
"NeuroSEED: Geometric Deep Learning for Sequence-to-Sequence Tasks"
https://github.com/gcorso/NeuroSEED
"""

import os
import pickle
from typing import Tuple

import networkx as nx
import numpy as np
import torch


class NeuroSEEDLoader:
    """
    Loader for NeuroSEED dataset.
    
    The NeuroSEED dataset provides pre-computed hyperbolic embeddings
    for biological sequence data. For the purposes of KNN classification,
    we need to construct a graph from these embeddings.
    """

    def __init__(self, data_dir: str = "./data/neuroseed"):
        """
        Initialize the NeuroSEED loader.

        Parameters:
        -----------
        data_dir : str
            Directory containing the NeuroSEED dataset files
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def load_embeddings_and_labels(
        self, 
        split: str = "train",
        task: str = "edit_distance",
        num_samples: int = None,
        seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load pre-computed embeddings and labels from NeuroSEED dataset.

        Parameters:
        -----------
        split : str
            Dataset split to load ("train", "val", or "test")
        task : str
            Task name (e.g., "edit_distance", "closest_string")
        num_samples : int
            Number of samples to load (None for all)
        seed : int
            Random seed for sampling

        Returns:
        --------
        embeddings : np.ndarray
            Pre-computed embeddings (N, D)
        labels : np.ndarray
            Labels (N,)
        """
        # Try to load pre-computed embeddings
        embeddings_path = os.path.join(self.data_dir, task, f"{split}_embeddings.npy")
        labels_path = os.path.join(self.data_dir, task, f"{split}_labels.npy")

        if os.path.exists(embeddings_path) and os.path.exists(labels_path):
            embeddings = np.load(embeddings_path)
            labels = np.load(labels_path)
            print(f"Loaded {len(embeddings)} samples from {embeddings_path}")
        else:
            print(f"Warning: Could not find pre-computed embeddings at {embeddings_path}")
            print("Attempting to load raw data and generate synthetic embeddings...")
            embeddings, labels = self._generate_synthetic_data(num_samples or 1000, seed)

        # Sample if requested
        if num_samples is not None and num_samples < len(embeddings):
            np.random.seed(seed)
            indices = np.random.choice(len(embeddings), num_samples, replace=False)
            embeddings = embeddings[indices]
            labels = labels[indices]

        return embeddings, labels

    def _generate_synthetic_data(
        self, 
        num_samples: int = 1000, 
        seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic hyperbolic embeddings for testing purposes.

        This is a fallback when the actual NeuroSEED dataset is not available.
        It generates random embeddings in the Poincaré disk with clustered labels.

        Parameters:
        -----------
        num_samples : int
            Number of samples to generate
        seed : int
            Random seed

        Returns:
        --------
        embeddings : np.ndarray
            Synthetic embeddings in Poincaré disk (N, 2)
        labels : np.ndarray
            Synthetic labels (N,)
        """
        np.random.seed(seed)
        print(f"Generating {num_samples} synthetic hyperbolic embeddings...")

        # Generate embeddings in Poincaré disk (2D for visualization)
        # Create clusters in hyperbolic space
        num_classes = 4
        samples_per_class = num_samples // num_classes

        embeddings_list = []
        labels_list = []

        # Create clusters at different positions in the Poincaré disk
        cluster_centers = [
            np.array([0.3, 0.3]),
            np.array([-0.3, 0.3]),
            np.array([0.3, -0.3]),
            np.array([-0.3, -0.3]),
        ]

        for class_idx in range(num_classes):
            center = cluster_centers[class_idx]
            # Generate points around the cluster center
            for _ in range(samples_per_class):
                # Add Gaussian noise in tangent space
                noise = np.random.randn(2) * 0.15
                point = center + noise
                # Project back to Poincaré disk (norm < 1)
                norm = np.linalg.norm(point)
                if norm >= 0.95:
                    point = point / norm * 0.95
                embeddings_list.append(point)
                labels_list.append(class_idx)

        embeddings = np.array(embeddings_list)
        labels = np.array(labels_list)

        # Shuffle
        indices = np.random.permutation(len(embeddings))
        embeddings = embeddings[indices]
        labels = labels[indices]

        print(f"Generated synthetic data with {num_classes} classes")
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Labels distribution: {np.bincount(labels)}")

        return embeddings, labels

    def embeddings_to_graph(
        self, 
        embeddings: np.ndarray,
        labels: np.ndarray,
        k_neighbors: int = 10,
        distance_metric: str = "poincare"
    ) -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
        """
        Convert embeddings to a k-NN graph.

        Parameters:
        -----------
        embeddings : np.ndarray
            Node embeddings (N, D)
        labels : np.ndarray
            Node labels (N,)
        k_neighbors : int
            Number of nearest neighbors for graph construction
        distance_metric : str
            Distance metric to use ("poincare" or "euclidean")

        Returns:
        --------
        graph : nx.Graph
            NetworkX graph constructed from k-NN
        labels : np.ndarray
            Node labels
        node_indices : np.ndarray
            Node indices
        """
        print(f"Converting embeddings to k-NN graph (k={k_neighbors})...")

        num_nodes = len(embeddings)
        graph = nx.Graph()
        graph.add_nodes_from(range(num_nodes))

        # Compute pairwise distances
        if distance_metric == "poincare":
            distances = self._compute_poincare_distances(embeddings)
        else:
            from sklearn.metrics.pairwise import euclidean_distances
            distances = euclidean_distances(embeddings)

        # For each node, connect to k nearest neighbors
        for i in range(num_nodes):
            # Get k+1 nearest neighbors (excluding self)
            nearest_indices = np.argsort(distances[i])[1 : k_neighbors + 1]
            for j in nearest_indices:
                graph.add_edge(i, j)

        node_indices = np.arange(num_nodes)

        print(f"Graph created: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Check connectivity
        if not nx.is_connected(graph):
            num_components = nx.number_connected_components(graph)
            largest_cc = max(nx.connected_components(graph), key=len)
            print(f"Warning: Graph has {num_components} connected components")
            print(f"Largest component has {len(largest_cc)} nodes ({100*len(largest_cc)/num_nodes:.1f}%)")

        return graph, labels, node_indices

    def _compute_poincare_distances(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise Poincaré distances between embeddings.

        Parameters:
        -----------
        embeddings : np.ndarray
            Embeddings in Poincaré disk (N, D)

        Returns:
        --------
        distances : np.ndarray
            Pairwise distance matrix (N, N)
        """
        # Import here to avoid circular dependency
        from utils.geometric_conversions import HyperbolicConversions

        distances = HyperbolicConversions.compute_distances(embeddings, space="poincare")
        return distances

    def load_as_networkx(
        self,
        split: str = "train",
        task: str = "edit_distance",
        num_samples: int = None,
        k_neighbors: int = 10,
        seed: int = 42,
    ) -> Tuple[nx.Graph, np.ndarray, np.ndarray, dict]:
        """
        Load NeuroSEED data and convert to NetworkX graph.

        Parameters:
        -----------
        split : str
            Dataset split ("train", "val", or "test")
        task : str
            Task name
        num_samples : int
            Number of samples to load
        k_neighbors : int
            Number of neighbors for k-NN graph construction
        seed : int
            Random seed

        Returns:
        --------
        graph : nx.Graph
            NetworkX graph
        labels : np.ndarray
            Node labels
        node_indices : np.ndarray
            Node indices
        metadata : dict
            Dataset metadata
        """
        # Load embeddings and labels
        embeddings, labels = self.load_embeddings_and_labels(
            split=split,
            task=task,
            num_samples=num_samples,
            seed=seed
        )

        # Convert to graph
        graph, labels, node_indices = self.embeddings_to_graph(
            embeddings,
            labels,
            k_neighbors=k_neighbors,
            distance_metric="poincare"
        )

        # Create metadata
        metadata = {
            "dataset_name": f"neuroseed_{task}",
            "split": split,
            "num_samples": len(labels),
            "num_classes": len(np.unique(labels)),
            "k_neighbors": k_neighbors,
            "embedding_dim": embeddings.shape[1],
        }

        return graph, labels, node_indices, metadata


def load_neuroseed_dataset(
    task: str = "edit_distance",
    num_samples: int = 1000,
    k_neighbors: int = 10,
    seed: int = 42,
    data_dir: str = "./data/neuroseed",
    use_predefined_splits: bool = False,
) -> Tuple[nx.Graph, np.ndarray, np.ndarray] | Tuple[nx.Graph, nx.Graph, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to load NeuroSEED dataset.

    Parameters:
    -----------
    task : str
        Task name (e.g., "edit_distance")
    num_samples : int
        Number of samples to load (only used if use_predefined_splits=False)
    k_neighbors : int
        Number of neighbors for k-NN graph construction
    seed : int
        Random seed
    data_dir : str
        Directory containing NeuroSEED data
    use_predefined_splits : bool
        If True, loads both train and test splits separately and returns both.
        If False, loads synthetic data for the whole dataset (for testing).

    Returns:
    --------
    If use_predefined_splits=False (default):
        graph : nx.Graph
            NetworkX graph representation
        labels : np.ndarray
            Node labels
        node_indices : np.ndarray
            Node indices
    
    If use_predefined_splits=True:
        train_graph : nx.Graph
            Training graph
        test_graph : nx.Graph
            Testing graph
        train_labels : np.ndarray
            Training labels
        test_labels : np.ndarray
            Testing labels
        train_indices : np.ndarray
            Training node indices
        test_indices : np.ndarray
            Testing node indices
    """
    loader = NeuroSEEDLoader(data_dir=data_dir)
    
    if use_predefined_splits:
        # Load train and test splits separately
        print("\nLoading NeuroSEED with pre-defined train/test splits...")
        
        # Load training data
        train_graph, train_labels, train_indices, train_metadata = loader.load_as_networkx(
            split="train",
            task=task,
            num_samples=None,  # Load all available training data
            k_neighbors=k_neighbors,
            seed=seed
        )
        
        # Load test data
        test_graph, test_labels, test_indices, test_metadata = loader.load_as_networkx(
            split="test",
            task=task,
            num_samples=None,  # Load all available test data
            k_neighbors=k_neighbors,
            seed=seed
        )
        
        print(f"\nNeuroSEED Dataset Loaded (Pre-defined Splits):")
        print(f"  Task: {train_metadata['dataset_name']}")
        print(f"  Training samples: {train_metadata['num_samples']}")
        print(f"  Test samples: {test_metadata['num_samples']}")
        print(f"  Classes: {train_metadata['num_classes']}")
        print(f"  Embedding dimension: {train_metadata['embedding_dim']}")
        
        return train_graph, test_graph, train_labels, test_labels, train_indices, test_indices
    
    else:
        # Load single dataset (synthetic or specified split)
        # This is for testing/development with synthetic data
        graph, labels, node_indices, metadata = loader.load_as_networkx(
            split="train",  # Default to train, but will use synthetic data anyway
            task=task,
            num_samples=num_samples,
            k_neighbors=k_neighbors,
            seed=seed
        )
        
        print(f"\nNeuroSEED Dataset Loaded (Synthetic/Single Split):")
        print(f"  Task: {metadata['dataset_name']}")
        print(f"  Samples: {metadata['num_samples']}")
        print(f"  Classes: {metadata['num_classes']}")
        print(f"  Embedding dimension: {metadata['embedding_dim']}")
        print(f"  Note: Use --use_predefined_splits to load real train/test splits")

        return graph, labels, node_indices

