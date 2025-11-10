"""
AS Dataset Loader for Stanford SNAP Autonomous Systems Dataset

This module provides functionality to download, parse, and load the Stanford SNAP
Autonomous Systems dataset (January 2, 2000) as NetworkX graphs.

Dataset Information:
- Source: https://snap.stanford.edu/data/as.html
- File: as20000102.txt.gz
- Expected Statistics:
  - Nodes: 6474
  - Edges: 12572 (13895 total - 1323 self-loops)
  - Format: Tab-separated edge list (node1\tnode2)
"""

import gzip
import os
import urllib.request
from typing import Optional, Tuple

import networkx as nx


class ASDatasetLoader:
    """
    Loader for the Stanford SNAP Autonomous Systems dataset.

    This class handles downloading, caching, parsing, and converting the AS dataset
    into NetworkX graph format with proper validation.
    """

    # Dataset URL and expected statistics
    DATASET_URL = "https://snap.stanford.edu/data/as20000102.txt.gz"
    EXPECTED_NODES = 6474
    EXPECTED_EDGES = 12572  # Actual valid edges (13895 total - 1323 self-loops)

    def __init__(self, cache_dir: str = "data/as_dataset"):
        """
        Initialize the AS dataset loader.

        Args:
            cache_dir: Directory to cache downloaded dataset files
        """
        self.cache_dir = cache_dir
        self.dataset_filename = "as20000102.txt.gz"
        self.dataset_path = os.path.join(cache_dir, self.dataset_filename)

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

    def download_dataset(self, force_download: bool = False) -> str:
        """
        Download the AS dataset if not already cached.

        Args:
            force_download: If True, download even if file exists

        Returns:
            Path to the downloaded dataset file

        Raises:
            Exception: If download fails
        """
        if os.path.exists(self.dataset_path) and not force_download:
            print(f"Dataset already cached at: {self.dataset_path}")
            return self.dataset_path

        print(f"Downloading AS dataset from: {self.DATASET_URL}")
        try:
            urllib.request.urlretrieve(self.DATASET_URL, self.dataset_path)
            print(f"Dataset downloaded successfully to: {self.dataset_path}")
            return self.dataset_path
        except Exception as e:
            raise Exception(f"Failed to download dataset: {str(e)}")

    def parse_dataset(self, file_path: Optional[str] = None) -> nx.Graph:
        """
        Parse the AS dataset file and create a NetworkX graph.

        Args:
            file_path: Path to the dataset file (uses cached file if None)

        Returns:
            NetworkX Graph object

        Raises:
            FileNotFoundError: If dataset file doesn't exist
            Exception: If parsing fails
        """
        if file_path is None:
            file_path = self.dataset_path

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        print(f"Parsing AS dataset from: {file_path}")

        # Create empty graph
        graph = nx.Graph()

        try:
            # Open and parse the gzipped file
            with gzip.open(file_path, "rt") as f:
                edge_count = 0
                for line_num, line in enumerate(f, 1):
                    line = line.strip()

                    # Skip empty lines and comments
                    if not line or line.startswith("#"):
                        continue

                    # Parse edge (tab-separated node IDs)
                    parts = line.split("\t")
                    if len(parts) != 2:
                        # Fallback to space-separated if tab-separated fails
                        parts = line.split()
                        if len(parts) != 2:
                            print(f"Warning: Skipping malformed line {line_num}: {line}")
                            continue

                    try:
                        node1, node2 = int(parts[0]), int(parts[1])

                        # Skip self-loops
                        if node1 == node2:
                            continue

                        # Add edge to graph
                        graph.add_edge(node1, node2)
                        edge_count += 1

                    except ValueError:
                        print(f"Warning: Invalid node IDs on line {line_num}: {line}")
                        continue

                print(f"Parsed {edge_count} edges from dataset")

        except Exception as e:
            raise Exception(f"Failed to parse dataset: {str(e)}")

        return graph

    def validate_graph(self, graph: nx.Graph) -> bool:
        """
        Validate the loaded graph against expected statistics.

        Args:
            graph: NetworkX graph to validate

        Returns:
            True if validation passes, False otherwise
        """
        num_nodes = len(graph.nodes())
        num_edges = len(graph.edges())

        print("Graph Statistics:")
        print(f"  Nodes: {num_nodes} (expected: {self.EXPECTED_NODES})")
        print(f"  Edges: {num_edges} (expected: {self.EXPECTED_EDGES})")

        # Check if statistics match expected values
        nodes_match = num_nodes == self.EXPECTED_NODES
        edges_match = num_edges == self.EXPECTED_EDGES

        if nodes_match and edges_match:
            print("✓ Graph validation passed!")
            return True
        else:
            print("✗ Graph validation failed!")
            if not nodes_match:
                print(f"  Node count mismatch: got {num_nodes}, expected {self.EXPECTED_NODES}")
            if not edges_match:
                print(f"  Edge count mismatch: got {num_edges}, expected {self.EXPECTED_EDGES}")
            return False

    def get_graph_info(self, graph: nx.Graph) -> dict:
        """
        Get detailed information about the graph.

        Args:
            graph: NetworkX graph

        Returns:
            Dictionary with graph statistics
        """
        info = {
            "nodes": len(graph.nodes()),
            "edges": len(graph.edges()),
            "is_connected": nx.is_connected(graph),
            "number_of_components": nx.number_connected_components(graph),
        }

        if info["is_connected"]:
            info["diameter"] = nx.diameter(graph)
            info["average_clustering"] = nx.average_clustering(graph)
            info["average_shortest_path_length"] = nx.average_shortest_path_length(graph)
        else:
            # Get info for largest connected component
            largest_cc = max(nx.connected_components(graph), key=len)
            largest_subgraph = graph.subgraph(largest_cc)
            info["largest_component_nodes"] = len(largest_cc)
            info["largest_component_edges"] = len(largest_subgraph.edges())
            info["largest_component_diameter"] = nx.diameter(largest_subgraph)
            info["largest_component_avg_clustering"] = nx.average_clustering(largest_subgraph)

        return info

    def load_as_networkx(self, force_download: bool = False, validate: bool = True) -> Tuple[nx.Graph, dict]:
        """
        Complete pipeline to load AS dataset as NetworkX graph.

        Args:
            force_download: If True, re-download dataset even if cached
            validate: If True, validate graph against expected statistics

        Returns:
            Tuple of (NetworkX graph, metadata dictionary)

        Raises:
            Exception: If any step in the pipeline fails
        """
        print("Loading Stanford SNAP AS Dataset (January 2, 2000)")
        print("=" * 55)

        # Download dataset
        dataset_path = self.download_dataset(force_download=force_download)

        # Parse dataset
        graph = self.parse_dataset(dataset_path)

        # Validate if requested
        if validate:
            validation_passed = self.validate_graph(graph)
            if not validation_passed:
                print("Warning: Graph validation failed, but continuing...")

        # Get detailed graph information
        graph_info = self.get_graph_info(graph)

        # Create metadata
        metadata = {
            "dataset_name": "Stanford SNAP AS-20000102",
            "dataset_url": self.DATASET_URL,
            "date": "January 2, 2000",
            "source": "University of Oregon Route Views Project",
            "description": "Autonomous Systems graph from BGP routing data",
            "validation_passed": validate and self.validate_graph(graph),
            **graph_info,
        }

        print("\nDataset loaded successfully!")
        print(f"Graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges")

        return graph, metadata


def main():
    """
    Example usage of the AS dataset loader.
    """
    # Create loader instance
    loader = ASDatasetLoader()

    # Load the dataset
    try:
        graph, metadata = loader.load_as_networkx()

        print("\nDataset Metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

        # Example: Get some basic graph properties
        print("\nSample edges (first 10):")
        for i, (u, v) in enumerate(list(graph.edges())[:10]):
            print(f"  {u} -- {v}")

        print("\nNode degree statistics:")
        degrees = [graph.degree(n) for n in graph.nodes()]
        print(f"  Min degree: {min(degrees)}")
        print(f"  Max degree: {max(degrees)}")
        print(f"  Average degree: {sum(degrees) / len(degrees):.2f}")

    except Exception as e:
        print(f"Error loading dataset: {str(e)}")


if __name__ == "__main__":
    main()
