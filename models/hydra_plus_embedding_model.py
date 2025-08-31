import pickle
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from models.base_hyperbolic_model import BaseHyperbolicModel
from models.hydra.hydra import hydra_plus
from utils.geometric_conversions import spherical_to_hyperboloid


class HydraPlusModel(BaseHyperbolicModel):
    def __init__(self, config: Dict):
        self.dim = config.get("dim", 2)
        self.curvature = config.get("curvature", 1)
        self.alpha = config.get("alpha", 1.1)
        self.equi_adj = config.get("equi_adj", 0.5)
        self.control = config.get("control", None)
        self.curvature_bias = config.get("curvature_bias", 1.0)
        self.curvature_freeze = config.get("curvature_freeze", True)
        self.curvature_max = config.get("curvature_max", None)
        self.maxit = config.get("maxit", 1000)

    def train(
        self,
        edge_list: Optional[List[Tuple[str, str]]] = None,
        adjacency_matrix: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        model_path: str = "saved_models/model.bin",
    ):
        if edge_list is not None:
            n = max(max(u, v) for u, v in edge_list) + 1
            G = nx.DiGraph()
            G.add_edges_from(edge_list)
            G.add_nodes_from(range(n))
        elif adjacency_matrix is not None:
            G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
        else:
            raise ValueError("You must provide either edge_list or adjacency_matrix.")

        # Build distance matrix
        lengths = dict(nx.all_pairs_shortest_path_length(G))
        n = G.number_of_nodes()
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i in lengths and j in lengths[i]:
                    D[i, j] = lengths[i][j]

        # Run Hydra
        self.embeddings = hydra_plus(
            D,
            self.dim,
            self.curvature,
            self.alpha,
            self.equi_adj,
            self.control,
            self.curvature_bias,
            self.curvature_freeze,
            self.curvature_max,
            self.maxit,
        )

        # Save embeddings
        with open(model_path, "wb") as f:
            pickle.dump(self.embeddings, f)

    def spherical_to_hyperboloid_coordinates(self, theta: np.ndarray, radius: np.ndarray) -> np.ndarray:
        """
        Convert spherical coordinates (theta, radius) to hyperboloid coordinates.

        Parameters:
        - theta: Angular coordinates
        - radius: Radial coordinates (distance from origin)

        Returns:
        - Hyperboloid coordinates (x, y, t)
        """

        spherical_coords = np.column_stack([radius, theta])

        # Convert to hyperboloid coordinates
        hyperboloid_coords = spherical_to_hyperboloid(spherical_coords)

        return hyperboloid_coords

    def get_all_embeddings(self, model_path: Optional[str] = None) -> np.ndarray:
        if model_path:
            with open(model_path, "rb") as f:
                self.embeddings = pickle.load(f)

        theta = self.embeddings["theta"]
        radius = self.embeddings["r"]
        _embeddings = self.spherical_to_hyperboloid_coordinates(theta, radius)

        return _embeddings

    def get_embedding(self, node_id: str, model_path: Optional[str] = None) -> np.ndarray:
        embeddings = self.get_all_embeddings(model_path)
        return embeddings[int(node_id)]

    def most_similar(self, node_id: str, topn: int = 5, model_path: Optional[str] = None) -> List[Tuple[str, float]]:
        pass
