from typing import Dict, List, Optional, Tuple

import dmercator
import numpy as np

from models.base_hyperbolic_model import BaseHyperbolicModel
from utils.geometric_conversions import spherical_to_hyperboloid


class DMercatorModel(BaseHyperbolicModel):
    def __init__(self, config: Dict):
        self.dim = config.get("dim", 1)
        self.beta = config.get("beta", -1)
        self._edge_file = "tmp/data.edges"

    def train(
        self,
        edge_list: Optional[List[Tuple[str, str]]] = None,
        adjacency_matrix: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        model_path: str = "saved_models/model.bin",
    ):
        if edge_list is not None:
            with open(self._edge_file, "w") as f:
                for u, v in edge_list:
                    f.write(f"{u} {v}\n")
        elif adjacency_matrix is not None:
            rows, cols = np.where(adjacency_matrix)
            with open(self._edge_file, "w") as f:
                for u, v in zip(rows, cols):
                    f.write(f"{u} {v}\n")
        else:
            raise ValueError("You must provide either edge_list or adjacency_matrix.")

        # Run d-mercator
        dmercator.embed(edgelist_filename=self._edge_file, output_name=model_path, dimension=self.dim, beta=self.beta)

        self.embeddings_path = model_path + ".inf_coord"

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
            self.embeddings_path = model_path + ".inf_coord"

        theta = np.loadtxt(self.embeddings_path, usecols=[2])
        radius = np.loadtxt(self.embeddings_path, usecols=[3])

        # Convert to hyperboloid coordinates instead of Poincaré
        embeddings = self.spherical_to_hyperboloid_coordinates(theta, radius)

        return embeddings

    def get_embedding(self, node_id: str, model_path: Optional[str] = None) -> np.ndarray:
        embeddings = self.get_all_embeddings(model_path)
        return embeddings[int(node_id)]

    def most_similar(self, node_id: str, topn: int = 5, model_path: Optional[str] = None) -> List[Tuple[str, float]]:
        pass
