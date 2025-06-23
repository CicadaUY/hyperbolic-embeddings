from typing import Dict, List, Optional, Tuple

import dmercator
import numpy as np

from models.base_hyperbolic_model import BaseHyperbolicModel


class DMercatorModel(BaseHyperbolicModel):
    def __init__(self, config: Dict):
        self.dim = config.get("dim", 1)
        self.beta = config.get("beta", -1)
        self._edge_file = "tmp/data.edges"

    def train(
        self,
        edge_list: Optional[List[Tuple[str, str]]] = None,
        adjacency_matrix: Optional[np.ndarray] = None,
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

    def polar_array_to_cartesian(self, theta: np.ndarray, radius: np.ndarray) -> np.ndarray:
        rho = np.tanh(radius / 2)
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return np.stack((x, y), axis=1)

    def get_all_embeddings(self, model_path: Optional[str] = None) -> np.ndarray:
        if model_path:
            self.embeddings_path = model_path

        theta = np.loadtxt(self.embeddings_path, usecols=[2])
        radius = np.loadtxt(self.embeddings_path, usecols=[3])

        embeddings = self.polar_array_to_cartesian(theta, radius)

        return embeddings

    def get_embedding(self, node_id: str, model_path: Optional[str] = None) -> np.ndarray:
        embeddings = self.get_all_embeddings(model_path)
        return embeddings[int(node_id)]

    def most_similar(self, node_id: str, topn: int = 5, model_path: Optional[str] = None) -> List[Tuple[str, float]]:
        pass
