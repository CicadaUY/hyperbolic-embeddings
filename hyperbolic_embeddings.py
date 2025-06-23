import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from models.base_hyperbolic_model import BaseHyperbolicModel
from models.dmercator_embedding_model import DMercatorModel
from models.lorentz_embedding_model import LorentzEmbeddingsModel
from models.poincare_embedding_model import PoincareEmbeddingModel


class HyperbolicEmbeddings:

    def __init__(self, embedding_type: str, config: Dict):
        self.embedding_type = embedding_type
        self.model = self._load_model(embedding_type, config)

    def _load_model(self, embedding_type, config) -> BaseHyperbolicModel:
        if embedding_type == "poincare_embeddings":
            return PoincareEmbeddingModel(config)
        elif embedding_type == "lorentz":
            return LorentzEmbeddingsModel(config)
        elif embedding_type == "poincare_maps":
            return NotImplementedError
        elif embedding_type == "dmercator":
            return DMercatorModel(config)
        elif embedding_type == "hydra":  # R con código de Python
            raise NotImplementedError
        else:
            raise ValueError(f"Unsupported model type: {embedding_type}")

    def train(
        self,
        edge_list: Optional[List[tuple]] = None,
        adjacency_matrix: Optional[np.ndarray] = None,
        model_path: str = "saved_models/model.bin",
    ):
        self.model.train(edge_list, adjacency_matrix, model_path)

    def get_node_embedding(self, node_id: str, model_path: Optional[str] = None):
        return self.model.get_embedding(node_id, model_path)

    def get_all_embeddings(self, model_path: Optional[str] = None):
        return self.model.get_all_embeddings(model_path)

    def plot_embeddings(self, labels: Optional[List[str]] = None, edge_list: Optional[List[Tuple]] = None, save_path: Optional[str] = None):
        embeddings = self.get_all_embeddings()  # Must be implemented by each model
        if embeddings.shape[1] > 2:
            warnings.warn(f"Embedding dimension is {embeddings.shape[1]}; plotting only the first 2 dimensions.")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(f"Hyperbolic Embeddings using {self.embedding_type} model")

        x, y = embeddings[:, 0], embeddings[:, 1]

        if edge_list:
            for u, v in edge_list:
                x_vals = [x[u], x[v]]
                y_vals = [y[u], y[v]]
                ax.plot(x_vals, y_vals, color="gray", linewidth=0.5, alpha=0.5, zorder=1)

        if labels:
            labels = np.array(labels)
            unique_labels = sorted(set(labels))
            colormap = cm.get_cmap("tab10", len(unique_labels))
            label_to_color = {label: colormap(i) for i, label in enumerate(unique_labels)}

            for label in unique_labels:
                indices = np.where(labels == label)[0]
                ax.scatter(x[indices], y[indices], s=100, edgecolor="black", color=label_to_color[label], label=label)

            ax.legend(loc="upper right", fontsize=8, frameon=True)
        else:
            ax.scatter(x, y, s=100, edgecolor="black", color="skyblue")

        for i in range(len(x)):
            ax.text(x[i], y[i], str(i), fontsize=8, ha="center", va="center", zorder=3)

        # Draw boundary circle (e.g., for Poincaré disk)
        ax.add_artist(plt.Circle((0, 0), 1, fill=False, color="black", linestyle="--"))
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal")
        ax.axis("off")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
