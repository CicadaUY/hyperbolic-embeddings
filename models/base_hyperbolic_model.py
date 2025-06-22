from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np


class BaseHyperbolicModel(ABC):

    @abstractmethod
    def train(
        self,
        edge_list: Optional[List[tuple]] = None,
        adjacency_matrix: Optional[np.ndarray] = None,
        model_path: str = "saved_models/model.bin",
    ):
        pass

    @abstractmethod
    def get_embedding(self, node_id: str, model_path: Optional[str] = None) -> np.ndarray:
        pass

    @abstractmethod
    def get_all_embeddings(self, model_path: Optional[str] = None) -> np.ndarray:
        pass

    @abstractmethod
    def most_similar(self, node_id: str, topn: int = 5, model_path: Optional[str] = None) -> List[Tuple[str, float]]:
        pass
