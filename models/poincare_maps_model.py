import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from scipy.sparse import csgraph
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.base_hyperbolic_model import BaseHyperbolicModel
from models.PoincareMaps.data import compute_rfa
from models.PoincareMaps.model import PoincareDistance, PoincareEmbedding
from models.PoincareMaps.rsgd import RiemannianSGD
from utils.geometric_conversions import poincare_to_hyperboloid

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Disable PyTorch multiprocessing to prevent segfaults on macOS
torch.multiprocessing.set_sharing_strategy("file_system")


class PoincareMapsModel(BaseHyperbolicModel):
    def __init__(self, config: Dict):
        self.dim = config.get("dim", 2)
        self.epochs = config.get("epochs", 50)
        self.batch_size = config.get("batch_size", -1)
        self.lr = config.get("lr", 0.3)
        self.sigma = config.get("sigma", 1.0)
        self.gamma = config.get("gamma", 1.0)
        self.burnin = config.get("burnin", 500)
        self.lrm = config.get("lrm", 1.0)
        self.earlystop = config.get("earlystop", 0.0001)
        self.mode = config.get("mode", "features")
        self.distlocal = config.get("distlocal", "minkowski")
        self.k_neighbours = config.get("k_neighbours", 15)
        # Force CPU on macOS to avoid CUDA-related segfaults
        self.device = config.get("device", "cpu" if not torch.cuda.is_available() else "cuda")
        self.logger = logging.getLogger(__name__)

    @property
    def native_space(self) -> str:
        """Get the native embedding space for this model."""
        return "poincare"

    def train(
        self,
        edge_list: Optional[List[tuple]] = None,
        adjacency_matrix: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        model_path: str = "saved_models/model.bin",
    ):
        if features:
            RFA = compute_rfa(
                features,
                mode=self.mode,
                k_neighbours=self.k_neighbours,
                distlocal=self.distlocal,
                distfn="MFIsym",
                connected=True,
                sigma=self.sigma,
            )
        else:
            if edge_list is None and adjacency_matrix is None:
                raise ValueError("Either edge_list or adjacency_matrix must be provided")

            if adjacency_matrix is None:
                self.logger.info("Converting edge list into adjacency matrix")
                # Convert edge list to adjacency matrix
                G = nx.Graph()
                G.add_edges_from(edge_list)
                adjacency_matrix = nx.to_numpy_array(G, dtype=int)

            self.logger.info("Computing Laplacian")
            L = csgraph.laplacian(adjacency_matrix, normed=False)
            RFA = np.linalg.inv(L + np.eye(L.shape[0]))
            RFA[RFA == np.nan] = 0.0
            RFA = torch.Tensor(RFA)

        # RFA = RFA.to(self.device)

        if self.batch_size < 0:
            # Force batch_size=1 to avoid PyTorch softmax segfault on Python 3.11.9/macOS
            self.batch_size = 1  # min(512, int(len(RFA) / 10))

        self.lr = self.batch_size / 16 * self.lr

        indices = torch.arange(len(RFA))
        # indices = indices.to(self.device)

        dataset = TensorDataset(indices, RFA)

        # Instantiate Embedding predictor
        # Explicitly set cuda=0 to ensure CPU mode on macOS
        self.model = PoincareEmbedding(
            size=len(dataset),
            dim=self.dim,
            dist=PoincareDistance,
            max_norm=1,
            Qdist="laplace",
            lossfn="klSym",
            gamma=self.gamma,
            cuda=0,  # Force CPU mode
        )

        optimizer = RiemannianSGD(self.model.parameters(), lr=self.lr)

        # Train Embeddings
        self.logger.info("Starting training...")
        self.logger.info(f"Using device: {self.device}, CUDA available: {torch.cuda.is_available()}")

        # Disable all multiprocessing features to prevent segfaults on macOS
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,  # Can cause issues on macOS
            persistent_workers=False,  # Ensure no worker persistence
            prefetch_factor=None,  # Disable prefetching
        )

        pbar = tqdm(range(self.epochs), ncols=80, file=sys.stdout, disable=False)

        n_iter = 0
        epoch_loss = []
        earlystop_count = 0

        for epoch in pbar:
            grad_norm = []

            # determine learning rate
            lr = self.lr
            if epoch < self.burnin:
                lr = lr * self.lrm

            epoch_error = 0
            try:
                for batch_idx, (inputs, targets) in enumerate(loader):
                    loss = self.model.lossfn(self.model(inputs), targets)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step(lr=lr)

                    epoch_error += loss.item()

                    grad_norm.append(self.model.lt.weight.grad.data.norm().item())

                    n_iter += 1
            except Exception as e:
                self.logger.error(f"Error during training iteration: {e}")
                self.logger.error(f"Inputs shape: {inputs.shape if 'inputs' in locals() else 'N/A'}")
                self.logger.error(f"Targets shape: {targets.shape if 'targets' in locals() else 'N/A'}")
                raise
                raise

            epoch_error /= len(loader)
            epoch_loss.append(epoch_error)
            # pbar.set_description("loss: {:.5f}".format(epoch_error))
            pbar.set_description(f"loss: {epoch_error:.5f}")

            if epoch > 10:
                delta = abs(epoch_loss[epoch] - epoch_loss[epoch - 1])
                if delta < self.earlystop:
                    earlystop_count += 1
                if earlystop_count > 50:
                    self.logger.info(f"\nStopped at epoch {epoch}")
                    break

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        self.logger.info(f"Model saved to {model_path}")

    def get_embedding(self, node_id: str, model_path: Optional[str] = None) -> np.ndarray:
        if model_path:
            state = torch.load(model_path)
            self.model.load_state_dict(state)

        embeddings = self.model.lt.weight.cpu().detach().numpy()

        return embeddings[int(node_id)]

    def get_all_embeddings(self, model_path: Optional[str] = None) -> np.ndarray:
        if model_path:
            state = torch.load(model_path)
            self.model.load_state_dict(state)

        return self.model.lt.weight.detach().cpu().numpy()

    def most_similar(self, node_id: str, topn: int = 5, model_path: Optional[str] = None) -> List[Tuple[str, float]]:
        pass

    def to_hyperboloid(self, model_path: Optional[str] = None) -> np.ndarray:
        """Convert Poincaré embeddings to hyperboloid coordinates."""
        poincare_embeddings = self.get_all_embeddings(model_path)
        return poincare_to_hyperboloid(poincare_embeddings)

    def to_poincare(self, model_path: Optional[str] = None) -> np.ndarray:
        """Return embeddings in Poincaré coordinates (already in Poincaré space)."""
        return self.get_all_embeddings(model_path)
