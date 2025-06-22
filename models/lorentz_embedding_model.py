from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datetime import datetime
import os
from tqdm import tqdm
import logging

from models.base_hyperbolic_model import BaseHyperbolicModel
from models.lorentz.lorentz import Lorentz, RSGD, Graph, recon

# Enable logging for gensim
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)


class LorentzEmbeddingsModel(BaseHyperbolicModel):
    def __init__(self, config: dict):
        self.dim = config.get("dim", 2)
        self.epochs = config.get("epochs", 1000)
        self.batch_size = config.get("batch_size", 1)
        self.sample_size = config.get("sample_size", 5)
        self.learning_rate = config.get("learning_rate", 0.1)
        self.burn_epochs = config.get("burn_epochs", 10)
        self.burn_c = config.get("burn_c", 10)
        self.loader_workers = config.get("loader_workers", 2)
        self.logdir = config.get("logdir", "runs")
        self.save_step = config.get("save_step", 100)
        self.shuffle = config.get("shuffle", True)
        self.model = None

    def train(self, 
              edge_list: Optional[List[tuple]] = None,
              adjacency_matrix: Optional[np.ndarray] = None,
              model_path: str = "saved_models/model.bin"):
        

        if edge_list is None and adjacency_matrix is None:
            raise ValueError("You must provide either edge_list or adjacency_matrix.")
        
        if edge_list is not None:
            # Extract node set and map to integer IDs
            node_ids = sorted(set(str(u) for u, v in edge_list) | set(str(v) for u, v in edge_list))
            id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
            self.node_lookup = {idx: node_id for node_id, idx in id_to_index.items()}  # For decoding later

            size = len(node_ids)
            adjacency_matrix = np.zeros((size, size), dtype=int)
            for u, v in edge_list:
                adjacency_matrix[id_to_index[str(u)], id_to_index[str(v)]] = 1

        self.num_nodes = adjacency_matrix.shape[0]
        self.model = Lorentz(n_items=self.num_nodes, dim=self.dim + 1)
        dataset = Graph(adjacency_matrix, sample_size=self.sample_size)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.loader_workers)

        self.writer = SummaryWriter(f"{self.logdir}/{datetime.utcnow()}")
        self.optimizer = RSGD(self.model.parameters(), learning_rate=self.learning_rate)

        with tqdm(total=self.epochs, ncols=80, desc="Training", dynamic_ncols=True) as pbar:
            for epoch in range(self.epochs):
                if hasattr(self, 'optimizer') and self.optimizer is not None:
                    self.optimizer.learning_rate = (
                        self.learning_rate / self.burn_c
                        if epoch < self.burn_epochs
                        else self.learning_rate
                    )
                for I, Ks in dataloader:
                    self.optimizer.zero_grad()
                    loss = self.model(I, Ks).mean()
                    loss.backward()
                    self.optimizer.step()

                self.writer.add_scalar("loss", loss, epoch)
                self.writer.add_scalar("recon_perf", recon(self.model.get_lorentz_table(), adjacency_matrix), epoch)
                self.writer.add_scalar("table_test", self.model._test_table(), epoch)


                if epoch % self.save_step == 0:
                    tmp_path = model_path + ".tmp"
                    torch.save(self.model.state_dict(), tmp_path)
                    os.replace(tmp_path, model_path)

                # Update progress bar
                pbar.set_description(f"Epoch {epoch+1} | Loss: {loss.item():.4f}" if 'loss' in locals() else f"Epoch {epoch+1}")
                pbar.update(1)

        logger.info("Training completed.")

        tmp_path = model_path + ".tmp"
        torch.save(self.model.state_dict(), tmp_path)
        os.replace(tmp_path, model_path)
        logger.info(f"Model saved in {model_path}.")

    def get_embedding(self, node_id: str, model_path: Optional[str] = None) -> np.ndarray:
        if model_path:
            self.model = Lorentz(n_items=self.num_nodes, dim=self.dim + 1)
            self.model.load_state_dict(torch.load(model_path))

        node_index = int(node_id) + 1  # +1 due to padding idx
        return self.model.get_lorentz_table()[node_index]

    def get_all_embeddings(self, model_path: Optional[str] = None) -> np.ndarray:
        if model_path:
            self.model = Lorentz(n_items=self.num_nodes, dim=self.dim + 1)
            self.model.load_state_dict(torch.load(model_path))
        return self.model.get_lorentz_table()[1:]

    def most_similar(self, node_id: str, topn: int = 5, model_path: Optional[str] = None) -> List[Tuple[str, float]]:
        pass
