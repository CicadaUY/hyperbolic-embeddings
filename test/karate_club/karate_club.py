import networkx as nx
import numpy as np

from hyperbolic_embeddings import HyperbolicEmbeddings

# Load Karate Club
G = nx.karate_club_graph()
edge_list = list(G.edges())
A = nx.to_numpy_array(G)
labels = [G.nodes[n]["club"] for n in sorted(G.nodes())]

config_poincare = {"dim": 2, "negs": 5, "epochs": 1000, "batch_size": 256, "dimension": 1}  # D-mercator

config_lorentz = {
    "dim": 2,
    "epochs": 50000,
    "batch_size": 256,
}

config_dmercator = {"dim": 1}

config_hydra = {"dim": 2}

print("Training Poincare Embeddings")
embedding_runner = HyperbolicEmbeddings(embedding_type="poincare_embeddings", config=config_poincare)
embedding_runner.train(edge_list=edge_list, model_path="saved_models/karate_club/poincare_embedddings.bin")
embedding_runner.plot_embeddings(labels=labels, edge_list=edge_list, save_path="test/karate_club/plots/poincare_embedddings.pdf")

print("Training Lorentz Embeddings")
embedding_runner = HyperbolicEmbeddings(embedding_type="lorentz", config=config_lorentz)
embedding_runner.train(edge_list=edge_list, model_path="saved_models/karate_club/lorentz_embeddings.bin")
embedding_runner.plot_embeddings(labels=labels, save_path="test/karate_club/plots/lorentz_embeddings.pdf")

print("Training D-Mercator Embeddings")
embedding_runner = HyperbolicEmbeddings(embedding_type="dmercator", config=config_dmercator)
embedding_runner.train(edge_list=edge_list, model_path="saved_models/karate_club/dmercator_embeddings.bin")
embedding_runner.plot_embeddings(labels=labels, edge_list=edge_list, save_path="test/karate_club/plots/dmercator_embeddings.pdf")

print("Training Hydra Embeddings")
embedding_runner = HyperbolicEmbeddings(embedding_type="hydra", config=config_hydra)
embedding_runner.train(adjacency_matrix=A, model_path="saved_models/karate_club/hydra_embeddings.bin")
embedding_runner.plot_embeddings(labels=labels, edge_list=edge_list, save_path="test/karate_club/plots/hydra_embeddings.pdf")
