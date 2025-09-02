# Hyperbolic Embeddings

This repository provides tools to train, evaluate, and visualize graph embeddings in hyperbolic space.
Supported models include:

- Poincare Embeddings
- Lorentz Embeddings
- Poincare Maps
- D-Mercator
- Hydra
- Hydra Plus
- Hypermap


## Setup

### Requirements:

- Python 3.9+ is required

### Clone required repositories

``` 
cd models/
git clone https://github.com/CicadaUY/d-mercator.git
git clone https://github.com/CicadaUY/hypermap.git
git clone https://github.com/CicadaUY/lorentz-embeddings.git
```

### Build and install dependencies 

```
make
```
This will:

- Create a virtual environment (venv)
- Install required dependencies
- Set up local packages like d-mercator and hypermap

## Usage

### Initialize Embedding Manager

```
embedding_manager = HyperbolicEmbeddings(embedding_type="poincare_maps", config=config)
```

### Train embeddings

```
embedding_manager.train(adjacency_matrix=A, model_path=model_path)
```

### Plot embeddings

```
embedding_manager.plot_embeddings(labels=labels, edge_list=edge_list, save_path=plot_path)
```


## Examples


### Tree Test
```
python -m test.tree_test.main --embedding_type='hypermap' --output_space='poincare'
```