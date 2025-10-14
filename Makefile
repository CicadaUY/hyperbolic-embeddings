VENV_DIR := venv
MODELS_DIR := models

.PHONY: all env install clone-repos setup clean

all: env clone-repos install setup

# Create virtual environment
env:
	python -m venv $(VENV_DIR)

# Clone required repositories if they don't exist
clone-repos:
	@echo "Checking and cloning required repositories..."
	@mkdir -p $(MODELS_DIR)
	@if [ ! -d "$(MODELS_DIR)/d-mercator" ]; then \
		echo "Cloning d-mercator..."; \
		cd $(MODELS_DIR) && git clone https://github.com/CicadaUY/d-mercator.git; \
	else \
		echo "d-mercator already exists, skipping..."; \
	fi
	@if [ ! -d "$(MODELS_DIR)/hypermap" ]; then \
		echo "Cloning hypermap..."; \
		cd $(MODELS_DIR) && git clone https://github.com/CicadaUY/hypermap.git; \
	else \
		echo "hypermap already exists, skipping..."; \
	fi
	@if [ ! -d "$(MODELS_DIR)/lorentz" ]; then \
		echo "Cloning lorentz-embeddings..."; \
		cd $(MODELS_DIR) && git clone https://github.com/CicadaUY/lorentz-embeddings.git; \
		mv $(MODELS_DIR)/lorentz-embeddings $(MODELS_DIR)/lorentz; \
		rm -rf $(MODELS_DIR)/lorentz-embeddings; \
	else \
		echo "lorentz-embeddings already exists, skipping..."; \
	fi
	@if [ ! -d "$(MODELS_DIR)/PoincareMaps" ]; then \
		echo "Cloning PoincareMaps..."; \
		cd $(MODELS_DIR) && git clone https://github.com/CicadaUY/PoincareMaps.git; \
	else \
		echo "PoincareMaps already exists, skipping..."; \
	fi
	@echo "Repository check complete!"

# Activate env and install dependencies
install: 
	. $(VENV_DIR)/bin/activate && \
	pip install --upgrade pip setuptools wheel && \
	pip install -r requirements.txt

# Run setup.py using the virtualenv
setup:
	. $(VENV_DIR)/bin/activate && \
	cd models/d-mercator/python && pip install -e . && \
	cd ../../../ && \
	$(VENV_DIR)/bin/python models/hypermap/python/setup.py build_ext --inplace

# Remove the virtual environment
clean:
	rm -rf $(VENV_DIR)
