VENV_DIR := venv

.PHONY: all env install setup clean

all: env install setup

# Create virtual environment
env:
	python -m venv $(VENV_DIR)

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
