SHELL = /bin/bash
SRC = src
DATA = data

# Development-related targets

# list dependencies
requirements: install-dev
	pipreqs --force --savepath requirements/prod.txt $(SRC)

# install dependencies for production
install: install-dev
	pip install -r requirements/prod.txt

# install dependencies for development
install-dev:
	pip install -r requirements/dev.txt

# remove Python file artifacts
clean:
	@find . -name '*.pyc' -exec rm -f {} +
	@find . -name '*.pyo' -exec rm -f {} +
	@find . -name '*~' -exec rm -f {} +
	@find . -name '__pycache__' -exec rm -rf {} +

# check style
lint: format
	pylint --exit-zero --jobs=0 --output-format=colorized $(SRC)
	pycodestyle --show-source $(SRC)
	pydocstyle $(SRC)

# format according to style guide
format:
	yapf --in-place --recursive $(SRC)
	isort -rc $(SRC)

# Data-related targets

# download the required datasets
download:
	$(SRC)/download.sh
	python $(SRC)/download.py
	cat $(DATA)/unimorph/fin/fin.1 $(DATA)/unimorph/fin/fin.2 > data/unimorph/fin/fin
	rm -rf $(DATA)/universaldependencies/UD_Swedish_Sign_Language-SSLC

# prepare feature data
features:
	python $(SRC)/features.py

# prepare cloze data
cloze:
	python $(SRC)/cloze.py

# get probabilities for cloze examples
probs:
	python $(SRC)/probabilities.py

# run the experiment
experiment:
	python $(SRC)/experiment.py

# remove all data. Be careful with this one.
remove:
	@rm -rf $(DATA)
	@mkdir $(DATA)

.PHONY: requirements install install-dev clean lint format download features
	cloze remove
