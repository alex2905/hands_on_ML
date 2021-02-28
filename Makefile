CONDA_ENV_NAME := hands-on-ml
artefacts_path := artefacts

environment:
	conda env create -n $(CONDA_ENV_NAME) --force --file environment.yml

nbstrip:
	find . -name "*.ipynb" -exec nbstripout "{}" +

nb2html:
	jupyter nbconvert --to html *.ipynb --output-dir="$(artefacts_path)" || true
