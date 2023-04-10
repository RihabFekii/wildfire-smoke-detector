#!make
include .env

PYTHON_INTERPRETER = python3

env:
	@echo ">>> Creating a python virtual environment with venv"
	$(PYTHON_INTERPRETER) -m venv env
	@echo ">>> A new virtual env is created. Activate it with:\nsource env/bin/activate ."


requirements: 
	@echo ">>> Installing project requirements"
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

mlflow:
	@echo ">>> Authenticating to MLflow remote server on DagsHub"
	export MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI)
	export MLFLOW_TRACKING_USERNAME=$(MLFLOW_TRACKING_USERNAME) 
	export MLFLOW_TRACKING_PASSWORD=$(MLFLOW_TRACKING_PASSWORD) 
	@echo ">>> Authenticating successful!"
	