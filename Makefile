PYTHON_INTERPRETER = python3

env:
	@echo ">>> Creating a python virtual environment with venv"
	$(PYTHON_INTERPRETER) -m venv env
	@echo ">>> A new virtual env is created. Activate it with:\nsource env/bin/activate ."