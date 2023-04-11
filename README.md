# wildfire-smoke-detector
ML project for wildfire smoke detection with YOLOv8. 

Follow this [article](https://rihab-feki.medium.com/ml-project-using-yolov8-roboflow-dvc-and-mlflow-on-dagshub-3e5c0b026297) as a detailed guide. 


This project is connected to a repository on [DagsHub](https://dagshub.com/Rihab.Feki/wildfire-smoke-detector). 

**DagsHub** is a GitHub for Machine Learning projects. It eases MLOps practices by enabling data scientists and machine learning engineers to version their data, models, experiments, and code, through its integration with [DVC](https://dvc.org/doc) and [MLflow](https://mlflow.org/docs/latest/index.html). 

Project Organization
------------

    ├── Makefile           <- Makefile with commands like `make env` or `make requirements`.
    ├── README.md          <- Documentation for using this project.
    ├── params.yaml        <- configuration parameters e.g for training 
    ├── data
    │   ├── processed      <- Processed dataset.
    │   └── raw            <- The original dataset (immutable data).
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │   └── train_metrics.csv    <- Relevant metrics after evaluating the model.
    │   └── train_params.yaml    <- Params for training the model.
    │
    ├── requirements.txt   <- The requirements file for reproducing the environment.
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── train.py       <- Scripts to train the model 
    │   ├── predict.py     <- Script to make predictions on test data
    │   └──utils.py       <- Utility functions 
    │
    ├── dvc.lock           <- The version definition of each dependency, stage, and output from the 
    │                         DVC data pipeline.
    └── dvc.yaml           <- Defining the data pipeline stages, dependencies, and outputs.


--------

## Run the project locally 

1. Clone the project: 
````shell 
git clone https://github.com/RihabFekii/
````

2. Set up a virtual python environment, by running the following commands:
````shell 
make env
source env/bin/activate .
````

3. Install requirements, by running this command:
````shell
make requirements
`````

4. Pull the data 
````
dvc pull
````
## Run experiments with DVC 

1. You can experiment with the different models of YOLOv8 and edit its hyperparameters 
by editing the [params.yaml](/params.yaml) file. 

2. Since the data pipeline is created with dvc, you can easily reproduce experiments by runing: 

````sell
dvc exp run 
`````
3. Visualize experiments by running: 
````
dvc exp show
`````


