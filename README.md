# wildfire-smoke-detector
ML project for wildfire smoke detection. Follow this [article](https://rihab-feki.medium.com/ml-project-using-yolov8-roboflow-dvc-and-mlflow-on-dagshub-3e5c0b026297) as a detailed guide. 

This project is connected to a repository on [DagsHub](https://dagshub.com/Rihab.Feki/wildfire-smoke-detector). 

**DagsHub** is a GitHub for Machine Learning projects. It eases MLOps practices by enabling data scientists and machine learning engineers to version their data, models, experiments, and code, through its integration with [DVC](https://dvc.org/doc) and [MLflow](https://mlflow.org/docs/latest/index.html). 

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


