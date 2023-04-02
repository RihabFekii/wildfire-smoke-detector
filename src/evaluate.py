from ultralytics import YOLO
from pathlib import Path
import yaml

import os

ROOT_DIR = Path(__file__).resolve().parents[1]  # root directory

with open(r"src/params.yaml") as f:
    params = yaml.safe_load(f)

experiment_name = params['name']
model_weights = experiment_name + "/weights/best.pt"
path_model_weights = os.path.join(ROOT_DIR, "runs/detect", model_weights)
print(path_model_weights)

#load trained model 
model = YOLO(path_model_weights)

model.val()