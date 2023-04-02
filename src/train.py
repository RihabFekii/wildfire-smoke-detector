from ultralytics import YOLO
import yaml
from pathlib import Path
import os


ROOT_DIR = Path(__file__).resolve().parents[1]  # root directory absolute path
DATA_DIR = os.path.join(ROOT_DIR, "data/raw/wildfire-raw-yolov8")
DATA_YAML = os.path.join(DATA_DIR, "data.yaml")
print(DATA_YAML)


if __name__ == '__main__':

    with open(r"src/params.yaml") as f:
        params = yaml.safe_load(f)

    # load a pre-trained model 
    pre_trained_model = YOLO(params['model_type'])

    # train 
    model = pre_trained_model.train(
        data=DATA_YAML,
        imgsz=params['imgsz'],
        batch=params['batch'],
        epochs=params['epochs'],
        optimizer=params['optimizer'],
        lr0=params['lr0'],
        seed=params['seed'],
        pretrained=params['pretrained'],
        name=params['name']
    )

    # evaluate model 
    model.val()







