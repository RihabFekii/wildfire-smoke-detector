import os
import glob
from pathlib import Path

from ultralytics import YOLO


root_dir = Path(__file__).resolve().parents[1]  # root directory
path_model_weights = os.path.join(root_dir, 'models/model.pt')
data_dir = os.path.join(root_dir, "data/raw/wildfire-raw-yolov8")
test_data_path = os.path.join(data_dir, 'test/images')


if __name__ == '__main__':

    #load trained model 
    model = YOLO(path_model_weights)

    for images in glob.glob(f"{test_data_path}/*.jpg"):
        predictions = model.predict(images, save=True)

