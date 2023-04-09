import os
import shutil
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]  # root directory absolute path


def save_model(experiment_name: str):
    """ saves the weights of trained model to the models directory """ 
    if os.path.isdir('runs'):
        model_weights = experiment_name + "/weights/best.pt"
        path_model_weights = os.path.join(ROOT_DIR, "runs/detect", model_weights)

        shutil.copy(src=path_model_weights, dst=f'{ROOT_DIR}/models/model.pt')


def save_metrics_and_params(experiment_name: str) -> None:
    """ saves training metrics, params and confusion matrix to the reports directory """ 
    if os.path.isdir('runs'):
        path_metrics = os.path.join(ROOT_DIR, "runs/detect", experiment_name)

        # save experiment training metrics  
        shutil.copy(src=f'{path_metrics}/results.csv', dst=f'{ROOT_DIR}/reports/train_metrics.csv')

        # save the confusion matrix associated to the training experiment
        shutil.copy(src=f'{path_metrics}/confusion_matrix.png', dst=f'{ROOT_DIR}/reports/train_confusion_matrix.png')

        # save training params
        shutil.copy(src=f'{path_metrics}/args.yaml', dst=f'{ROOT_DIR}/reports/train_params.yaml')
    
