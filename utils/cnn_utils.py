import cv2
import torch
import yaml
from torchvision import transforms
from ultralytics import YOLO
import numpy as np

from dataset.classification import lisa, gtsrb
from dataset.detection.FasterRCNN import build_model
from utils import utils


def load_trained_detector(dataset):
    assert dataset in ['GTSDB', 'Mapillary']

    if dataset == "Mapillary":
        with open('./dataset/detection/Mapillary.yaml', 'r') as f:
            mapillary_data = yaml.safe_load(f)
            class_n_mapillary = len(mapillary_data['names'])

        num_classes = class_n_mapillary
        model = YOLO('./model/detection/yolo_mapillary.pt')
        pre_process = transforms.Compose([lambda image: cv2.resize(image, (1376, 800)), transforms.ToTensor()])
    elif dataset == "GTSDB":
        with open('./dataset/detection/GTSDB.yaml', 'r') as f:
            gtsdb_data = yaml.safe_load(f)
            class_n_gtsdb = len(gtsdb_data['names'])

        num_classes = class_n_gtsdb
        model = build_model(num_classes)
        model.eval()
        model.load_state_dict(torch.load('model/detection/FasterRCNN_GTSDB.pth', map_location=utils.get_device()))
        model.to(utils.get_device())
        pre_process = transforms.Compose([lambda image: cv2.resize(image, (1360, 800)), transforms.ToTensor()])

    return pre_process, num_classes, model


def load_trained_classifier(attack_db, target_model, robust=False):
    assert attack_db in ['LISA', 'GTSRB']
    if attack_db == "LISA":
        model = lisa.initialize_model()
        model.load_state_dict(
            torch.load(f'./model/classification/model_lisa_LisaCNN.pth',
                       map_location=torch.device(utils.get_device())))
        pre_process = transforms.Compose([lambda image: cv2.resize(image, (32, 32)), transforms.ToTensor()])
        model.eval()
    elif attack_db == "GTSRB":
        model = gtsrb.initialize_model(target_model)
        model.load_state_dict(torch.load(
            f'./model/classification/model_gtsrb_{target_model}{"_adv_retrain" if robust else ""}.pth',
            map_location=torch.device(utils.get_device())))
        if not robust:
            pre_process = transforms.Compose(
                [lambda image: cv2.resize(image, (32, 32)),
                 transforms.ToTensor(),
                 transforms.Normalize((0.31100202, 0.31318352, 0.3089177), (0.271555, 0.26502392, 0.25716692))
                 ])
        else:
            pre_process = transforms.Compose([lambda image: image.astype(np.float32), transforms.ToTensor()])
        model.eval()
    return pre_process, model

