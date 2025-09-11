import argparse
import gc
import json
import os
import pickle
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from dataset.classification.traffic_sign_dataset import TrafficSignDataset
from model.classification.LisaCNN import LisaCNN
from utils import utils
from utils.ir_utils import equalize_and_transform_to_ir
from utils.utils import SmoothCrossEntropyLoss, load_mask

with open('./dataset/classification/classes.json', 'r') as config:
    params = json.load(config)
    class_n = params['LISA']['class_n']
    position_list, _ = load_mask()

data_transforms = transforms.Compose([
    transforms.ToTensor(),
])

loss_fun = SmoothCrossEntropyLoss(smoothing=0.1)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


def model_epoch(training_model, data_loader, train=False, optimizer=None, scheduler=None):
    loss = acc = 0.0

    for data_batch in data_loader:
        train_predict = training_model(data_batch[0].to(utils.get_device()))
        batch_loss = loss_fun(train_predict, data_batch[1].to(utils.get_device()))
        if train:
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        acc += (torch.argmax(train_predict.cpu(), dim=1) == data_batch[1]).sum()
        loss += batch_loss.item() * len(data_batch[1])

    if scheduler:
        scheduler.step()

    return acc, loss


def initialize_model():
    trained_model = LisaCNN(n_class=class_n).to(utils.get_device()).apply(weights_init)

    return trained_model


if __name__ == '__main__':
    print(f'Using {utils.get_device()} for computations.')

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="LISA model training and validation")
    parser.add_argument("--model", type=str, default="LisaCNN",
                        help="The classifier architecture to use.", choices=['LisaCNN'])

    subparsers = parser.add_subparsers(required=True)

    # Add parser for the "test" command
    parser_testing = subparsers.add_parser('test', aliases=['testing', 'validation', 'val'])
    parser_testing.add_argument("--robust", action="store_true",
                                help="Set this flag if you want to use the adversarially retrained model.")
    parser_testing.add_argument("--ir", action="store_true",
                                help="Set this flag if you want to perform the testing on the IR transformed test images")
    parser_testing.add_argument("--lux", type=int, default=10,
                                help="Specify the assumed ambient light brightness [Lux]. Requires the --ir parameter to be set.")
    parser_testing.add_argument("--dir", type=str,
                                help="Path to a folder containing images to test the model accuracy on. Overwrites all other parameters.")

    args = parser.parse_args()
    model = args.model

    # Execute the specified command
    args.func(args)
