import argparse
import json
import os
import pickle
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

from dataset.classification.traffic_sign_dataset import TrafficSignDataset
from model.classification.GtsrbCNN import GtsrbCNN
from utils import utils
from utils.ir_utils import equalize_and_transform_to_ir
from utils.utils import SmoothCrossEntropyLoss, load_mask

# Data is BGR and hence the normalization is applied to the dimensions correctly
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.31100202, 0.31318352, 0.3089177),
                                                     (0.271555, 0.26502392, 0.25716692))
                                ])

with open('./dataset/classification/classes.json', 'r') as config:
    params = json.load(config)
    class_n = params['GTSRB']['class_n']
    position_list, _ = load_mask()

loss_fun = SmoothCrossEntropyLoss(smoothing=0.1)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.05)
        nn.init.constant_(m.bias, 0.05)


def model_epoch(training_model, data_loader, train=False, optimizer=None):
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

    return acc, loss


def test_model(args):
    trained_model = initialize_model(args.model)
    trained_model.load_state_dict(
        torch.load(
            f'./model/classification/model_gtsrb_{args.model}{"_adv_retrain" if args.robust else ""}.pth',
            map_location=torch.device(utils.get_device())))

    with open('./dataset/classification/GTSRB/gtsrb_test.pkl', 'rb') as f:
        test = pickle.load(f)
        test_data, test_labels = test['data'], test['labels']

    # Determine accuracy on IR-transformed dataset instead
    if args.ir:
        _, test_data = equalize_and_transform_to_ir(test_data, args.lux)

    # Determine accuracy on given folder instead (e.g., to test accuracy on generated adversarial examples)
    if args.dir:
        if os.path.exists(args.dir) and os.path.isdir(args.dir):
            test_data, test_labels = [], []
            for img in os.listdir(args.dir):
                image = cv2.imread(f'{args.dir}/{img}')
                if image.shape != (32, 32):
                    image = cv2.resize(image, (32, 32))
                test_data.append(image)
                test_labels.append(int(img.split('_')[1].split('-')[0]))
            test_data = np.array(test_data)
        else:
            print("The specified directory does not exist!")
            return None

    test_data = np.array([img for img in test_data])

    test_set = TrafficSignDataset(test_data, test_labels, transform)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    trained_model.eval()
    with torch.no_grad():
        test_acc, _ = model_epoch(trained_model, test_loader)

    test_acc = round(float(test_acc / test_set.__len__()), 4)

    if args.verbose:
        print(f'Test Acc: {test_acc}')
        return None
    else:
        return test_acc

def initialize_model(model):
    if model == "GtsrbCNN":
        trained_model = GtsrbCNN(n_class=class_n).to(utils.get_device()).apply(weights_init)
    elif model == "ResNet50":
        trained_model = torchvision.models.resnet50().to(utils.get_device())
    elif model == "ConvNeXt":
        trained_model = torchvision.models.convnext_base().to(utils.get_device())
    elif model == "SwinTransformer":
        trained_model = torchvision.models.swin_b().to(utils.get_device())
    return trained_model


def get_parser():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="GTSRB model training and validation")
    parser.add_argument("--model", type=str, default="GtsrbCNN",
                        help="The classifier architecture to use.",
                        choices=['GtsrbCNN', 'ResNet50', 'SwinTransformer', 'ConvNeXt'])

    parser.add_argument("--verbose", action="store_true",
                  help="Set this flag if you want to print verbose messages.")

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
    parser_testing.set_defaults(func=test_model)

    return parser

if __name__ == '__main__':
    print(f'Using {utils.get_device()} for computations.')

    parser = get_parser()
    args = parser.parse_args()

    model = args.model

    # Execute the specified command
    args.func(args)
