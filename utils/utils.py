import argparse
import ast

import os
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss


def get_parser_classification():
    parser = argparse.ArgumentParser(description="Adversarial attack by infrared perturbations", fromfile_prefix_chars='@')
    parser.add_argument("--dataset", type=str, default="GTSRB", choices=["GTSRB", "LISA"],
                        help="the target dataset should be specified for a digital attack")
    parser.add_argument("--optimizer", type=str, default="lrs", choices=["pso", "es", "ga", "lrs", "rnd"],
                        help="The optimizer to use.")
    parser.add_argument("--mesh_count", type=int, default=16,
                        help="The number of columns / row in which to split the image to locate pixels (only applicable if pixel perturbation type)")
    parser.add_argument("--target_model", type=str, default="GtsrbCNN",
                        help="Select the classifier architecture to attack.",
                        choices=["GtsrbCNN", "ResNet50", "ConvNeXt", "SwinTransformer", "LisaCNN"])
    parser.add_argument("--adv_retrained", action="store_true",
                        help="Set this flag if you want to load the adversarially retrained version of the target model")
    parser.add_argument("--patch_count", type=int, default=192, help="number of perturbations")
    parser.add_argument("--max_queries", type=int, default=1000,
                        help="Maximum number of queries of the target model (per start of the optimizer)")
    parser.add_argument("--target_mapping", type=valid_mapping_classification, default='[("*", "*")]',
                        help="Specify one or multiple desired mis-classification label mappings. See README.md for more "
                             "details.")
    parser.add_argument("--lux", type=int, default=10,
                        help="The intensity of the ambient light to assume for the IR attack [Lux]")
    parser.add_argument("--save_dir", type=str, default="./output",
                        help="Set the path in which the results of the attack will be stored")
    parser.add_argument("--enforce_query_budget", action="store_true",
                        help="Set this flag if you want to enforce the given query budget to be fully used")
    parser.add_argument("--instances_per_class", type=int, default=25,
                        help="Set this value to set the number of instances per class to evaluate on. Only makes a difference for a digital attack on GTSRB (as the LISA dataset is way to small anyway).")
    parser.add_argument("--seed", type=int, default=42, help="Specify a seed for all PRNGs explicitly")

    return parser


def valid_mapping_classification(mappings: str):
    mappings = ast.literal_eval(mappings)

    if isinstance(mappings, str):
        mappings = ast.literal_eval(mappings)

    if isinstance(mappings, list):
        for mapping in mappings:
            source, target = mapping
            if not isinstance(mapping, tuple) or len(mapping) != 2 or (isinstance(source, int) and source < 0) or (
                    not isinstance(source, int) and source != '*') or (isinstance(target, int) and target < 0) or (
                    not isinstance(target, int) and target not in ['*', 'auto']):
                raise argparse.ArgumentTypeError(f"Mapping {mapping} is not valid.")
        if len(list(zip(*mappings))[0]) != len(set(list(zip(*mappings))[0])):
            raise argparse.ArgumentTypeError("Target mapping is not injective.")
        return mappings
    else:
        raise argparse.ArgumentTypeError("Target mapping is not a list.")


def get_parser_detection():
    parser = argparse.ArgumentParser(description="Adversarial object detection attack by infrared perturbations",
                                     fromfile_prefix_chars='@')
    parser.add_argument("--dataset", type=str, default="GTSDB", choices=["GTSDB", "Mapillary"],
                        help="the target dataset should be specified for a digital attack")
    parser.add_argument("--optimizer", type=str, default="lrs", choices=["pso", "es", "ga", "lrs", "rnd"],
                        help="The optimizer to use")
    parser.add_argument("--mesh_count", type=int, default=16,
                        help="The number of columns / row in which to split the image to locate pixels (only applicable if pixel perturbation type)")
    parser.add_argument("--patch_count", type=int, default=192, help="Number of perturbations (k)")
    parser.add_argument("--max_queries", type=int, default=1000,
                        help="Maximum number of queries of the target model (per start of the optimizer)")
    parser.add_argument("--lux", type=int, default=10,
                        help="The intensity of the ambient light to assume for the infrared attack [Lux]")
    parser.add_argument("--save_dir", type=str, default="./output",
                        help="Set the path in which the results of the attack will be stored")
    parser.add_argument("--store_predictions", action="store_true",
                        help="Set this flag to store the ground-truth prediction and the prediction on the adversarially perturbed images alongside the perturbed images")
    parser.add_argument("--target_mapping", type=valid_mapping_detection, default='[("*", "*")]',
                        help="Specify one or multiple desired mis-classification label mappings. See README for more details.")
    parser.add_argument("--enforce_query_budget", action="store_true",
                        help="Set this flag if you want to enforce the given query budget to be fully used")
    parser.add_argument("--seed", type=int, default=42, help="Specify a seed for all PRNGs explicitly")
    parser.add_argument("--instances", type=int, default=0,
                        help="Reduces the number of files the experiment is executed on.")

    subparsers = parser.add_subparsers()
    transferability_parser = subparsers.add_parser('transferability')
    transferability_parser.add_argument("--adv_dir", type=str, default=".",
                                        help="A folder to load perturbed images from. Only used for transferability evaluation.")
    transferability_parser.add_argument("--src_dataset", type=str, default="Mapillary",
                                        choices=["Mapillary", "GTSDB"], )

    return parser


def valid_mapping_detection(mappings: str):
    mappings = ast.literal_eval(mappings)

    if isinstance(mappings, list):
        target_labels = list(zip(*mappings))[1]  # Precompute for efficiency reasons
        if "hide" in target_labels and len(set(target_labels)) != 1:
            raise argparse.ArgumentTypeError(
                "You can only perform a hiding attack OR an untargeted/targeted attack. Mixing both is not supported yet.")
        for mapping in mappings:
            source, target = mapping
            if not isinstance(mapping, tuple) or len(mapping) != 2 or (isinstance(source, int) and source < 0) or (
                    not isinstance(source, int) and source != '*') or (isinstance(target, int) and target < 0) or (
                    not isinstance(target, int) and target not in ['*', 'hide']):
                raise argparse.ArgumentTypeError(f"Mapping {mapping} is not valid.")
        if len(list(zip(*mappings))[0]) != len(set(list(zip(*mappings))[0])):
            raise argparse.ArgumentTypeError("Target mapping is not injective.")
        return mappings
    else:
        raise argparse.ArgumentTypeError("Target mapping is not a list.")


def load_mask():
    with open("./utils/sign_masks.pkl", "rb") as f:
        position_list, mask_list = pickle.load(f)
    return position_list, mask_list


def get_mask_type(database, label):
    if database == "GTSRB" or database == "GTSDB":
        if label in [0, 1, 2, 3,
                     4, 5, 6, 7,
                     8, 9, 10, 15,
                     16, 17, 32, 33,
                     34, 35, 36, 37,
                     38, 39, 40, 41,
                     42]:
            return 0
        if label in [11, 18, 19, 20,
                     21, 22, 23, 24,
                     25, 26, 27, 28,
                     29, 30, 31]:
            return 6
        if label in [13]:
            return 1
        if label in [14]:
            return 2
        if label in [12]:
            return 5
    elif database == "LISA":
        if label in [0, 2, 3, 4, 7, 13, 14]:
            return 5
        if label in [1, 6, 8, 9, 10, 11]:
            return 4
        if label in [15]:
            return 1
        if label in [12]:
            return 2
        if label in [5]:
            return 3
    elif database == "Mapillary":
        if label in [0, 1, 2, 3, 4, 5, 6, 7]:
            return 0
        if label in [8]:
            return 2
        if label in [9]:
            return 4
    return None


class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.2):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes, smoothing=0.2):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size()[0], n_classes),
                                  device=targets.device) \
                .fill_(smoothing / (n_classes - 1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
                                                         self.smoothing)
        lsm = F.log_softmax(inputs, -1)
        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)
        loss = -(targets * lsm).sum(-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss


def fix_randomness(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(ignore=(None,)):
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    import torch

    cuda_available = torch.cuda.is_available()
    mps_available = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()

    if cuda_available and "cuda" not in ignore:
        device = torch.device("cuda")
        torch.cuda.set_device(0)
    elif mps_available and "mps" not in ignore:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device
