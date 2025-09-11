import torch

import os
import pickle
from collections import Counter

import cv2
import numpy as np

from attacks.ClassificationAttack import ClassificationAttack
from dataset.classification import gtsrb
from optimizers.EvolutionStrategy import EvolutionStrategy
from optimizers.GeneticAlgorithm import GeneticAlgorithm
from optimizers.LocalRandomSearch import LocalRandomSearch
from optimizers.RandomBaseline import RandomBaseline
from optimizers.ParticleSwarmOptimization import ParticleSwarmOptimization
from perturbations.PixelPerturbation import PixelPerturbation
from utils import utils
from utils.cnn_utils import load_trained_classifier
from utils.ir_utils import equalize_and_transform_to_ir
from utils.utils import get_mask_type, load_mask, fix_randomness, get_parser_classification

position_list, mask_list = load_mask()

parser = get_parser_classification()
args = parser.parse_args()

adv_retrained = args.adv_retrained
dataset = args.dataset
target_model = args.target_model
optimizer_type = args.optimizer
mesh_count = args.mesh_count
patch_count = args.patch_count
max_queries = args.max_queries
target_mapping = args.target_mapping
lux = args.lux
enforce_query_budget = args.enforce_query_budget
instances_per_class = args.instances_per_class
save_dir = args.save_dir
src_labels = list(zip(*target_mapping))[0]  # Precompute for efficiency reasons
seed = args.seed

pre_process, model = load_trained_classifier(dataset, target_model, adv_retrained)

os.makedirs(save_dir, exist_ok=True)

assert optimizer_type in ['pso', 'es', 'lrs', 'ga', 'rnd']

def get_mapping_for_label(label):
    """
    Return the mapping, i.e., the tuple (src_label, target_label), from the `--target_mapping` parameter that is the
    best match for the given label.

    Args:
        label: The label for which to find a mapping for

    Returns: one of the following (in descending order of precedence)
        1. a mapping that has `label` as its `src_label`
        2. a mapping that has '*' as its `src_label`
        3. None, if there is no matching mapping

    """
    if label in src_labels:
        return [m for m in target_mapping if m[0] == label][0]
    elif '*' in src_labels:
        return [m for m in target_mapping if m[0] == '*'][0]
    else:
        return None


def attack_step(attack_image, label, coords, targeted_attack=False, enforce_query_budget=False, target=None,
                image_ir=None, verbose=False):
    r"""
    Physical-world adversarial attack by IR perturbation.

    Args:
        attack_image: The image to be attacked.
        label: The ground-truth label of attack_image.
        coords: The coordinates of the points where mask == 1.
        targeted_attack: Targeted / Non-targeted attack.
        enforce_query_budget: Whether the optimizer should the full query budget or terminate on success
        target: The target class (if targeted attack)
        image_ir: The IR-transformed version of the attack image (if necessary)
        verbose: Whether the optimizer's output should be verbose

    Returns:
        adv_img: The generated adversarial image.
        succeed: Whether the attack is successful.
        num_query: Number of queries.
        perturbation_matrix:
        predicted_label: The label that has the highest probability in the model's prediction for adv_img
    """
    num_query = 0
    global_best_solution = float('inf')
    global_best_position = None

    adv_perturbation = PixelPerturbation(patch_count, 3, 0, 32, 0, 32, mesh_count, image_ir, False,
                                         args)

    # For now, we move this inside of the for loop, as the transform_num of attack_instance can get modified by pso.
    # However, a cleaner solution for that is required.
    attack_instance = ClassificationAttack(adv_perturbation, attack_image, coords, model, label, 0,
                                           pre_process, targeted_attack, target)

    if optimizer_type == 'es':
        optimizer = EvolutionStrategy(adv_perturbation, attack_instance, enforce_query_budget, max_queries, verbose)
    elif optimizer_type == 'lrs':
        optimizer = LocalRandomSearch(adv_perturbation, attack_instance, enforce_query_budget, max_queries, verbose)
    elif optimizer_type == 'ga':
        optimizer = GeneticAlgorithm(adv_perturbation, attack_instance, 10, enforce_query_budget, max_queries,
                                     verbose)
    elif optimizer_type == 'pso':
        optimizer = ParticleSwarmOptimization(adv_perturbation, attack_instance, 10, 1.5,
                                          enforce_query_budget,
                                          max_queries, verbose)
    elif optimizer_type == 'rnd':
        optimizer = RandomBaseline(adv_perturbation, attack_instance, enforce_query_budget, max_queries, verbose)
    else:
        raise NotImplementedError()

    best_solution, best_pos, succeed, query, predicted_label = optimizer.optimize()

    print(f"Best loss {best_solution:.04f} was {'successful' if succeed else 'a failure'}.")
    if best_solution < global_best_solution:
        global_best_solution = best_solution
        global_best_position = best_pos
    num_query += query

    adv_img, perturbation_matrix = adv_perturbation.draw(attack_image, coords, global_best_position)

    return adv_img, succeed, num_query, perturbation_matrix, predicted_label, global_best_solution


def attack(image, label, mask, target, enforce_query_budget, image_ir=None, verbose=False):
    query_ctr, success, predicted_label = 0, False, label
    if target == '*' or target == 'auto':
        # First run untargeted attack
        adv_img, success, num_query, perturbation_matrix, predicted_label, best_loss = attack_step(image, label, mask,
                                                                                             enforce_query_budget=enforce_query_budget,
                                                                                             image_ir=image_ir,
                                                                                             verbose=verbose)
        query_ctr += num_query
        if not success:
            print('Attack failed! :(')
            return adv_img, predicted_label, query_ctr, success, perturbation_matrix, best_loss
        else:
            print(f"Attack successful with predicted label {predicted_label}.")

        if target == 'auto':
            # Run a targeted attack on the predicted label afterward
            target, success = int(predicted_label), False
    if isinstance(target, int):
        # Target attack with given target label
        adv_img, success, num_query, perturbation_matrix, predicted_label, best_loss = attack_step(image, label, mask,
                                                                                             targeted_attack=True,
                                                                                             enforce_query_budget=enforce_query_budget,
                                                                                             target=target,
                                                                                             image_ir=image_ir,
                                                                                             verbose=verbose)
        query_ctr += num_query
    return adv_img, predicted_label, query_ctr, success, perturbation_matrix, best_loss


def attack_digital(enforce_query_budget=False):
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        for name in os.listdir(save_dir):
            os.remove(os.path.join(save_dir, name))

    with open(f'./dataset/classification/{dataset.lower()}_test.pkl', 'rb') as ds:
        test_data = pickle.load(ds)
        images, labels = test_data['data'], test_data['labels']

    images, images_ir = equalize_and_transform_to_ir(images, lux)

    if dataset == "GTSRB":
        # For GTSRB randomly chose instances_per_class samples for each of the 43 classes to speed up the process
        indices = np.array([], dtype=int)
        for label in range(gtsrb.class_n):
            try:
                indices = np.append(indices, np.random.choice(np.nonzero(labels == label)[0], instances_per_class,
                                                              replace=False))
            except ValueError:
                print(f"There are not enough samples for class {label} to sample (< {instances_per_class})")
                indices = np.append(indices, np.nonzero(labels == label)[0])
    else:
        # The LISA dataset is way too small anyway, so we don't reduce it further
        indices = list(range(len(images)))

    for i in indices:
        print(f"Image {i}")
        mask_type = get_mask_type(dataset, labels[i])
        mapping = get_mapping_for_label(labels[i])
        if mapping is not None:
            adv_img, predicted_label, query_ctr, success, _, _ = attack(images[i], labels[i], position_list[mask_type],
                                                                        mapping[1], enforce_query_budget,
                                                                        image_ir=images_ir[i], verbose=False)
            cv2.imwrite(f"{save_dir}/{i}_{labels[i]}-{predicted_label}_{query_ctr}_{success}.bmp", adv_img)

    print("Attack success rate: ", end='')
    print(f"{Counter(map(lambda x: x[:-4].split('_')[-1], os.listdir(save_dir)))['True'] / len(os.listdir(save_dir)):.03f}")
    print("Average number of queries: ", end='')
    print(f"{sum(list(map(lambda x: int(x[:-4].split('_')[-2]), os.listdir(save_dir)))) / len(os.listdir(save_dir)):.03f}")


if __name__ == '__main__':
    print(f'Using {utils.get_device()} for computations.')

    fix_randomness(42)

    attack_digital(enforce_query_budget)

