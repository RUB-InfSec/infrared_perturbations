import os
from collections import Counter
from math import ceil

import torch
import pickle
import cv2
import numpy as np
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from attacks.DetectionAttack import DetectionAttack
from dataset.classification.traffic_sign_dataset import TrafficSignDataset
from dataset.detection.FasterRCNN import Collator, plot_predictions_faster_rcnn
from dataset.detection.gtsdb import GTSDBDataset
from optimizers.EvolutionStrategy import EvolutionStrategy
from optimizers.GeneticAlgorithm import GeneticAlgorithm
from optimizers.LocalRandomSearch import LocalRandomSearch
from optimizers.RandomBaseline import RandomBaseline
from optimizers.ParticleSwarmOptimization import ParticleSwarmOptimization
from perturbations.PixelPerturbation import PixelPerturbation
from utils import utils
from utils.cnn_utils import load_trained_detector
from utils.ir_utils import equalize_and_transform_to_ir, back_transform
from utils.utils import get_mask_type, load_mask, fix_randomness, get_parser_detection

position_list, mask_list = load_mask()


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


def attack_step(attack_image, coords, mapping, enforce_query_budget, image_ir=None, true_label=0, verbose=False):
    num_query = 0
    global_best_solution = float('inf')
    global_best_position = None

    x_min, x_max, y_min, y_max = np.min(coords[0]), np.max(coords[0]), np.min(coords[1]), np.max(coords[1])
    max_speed = 1.5 * (x_max - x_min) / 32

    adv_perturbation = PixelPerturbation(patch_count, 3, x_min, x_max, y_min, y_max, mesh_count, image_ir, True,
                                             args)

    if mapping[1] == "hide":
        attack_type = "hide"
    elif mapping[1] == "*":
        attack_type = "untargeted"
    else:
        attack_type = "targeted"

    # For now, we move this inside of the for loop, as the transform_num of attack_instance can get modified by pso.
    # However, a cleaner solution for that is required.
    attack_instance = DetectionAttack(adv_perturbation, attack_type, attack_image, coords, model, true_label,
                                      0, pre_process, 0.5,
                                      target_label=mapping[1] if attack_type == "targeted" else None)

    if optimizer_type == 'es':
        optimizer = EvolutionStrategy(adv_perturbation, attack_instance, enforce_query_budget, max_queries, verbose)
    elif optimizer_type == 'lrs':
        optimizer = LocalRandomSearch(adv_perturbation, attack_instance, enforce_query_budget, max_queries, verbose)
    elif optimizer_type == 'ga':
        optimizer = GeneticAlgorithm(adv_perturbation, attack_instance, 10, enforce_query_budget, max_queries,
                                     verbose)
    elif optimizer_type == 'rnd':
        optimizer = RandomBaseline(adv_perturbation, attack_instance, enforce_query_budget, max_queries=1, verbose=verbose)
    elif optimizer_type == 'pso':
        optimizer = ParticleSwarmOptimization(adv_perturbation, attack_instance, 10, max_speed, enforce_query_budget,
                                              max_queries, verbose)
    else:
        raise NotImplementedError

    best_solution, best_pos, succeed, query, predicted_label = optimizer.optimize()

    print(f"Best loss {best_solution:.04f} was {'successful' if succeed else 'a failure'}.")
    if best_solution < global_best_solution:
        global_best_solution = best_solution
        global_best_position = best_pos
    num_query += query

    adv_img, perturbation_matrix = adv_perturbation.draw(attack_image, coords, global_best_position)

    return adv_img, succeed, num_query, perturbation_matrix, predicted_label, global_best_solution


def attack_digital(args):
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        for name in os.listdir(save_dir):
            os.remove(os.path.join(save_dir, name))

    batch_size = 64

    if args.dataset == "GTSDB":
        # GTSDB is small enough anyway, thus, we use all images
        dataset = GTSDBDataset(root='./dataset/detection/GTSDB/', train=True, num_classes=num_classes)
        selected_img_idx = torch.IntTensor(range(len(dataset.imgs)))
    elif args.dataset == "Mapillary":
        # Use only 25 images per class for Mapillary. These are the indices of images for our classes from the original dataset with bboxes of a minimum of 32 pixels.
        # indices_per_class = [
        #     [11, 130, 142, 191, 246, 246, 263, 304, 441, 477, 532, 557, 648, 664, 890, 1158, 1228, 1305, 1356, 1426,
        #      1426, 1438, 1533, 1623, 1623],
        #     [263, 375, 375, 573, 606, 606, 648, 934, 934, 1470, 1470, 1766, 1788, 1788, 1826, 1826, 1858, 1858, 1884,
        #      1988, 2740, 2838, 2849, 3051, 3053],
        #     [34, 49, 161, 198, 213, 220, 220, 307, 317, 474, 533, 555, 632, 640, 668, 688, 710, 720, 754, 808, 818, 865,
        #      905, 1060, 1090],
        #     [4, 36, 41, 52, 53, 59, 124, 138, 138, 162, 179, 215, 218, 226, 228, 230, 276, 279, 288, 305, 335, 345, 346,
        #      374, 393],
        #     [10, 36, 38, 85, 98, 100, 112, 130, 166, 178, 185, 202, 212, 236, 236, 261, 267, 274, 303, 344, 347, 362,
        #      367, 373, 388],
        #     [8, 47, 57, 61, 67, 83, 121, 140, 140, 143, 222, 231, 294, 302, 322, 370, 382, 382, 394, 407, 408, 408, 465,
        #      503, 538],
        #     [23, 141, 234, 234, 248, 250, 284, 350, 351, 376, 392, 617, 623, 626, 669, 677, 734, 874, 918, 976, 1033,
        #      1039, 1058, 1069, 1146],
        #     [91, 92, 92, 114, 119, 122, 130, 208, 272, 343, 353, 418, 439, 473, 476, 476, 535, 650, 664, 671, 671, 685,
        #      806, 910, 910],
        #     [2, 6, 19, 24, 24, 31, 48, 53, 66, 79, 79, 81, 101, 102, 111, 126, 126, 127, 131, 134, 149, 151, 153, 155,
        #      157]]

        with open("./dataset/detection/Mapillary/mapillary_subset.pkl", "rb") as f:
            imgs_loaded, targets_loaded = pickle.load(f)

        dataset = TrafficSignDataset(imgs_loaded, targets_loaded, transform=None)
        bbox_size = 32
        selected_img_idx = torch.IntTensor(range(len(dataset.x)))

    if args.instances > 0:
        data_subset = Subset(dataset, selected_img_idx[:args.instances])
    else:
        data_subset = Subset(dataset, selected_img_idx)

    dataloader = DataLoader(data_subset, batch_size=batch_size, shuffle=False, collate_fn=Collator(utils.get_device()))

    for batch_id, batch in tqdm(enumerate(dataloader)):
        images, labels = batch

        # Generate IR-transformed versions of the images
        images = back_transform(
            batch[0])  # The equalize_and_transform_to_ir method takes a list of BGR np.uint8 arrays, not a tensor...
        images, images_ir = equalize_and_transform_to_ir(images, lux)

        for i, (image, image_ir, target) in enumerate(zip(images, images_ir, labels)):
            # Since YOLO was trained on RGB and not BGR images, convert them
            image, image_ir = cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cv2.cvtColor(image_ir, cv2.COLOR_BGR2RGB)

            if args.store_predictions and len(target['labels']) > 0:
                if args.dataset == "Mapillary":
                    results = model(torch.unsqueeze(pre_process(image), dim=0))
                    im_array = results[0].plot()  # plot a numpy array of predictions
                    cv2.imwrite(
                        f"{save_dir}/{selected_img_idx[batch_id * batch_size + i]}_ground_truth_prediction.bmp",
                        cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR))
                elif args.dataset == "GTSDB":
                    result = model(pre_process(image).to(utils.get_device()))[0]
                    im_array = np.array(to_pil_image(pre_process(image)))
                    im_array = plot_predictions_faster_rcnn(im_array, result)
                    cv2.imwrite(
                        f"{save_dir}/{selected_img_idx[batch_id * batch_size + i]}_ground_truth_prediction.bmp",
                        cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR))

            for j, (bbox, label) in enumerate(zip(target['boxes'], target['labels'])):
                mapping = get_mapping_for_label(label)
                if mapping is not None:
                    boundary_box_size = (ceil((bbox[2] - bbox[0]).item()), ceil((bbox[3] - bbox[1]).item()))

                    if args.dataset == "Mapillary" and boundary_box_size[0] < bbox_size and boundary_box_size[
                        1] < bbox_size:
                        continue

                    x0, y0 = int(bbox[0].item()), int(bbox[1].item())

                    # Load the 32x32 mask that corresponds to the given traffic sign type and scale it to the bounding box
                    mask = np.zeros((32, 32))
                    mask[position_list[get_mask_type(args.dataset, label.item())]] = 1
                    mask = cv2.resize(mask, (boundary_box_size[0], boundary_box_size[1]))

                    # Render the mask that has the size of the bounding box onto the full image
                    big_mask = np.zeros((image.shape[0], image.shape[1]))
                    big_mask[y0:y0 + boundary_box_size[1], x0:x0 + boundary_box_size[0]] += mask

                    adv_img, success, query_ctr, _, predicted_label, _ = attack_step(image, np.nonzero(big_mask == 1),
                                                                                     mapping, args.enforce_query_budget,
                                                                                     image_ir=image_ir,
                                                                                     true_label=label.item(),
                                                                                     verbose=False)

                    cv2.imwrite(
                        f"{save_dir}/{selected_img_idx[batch_id * batch_size + i]}_{j}_{label}-{predicted_label}_{query_ctr}_{success}.bmp",
                        cv2.cvtColor(adv_img, cv2.COLOR_RGB2BGR))

                    if args.store_predictions:
                        if args.dataset == "Mapillary":
                            results = model(torch.unsqueeze(pre_process(adv_img), dim=0))
                            im_array = results[0].plot()  # plot a numpy array of predictions
                            cv2.imwrite(
                                f"{save_dir}/{selected_img_idx[batch_id * batch_size + i]}_{j}_{label}-{predicted_label}_{query_ctr}_{success}_prediction.bmp",
                                cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR))
                        elif args.dataset == "GTSDB":
                            result = model(pre_process(adv_img).to(utils.get_device()))[0]
                            im_array = np.array(to_pil_image(pre_process(adv_img)))
                            im_array = plot_predictions_faster_rcnn(im_array, result)
                            cv2.imwrite(
                                f"{save_dir}/{selected_img_idx[batch_id * batch_size + i]}_{j}_{label}-{predicted_label}_{query_ctr}_{success}_prediction.bmp",
                                cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR))

    files = [file for file in os.listdir(save_dir) if file[:-4].split('_')[-1] != 'prediction']
    print("Attack success rate: ", end='')
    print(f"{Counter(map(lambda x: x[:-4].split('_')[-1], files))['True'] / len(files):.03f}")
    print("Average number of queries: ", end='')
    print(f"{sum(list(map(lambda x: int(x[:-4].split('_')[-2]), files))) / len(files):.03f}")


def evaluate_transferability(args):
    pre_process, num_classes, model = load_trained_detector(args.dataset)
    save_dir = args.save_dir

    image_paths = [adv_img for adv_img in os.listdir(f'{args.adv_dir}') if "prediction" not in adv_img]

    if args.src_dataset == "GTSDB":
        src_dataset = GTSDBDataset(root='./dataset/detection/GTSDB/', train=True, num_classes=num_classes)
    elif args.src_dataset == "Mapillary":
        # Use only 25 images per class for Mapillary
        with open("./dataset/detection/Mapillary/mapillary_subset.pkl", "rb") as f:
            imgs_loaded, targets_loaded = pickle.load(f)

        src_dataset = TrafficSignDataset(imgs_loaded, targets_loaded, transform=None)

    correct, total = 0, 0

    for image_path in tqdm(image_paths):
        _, target_class = image_path.split('_')[2].split('-')
        _, label = src_dataset.__getitem__(idx := int(image_path.split('_')[0]))

        try:
            bbox = label['boxes'][int(image_path.split('_')[1])].cpu()
        except IndexError:
            continue

        perturbed_image = cv2.cvtColor(cv2.imread(os.path.join(args.adv_dir, image_path)), cv2.COLOR_BGR2RGB)
        height, width, _ = pre_process(perturbed_image).permute(1, 2, 0).numpy().shape

        if args.dataset == "Mapillary":
            result = model(torch.unsqueeze(pre_process(perturbed_image), dim=0), verbose=False)[0]
            if result.boxes.xyxyn.nelement() == 0:
                predicted_class = 'hide'
            else:
                ious = torchvision.ops.box_iou(bbox.unsqueeze(0), result.boxes.xyxy.to('cpu')).squeeze(0)
                matching_indices = torch.argwhere(ious).squeeze(1)
                ious, confs, labels = ious[matching_indices], result.boxes.conf[matching_indices], result.boxes.cls[
                    matching_indices]
                hidden = all(
                    [best_match_iou < 0.5 or confidence < 0.3 for best_match_iou, confidence in zip(ious, confs)])
                if hidden:
                    predicted_class = 'hide'
                else:
                    predicted_class = labels[torch.argmax(ious)]
            if args.store_predictions:
                im_array = result.plot()  # plot a numpy array of predictions
                cv2.imwrite(f"{save_dir}/{idx}_{predicted_class}_prediction.bmp",
                            cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR))
        elif args.dataset == "GTSDB":
            result = model(pre_process(perturbed_image).to(utils.get_device()))[0]
            if torch.numel(result['boxes']) == 0:
                predicted_class = 'hide'
            else:
                predicted_bbox = result['boxes'].to('cpu')
                ious = torchvision.ops.box_iou(bbox.unsqueeze(0), predicted_bbox).squeeze(0)
                matching_indices = torch.argwhere(ious).squeeze(1)
                ious, confs, labels = ious[matching_indices], result['scores'][matching_indices], result['labels'][
                    matching_indices]
                hidden = all(
                    [best_match_iou < 0.5 or confidence < 0.3 for best_match_iou, confidence in zip(ious, confs)])
                if hidden:
                    predicted_class = 'hide'
                else:
                    predicted_class = labels[torch.argmax(ious)]
            if args.store_predictions:
                im_array = np.array(to_pil_image(pre_process(perturbed_image)))
                im_array = plot_predictions_faster_rcnn(im_array, result)
                cv2.imwrite(f"{save_dir}/{idx}_{predicted_class}_prediction.bmp",
                            cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR))

        total += 1
        correct += 1 if predicted_class == target_class else 0

    print(
        f"Transferability for {args.adv_dir.split('/')[-1]} -> {args.dataset}: {100 * correct / total:.2f} %")


if __name__ == '__main__':
    print(f'Using {utils.get_device()} for computations.')

    parser = get_parser_detection()

    args = parser.parse_args()
    optimizer_type = args.optimizer
    mesh_count = args.mesh_count
    patch_count = args.patch_count
    max_queries = args.max_queries
    lux = args.lux
    save_dir = args.save_dir
    instances = args.instances
    target_mapping = args.target_mapping
    src_labels = list(zip(*target_mapping))[0]  # Precompute for efficiency reasons
    target_labels = list(zip(*target_mapping))[1]  # Precompute for efficiency reasons

    pre_process, num_classes, model = load_trained_detector(args.dataset)

    fix_randomness(args.seed)

    if hasattr(args, "adv_dir"):  # Quite hacky way to determine that the subparser was called
        evaluate_transferability(args)
    else:
        attack_digital(args)
