import copy
import os
import random
from abc import abstractmethod, ABC

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


class AdversarialPerturbation(ABC):
    def __init__(self, patch_count, patch_dimensionality, blur_coefficient, x_min, x_max, y_min, y_max, image_ir, od,
                 args):
        self.patch_count = patch_count
        self.blur_coefficient = blur_coefficient
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.od = od
        self.patch_dimensionality = patch_dimensionality
        self.image_ir = image_ir
        self.args = args
        assert self.image_ir is not None

    def draw(self, img, mask, positions):
        """
        Draw the adversarial perturbation onto the image.

        Args:
            img: the image to draw the perturbation on
            mask(list): The coordinates, i.e., indices of the points where mask == 1 (given as a tuple of two lists - first entry contains x-indices, second one y-indices)
            positions: a list of coordinates identifying the perturbations

        Returns:

        """
        perturbation_matrix = np.ones_like(img) * 255
        perturbation_area = np.zeros_like(img, dtype=np.uint8)

        mask_array = self.fill_mask_array(mask, positions)
        inside_list = np.nonzero(mask_array)
        x_list, y_list = mask[0][inside_list], mask[1][inside_list]
        perturbation_area[x_list, y_list] = 255

        res = copy.deepcopy(img)
        res[mask] = self.image_ir[mask]
        res[x_list, y_list] = img[x_list, y_list]
        perturbation_matrix[x_list, y_list, :] = 0

        return perturbation_blur(res, perturbation_area, self.blur_coefficient), perturbation_matrix

    def image_transformation(self, img, position, pos_list, pre_process):

        h, w, _ = img.shape
        res_images = []
        res_masks = []

        if self.od:
            # When we're in the OD case, the input images are in RGB, not BGR.
            # As the following transformations partially rely on the input images being BGR, transform the image to BGR and back after all transformations are done.
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # perturbation
        adv_img, _ = self.draw(img, pos_list, position)

        # mask
        mask_img = np.zeros_like(adv_img, dtype=np.float32)
        mask_img[pos_list[0], pos_list[1]] = 1

        res_images.append(adv_img)
        res_masks.append(mask_img)

        if self.od:
            res_images[0] = cv2.cvtColor(res_images[0], cv2.COLOR_BGR2RGB)

        res_images[0] = pre_process(res_images[0])
        target_size = res_images[0].size()
        res_masks[0] = cv2.resize(res_masks[0], (target_size[2], target_size[1]))

        stacks = torch.stack(res_images, dim=0)

        return stacks, [np.nonzero(mask_image.sum(axis=2) > 0) for mask_image in res_masks]

    @abstractmethod
    def fill_mask_array(self, mask, positions):
        """
        Return an array that contains values > 0 everywhere where any patch covers the image

        Args:
            mask: The mask to crop the image
            positions: The coordinates of the patches

        Returns: an array that contains values > 0 everywhere where any patch covers the image

        """
        pass

    @abstractmethod
    def initial_solution(self):
        """

        Returns: a valid initial solution

        """
        pass

    @abstractmethod
    def repair(self, x):
        """
        Repair the solution x if it is invalid.

        Args:
            x: The solution to repair

        Returns:
            - x if x is valid, else a repaired version of x
            - a boolean value that is True, if x could not be repaired

        """
        pass


def contains(vertices, p):
    vertices = np.append(vertices, vertices[0].reshape(1, 2), 0)
    res = np.zeros(p.shape[0], dtype=np.bool_)
    x = p[0:, 0]
    for i in range(len(vertices) - 1):
        (x1, _), (x2, _) = vertices[i], vertices[i + 1]
        vector1 = vertices[i + 1] - vertices[i]
        vector2 = p - vertices[i]
        cross = np.cross(vector1, vector2)
        res ^= ((x1 <= x) & (x <= x2) & (cross < 0)) | ((x2 <= x) & (x <= x1) & (cross > 0))

    return res


def perturbation_blur(image, perturbation_area, coefficient):
    blurred_img = cv2.GaussianBlur(image, (coefficient, coefficient), 0)
    gray_img = cv2.cvtColor(perturbation_area, cv2.COLOR_BGR2GRAY)
    threshold = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _mask = np.zeros_like(image, dtype=np.uint8)
    cv2.drawContours(_mask, contours, -1, (255, 255, 255), coefficient)
    return np.where(_mask == np.array([255, 255, 255]), blurred_img, image)

