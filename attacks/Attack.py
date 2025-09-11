from abc import ABC, abstractmethod

from utils import utils


class Attack(ABC):
    def __init__(self, adv_perturbation, img, mask_coords, model, transform_num, pre_process, targeted,
                 target=None):
        self.img = img
        self.mask_coords = mask_coords
        self.model = model
        self.transform_num = transform_num
        self.pre_process = pre_process
        self.targeted = targeted
        self.target = target
        self.num_query = 0
        self.adv_perturbation = adv_perturbation

    def perturb(self, x):
        img, mask_coords = self.adv_perturbation.image_transformation(self.img, x, self.mask_coords,self.pre_process)
        return img.to(utils.get_device()), mask_coords

    @abstractmethod
    def loss_function(self, perturbed_img, mask_coords):
        """
        Loss function for gradient-free optimizers

        Args:
            perturbed_img: The adversarial example to evaluate
            mask_coords: The coordinates of the mask. This allows to scale the bounding box for object detection tasks alongside the image.

        Returns:
            confidence: The lower the value, the more successful the attack.
            success: Whether the attack is successful.
            label: The model's predicted label
        """
        pass
