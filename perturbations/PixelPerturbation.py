import numpy as np

from perturbations.AdversarialPerturbation import AdversarialPerturbation, contains


class PixelPerturbation(AdversarialPerturbation):
    def __init__(self, patch_count, blur_coefficient, x_min, x_max, y_min, y_max, mesh_count, image_ir, od, args):
        super().__init__(patch_count, 2, blur_coefficient, x_min, x_max, y_min, y_max, image_ir, od, args)
        self.mesh_count = mesh_count
        self.pixel_size = ((x_max - x_min) / mesh_count, (y_max - y_min) / mesh_count)

    def fill_mask_array(self, mask, positions):
        position_array = np.zeros_like(mask[0])
        for patch_pos in np.reshape(positions, (self.patch_count, 2)):
            pixel_coords = np.array([
                [patch_pos[0], patch_pos[1]],
                [patch_pos[0] + self.pixel_size[0], patch_pos[1]],
                [patch_pos[0] + self.pixel_size[0], patch_pos[1] + self.pixel_size[1]],
                [patch_pos[0], patch_pos[1] + self.pixel_size[1]]
            ])
            position_array += contains(pixel_coords, np.transpose(mask, [1, 0]))
        return position_array

    def initial_solution(self):
        # The repair function never returns errors, so we don't have to hassle with it
        coords = np.empty((self.patch_count * 2,))
        coords[0::2] = np.random.uniform(self.x_min, self.x_max, self.patch_count)
        coords[1::2] = np.random.uniform(self.y_min, self.y_max, self.patch_count)
        solution, _ = self.repair(coords)
        return solution

    def repair(self, x):
        # Clip at image / mask boundaries and then align with pixel grid
        x[0::2] = np.floor(np.clip(x[0::2], self.x_min, self.x_max) / self.pixel_size[0]) * self.pixel_size[0]
        x[1::2] = np.floor(np.clip(x[1::2], self.y_min, self.y_max) / self.pixel_size[1]) * self.pixel_size[1]
        return x, False
