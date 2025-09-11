import copy
from math import ceil

import numpy as np

from attacks.Attack import Attack
from optimizers.Optimizer import Optimizer
from perturbations.AdversarialPerturbation import AdversarialPerturbation


class LocalRandomSearch(Optimizer):
    def __init__(self, adv_perturbation: AdversarialPerturbation, attack: Attack, enforce_query_budget: bool = False,
                 max_queries: int = 1000, verbose=False) -> None:
        super().__init__(adv_perturbation, attack, 1, enforce_query_budget, max_queries, verbose)

    def optimize(self):
        x = self.adv_perturbation.initial_solution()
        img, mask_coords = self.attack.perturb(x)
        fx, succeed, predicted_label = self.attack.loss_function(img, mask_coords)

        # Compute the maximum number of iterations such that we do not exceed the given query budgets. This formula
        # takes into account that the first 50 % of the iterations are performed with EOT while the last 50 % are not.
        iter_num = self.max_queries // (self.size * (self.attack.transform_num + 1))

        for iteration in range(iter_num):
            x_new = copy.deepcopy(x)

            # Compute the number of patches that are to be replaced in the current iteration. The way this function is
            # designed ensures, that there are patch_count / 2 patches replaced in the first iteration and 2 patches in
            # the last iteration. The space in between is modelled as exponentially decreasing.
            k = ceil(self.adv_perturbation.patch_count / 2 * np.exp(
                np.log(2 / self.adv_perturbation.patch_count) / iter_num * iteration))

            random_patch_idxs = np.random.choice(self.adv_perturbation.patch_count, size=k)

            random_point = self.adv_perturbation.initial_solution()
            for random_patch_idx in random_patch_idxs:
                for idx in range(random_patch_idx * self.adv_perturbation.patch_dimensionality,
                                 (random_patch_idx + 1) * self.adv_perturbation.patch_dimensionality):
                    x_new[idx] = random_point[idx]

            img, mask_coords = self.attack.perturb(x_new)

            fx_new, succeed_new, predicted_label_new = self.attack.loss_function(img, mask_coords)

            if fx_new < fx:
                if self.verbose:
                    print(f"Updated best solution at it. {iteration + 1}")
                x = x_new
                succeed = succeed_new
                predicted_label = predicted_label_new
                fx = fx_new

            if succeed and not self.enforce_query_budget:  # Terminate early if the attack succeeded and it's not physical
                break

            if self.verbose:
                print(f"iteration: {iteration + 1} {fx_new} {succeed_new} {predicted_label_new}")

        return fx, x, succeed, self.attack.num_query, predicted_label
