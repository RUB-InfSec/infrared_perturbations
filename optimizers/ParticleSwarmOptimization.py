# -*- coding: utf-8 -*-
import copy

import numpy as np

from optimizers.Optimizer import Optimizer
from perturbations.AdversarialPerturbation import AdversarialPerturbation


class Particle:
    def __init__(self, adv_perturbation, max_speed: float = 1.5) -> None:
        self.adv_perturbation = adv_perturbation
        self.position = self.adv_perturbation.initial_solution()
        self.speed = np.random.uniform(-max_speed, max_speed, self.position.shape)
        self.best_pos = np.zeros(self.position.shape)
        self.predicted_label = -1
        self.confidence = float('inf')

class ParticleSwarmOptimization(Optimizer):
    def __init__(self, adv_perturbation: AdversarialPerturbation, attack, size: int = 10, max_speed: float = 1.5,
                 enforce_query_budget: bool = False, max_queries: int = 1000, verbose=False) -> None:
        super().__init__(adv_perturbation, attack, size, enforce_query_budget, max_queries, verbose)
        self.max_speed = max_speed
        self.w = 1
        self.c1 = self.c2 = 2
        self.best_confidence = float('inf')
        self.predicted_label = -1
        self.succeed = False

        self.particle_list = [Particle(adv_perturbation, max_speed) for _ in range(size)]

        self.best_position = np.zeros(self.particle_list[0].position.shape)

    def update_speed(self, part):
        speed_value = self.w * part.speed \
                      + self.c1 * np.random.uniform(part.position.shape) * (part.best_pos - part.position) \
                      + self.c2 * np.random.uniform(part.position.shape) * (self.best_position - part.position)
        part.speed = speed_value.clip(-self.max_speed, self.max_speed)

    def update_position(self, part):
        part.position, error = self.adv_perturbation.repair(part.position + part.speed)
        while error:
            part.position, error = self.adv_perturbation.repair(part.position + part.speed)
            part.speed = np.random.uniform(-self.max_speed, self.max_speed, part.position.shape)

        img, mask_coords = self.attack.perturb(part.position)
        confidence, succeed, predicted_label = self.attack.loss_function(img, mask_coords)
        self.succeed |= succeed
        if confidence < part.confidence:
            part.confidence = confidence
            part.best_pos = copy.deepcopy(part.position)
            part.predicted_label = predicted_label
        if confidence < self.best_confidence or (succeed and not self.enforce_query_budget):
            self.best_confidence = confidence
            self.best_position = copy.deepcopy(part.position)
            self.predicted_label = predicted_label

    def optimize(self):
        # Compute the maximum number of iterations such that we do not exceed the given query budgets. This formula
        # takes into account that the first 50 % of the iterations are performed with EOT while the last 50 % are not.
        iter_num = 2 * self.max_queries // (self.size * (self.attack.transform_num + 2))

        for i in range(iter_num):
            if i == iter_num // 2:
                # The last 50 % of the iterations are performed without EOT
                self.attack.transform_num = 0
            for part in self.particle_list:
                if self.succeed and not self.enforce_query_budget:
                    # In the physical case, improve further even when already succeeded
                    break
                self.update_speed(part)
                self.update_position(part)
            if self.verbose:
                print(f"iteration: {i + 1} {self.best_confidence}")

        return self.best_confidence, self.best_position, self.succeed, self.attack.num_query, self.predicted_label
