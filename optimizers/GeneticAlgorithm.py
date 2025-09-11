import warnings

import numpy as np

from attacks.Attack import Attack
from optimizers.Optimizer import Optimizer
from perturbations.AdversarialPerturbation import AdversarialPerturbation


class GeneticAlgorithm(Optimizer):
    def __init__(self, adv_perturbation: AdversarialPerturbation, attack: Attack, size: int = 100,
                 enforce_query_budget: bool = False, max_queries: int = 1000, verbose=False) -> None:
        super().__init__(adv_perturbation, attack, size, enforce_query_budget, max_queries, verbose)
        self.population = [self.adv_perturbation.initial_solution() for _ in range(self.size)]
        self.f_vals, self.successes, self.predicted_labels = self.evaluate_population(self.population)

        # Compute the maximum number of iterations such that we do not exceed the given query budgets. This formula
        # takes into account that the first 50 % of the iterations are performed with EOT while the last 50 % are not.
        self.iter_num = 2 * self.max_queries // (self.size * (self.attack.transform_num + 2))

        # Define the coefficients of an exponentially degrading number of patches to draw freshly in each iteration.
        # a and b define a function f(iteration) = a * e^(-b*iteration) The way a and b are computed guarantees,
        # that in the first iteration mutations can jump up to 0.5 * width and in the last mutation only 1 px wide.
        self.width = self.adv_perturbation.x_max - self.adv_perturbation.x_min
        self.a = self.width / 5
        self.b = - np.log(5 / self.width) / self.iter_num

    def success(self):
        return any(self.successes)

    def evaluate_population(self, population):
        vals = []
        for ind in population:
            img, mask_coords = self.attack.perturb(ind)
            vals.append(self.attack.loss_function(img, mask_coords))
        return [val[0] for val in vals], [val[1] for val in vals], [val[2] for val in vals]

    def population_order(self, population_f_vals):
        return np.argsort(population_f_vals)

    def rank_based_selection(self, population_f_vals, k):
        # Truncated linear rank-based selection
        ranks = np.array([max([0.00001, len(population_f_vals) - 2 * i]) for i in range(len(population_f_vals))])
        return np.random.choice(self.population_order(population_f_vals), k, replace=False, p=ranks / sum(ranks))

    def recombine(self, parents):
        # Uniform Crossover of the patches
        child = np.zeros(self.adv_perturbation.patch_count * self.adv_perturbation.patch_dimensionality)
        for i in range(self.adv_perturbation.patch_count):
            parent_idx = np.random.randint(len(parents))
            child[i * self.adv_perturbation.patch_dimensionality:(i + 1) * self.adv_perturbation.patch_dimensionality] = \
                parents[parent_idx][
                i * self.adv_perturbation.patch_dimensionality:(i + 1) * self.adv_perturbation.patch_dimensionality]
        child, error = self.adv_perturbation.repair(child)
        if error:
            warnings.warn(
                "The recombination of the parents yielded a child that could not be repaired into a valid solution; sampled a new one instead.")
            return self.adv_perturbation.initial_solution()
        return child

    def mutate(self, ind, k):
        mutated_ind = ind + np.random.uniform(- k, k, ind.shape)
        mutated_ind, error = self.adv_perturbation.repair(mutated_ind)
        if error:
            warnings.warn(
                "The mutation of the individuum yielded a result that could not be repaired into a valid solution; sampled a new one instead.")
            return self.adv_perturbation.initial_solution()
        return mutated_ind

    def optimize(self):
        for iteration in range(self.iter_num):
            if iteration == self.iter_num // 2:
                # The last 50 % of the iterations are performed without EOT
                self.attack.transform_num = 0

            if self.verbose:
                print(
                    f'Generation {iteration} has avg fitness {sum(self.f_vals) / self.size} and min fitness {min(self.f_vals)}')
            offspring = []

            # Mating selection and Recombination
            for _ in range(self.size):
                parent_idxs = self.rank_based_selection(self.f_vals, 2)
                offspring.append(self.recombine([self.population[idx] for idx in parent_idxs]))

            # Mutation
            k = int(self.a * np.exp(- self.b * iteration))
            offspring = [self.mutate(ind, k) for ind in offspring]

            # Evaluation
            offspring_f_vals, offspring_successes, offspring_predicted_labels = self.evaluate_population(offspring)

            # Perform "+"-selection
            self.population += offspring
            self.f_vals += offspring_f_vals
            self.successes += offspring_successes
            self.predicted_labels += offspring_predicted_labels

            # Environmental Selection ("+"-selection)
            offspring_idxs = np.argsort(np.array(self.f_vals))[:self.size]
            self.population = [self.population[idx] for idx in offspring_idxs]
            self.f_vals = [self.f_vals[idx] for idx in offspring_idxs]
            self.successes = [self.successes[idx] for idx in offspring_idxs]
            self.predicted_labels = [self.predicted_labels[idx] for idx in offspring_idxs]

            if self.success() and not self.enforce_query_budget:
                # We stop searching if it's not a physical attack
                break

        order = self.population_order(self.f_vals)
        return self.f_vals[order[0]], self.population[order[0]], self.successes[order[0]], self.attack.num_query, \
            self.predicted_labels[order[0]]
