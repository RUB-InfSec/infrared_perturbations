import cma as cma
import numpy

from attacks.Attack import Attack
from optimizers.Optimizer import Optimizer
from perturbations.AdversarialPerturbation import AdversarialPerturbation


class EvolutionStrategy(Optimizer):
    def __init__(self, adv_perturbation: AdversarialPerturbation, attack: Attack, enforce_query_budget: bool = False,
                 max_queries: int = 1000, verbose=False) -> None:
        self.es = cma.CMAEvolutionStrategy(adv_perturbation.initial_solution,
                                           (adv_perturbation.x_max - adv_perturbation.x_min) / 4,
                                           {'CMA_elitist': True})
        super().__init__(adv_perturbation, attack, self.es.popsize, enforce_query_budget, max_queries, verbose)

        # Compute the maximum number of iterations such that we do not exceed the given query budgets. This formula
        # takes into account that the first 50 % of the iterations are performed with EOT while the last 50 % are not.
        self.iter_num = 2 * self.max_queries // (self.size * (self.attack.transform_num + 2))

    def objective(self, x):
        x, error = self.adv_perturbation.repair(x)
        if error:
            return numpy.NaN, False, None, None  # This causes ES to resample that solution!

        img, mask_coords = self.attack.perturb(x)
        return self.attack.loss_function(img, mask_coords)

    def optimize(self):
        terminate = False

        for iteration in range(self.iter_num):
            if iteration == self.iter_num // 2:
                # The last 50 % of the iterations are performed without EOT
                self.attack.transform_num = 0

            solutions = self.es.ask()
            function_vals = [self.objective(s) for s in solutions]
            if any([bool(x[1]) for x in function_vals]):  # There exist a succeeding solution
                terminate = True

            if self.verbose:
                self.es.disp()

            self.es.tell(solutions, [x[0] for x in function_vals])

            if self.es.stop() or (terminate and not self.enforce_query_budget):
                # We stop searching if a solution succeeded, and it's not a physical attack. Independent of whether
                # it's a physical or digital attack we stop searching when the algorithm does not make any progress
                # anymore.
                break

        best_x, best_x_fval = self.es.result[0], self.es.result[1]
        _, succeed, predicted_label = self.objective(best_x)
        return best_x_fval, best_x, succeed, self.attack.num_query, predicted_label
