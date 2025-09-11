from attacks.Attack import Attack
from optimizers.Optimizer import Optimizer
from perturbations.AdversarialPerturbation import AdversarialPerturbation


class RandomBaseline(Optimizer):
    def __init__(self, adv_perturbation: AdversarialPerturbation, attack: Attack, enforce_query_budget: bool = False,
                 max_queries: int = 1000, verbose=False) -> None:
        super().__init__(adv_perturbation, attack, 1, enforce_query_budget, max_queries, verbose)

    def optimize(self):
        x = self.adv_perturbation.initial_solution()
        img, mask_coords = self.attack.perturb(x)
        fx, succeed, predicted_label = self.attack.loss_function(img, mask_coords)

        if self.verbose:
            print(f"iteration: {1} {fx}")

        return fx, x, succeed, self.attack.num_query, predicted_label
