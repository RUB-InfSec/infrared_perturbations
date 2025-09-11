from abc import ABC, abstractmethod

from attacks.Attack import Attack
from perturbations.AdversarialPerturbation import AdversarialPerturbation


class Optimizer(ABC):
    def __init__(self, adv_perturbation: AdversarialPerturbation, attack: Attack, size: int = 10,
                 enforce_query_budget: bool = False, max_queries: int = 1000, verbose=False):
        self.adv_perturbation = adv_perturbation
        self.attack = attack
        self.size = size
        self.enforce_query_budget = enforce_query_budget
        self.max_queries = max_queries
        self.verbose = verbose

    @abstractmethod
    def optimize(self):
        """
        Run the optimizer to find the best solution

        Returns: Returns a tuple of four values:
            - The confidence value associated with the best solution
            - The best found solution
            - A boolean value indicating if the attack succeeded
            - the number of queries the optimizer used
            - the predicted label
        """
        pass
