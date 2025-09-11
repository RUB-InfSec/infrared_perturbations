import os
import numpy as np

from experiments.Experiment import Experiment

class e3_classification(Experiment):

    def __init__(self):
        params = {
            "scenario": ["brake", "accelerate", "stop", "untargeted"],
            "run": [i for i in range(max(len(self.target_mapping("GTSRB", "stop")), len(self.target_mapping("LISA", "stop")))+1)],
            "dataset": ["GTSRB", "LISA"],
            "lux": [10, 1000, 2000, 3000, 4000, 5000],
            "patch_count": [16, 32, 64, 96, 128, 192],
            }

        axes = ("lux", "patch_count")
        labels = ("Lux", "Patches")

        average_axis = (1,)

        Experiment.__init__(self, params, axes, labels, average_axis=average_axis, tasks_per_worker=6)

    def target_mapping(self, dataset, target):
        if target == "untargeted":
            return '[("*","*")]'
        if dataset == "GTSRB":
            if target == "stop":
                return [f'"[(14, {idx})]"' for idx in [0, 1, 2, 3, 4, 5, 7, 8]]  # Flip stop sign to any other sign
            elif target == "accelerate":
                return '"[(0,8),(1,8),(2,8),(3,8),(4,8),(5,8),(7,8)]"'  # Flip 20,30, 50, 60, 70 80, 100, speed limits to 120 speed limit
            elif target == "brake":
                return '"[(1,0),(2,0),(3,0),(4,0),(5,0),(7,0),(8,0)]"'  # Flip 30, 50, 60, 70 80, 100, 120 speed limits to 20 speed limit
            return None
        elif dataset == "LISA":
            if target == "stop":
                return [f'"[(12, {idx})]"' for idx in [8, 9, 10, 11]]  # Flip stop sign to any other sign
            elif target == "accelerate":
                return '"[(8,11),(9,11),(10,11)]"'  # Flip 25, 30, 35 speed limits to 45 speed limit
            elif target == "brake":
                return '"[(9,8),(10,8),(11,8)]"'  # Flip 25, 30, 35, 45 speed limits to 25 speed limit
            return None
        return None

    def get_tasks(self, args):
        if args.subset:
            patch_counts = [self.params['patch_count'][-1]]
            lux_values = [self.params['lux'][0]]
        else:
            patch_counts = self.params['patch_count']
            lux_values = self.params['lux']

        tasks = []
        for idx, scenario in enumerate(self.params['scenario']):
            for dataset in self.params['dataset']:
                mapping = self.target_mapping(dataset, scenario)
                for jdx, run in enumerate(self.params['run']):
                    if isinstance(mapping, str):
                        mapping = [mapping]
                        assert scenario != "stop"
                        # for all scenarios that just contained of a single-string only one experiment is needed
                    elif len(mapping) == 1 and run > 0:
                        break
                    elif run > len(self.target_mapping(dataset, 'stop')) - 1:
                        break

                    for lux in lux_values:
                        for patch_count in patch_counts:
                            tasks.append(
                                ['python3', 'infrared_perturbation.py',
                                 '--dataset', dataset,
                                 '--target_model', 'GtsrbCNN' if dataset == 'GTSRB' else 'LisaCNN',
                                 '--patch_count', str(patch_count),
                                 '--target_mapping', mapping[jdx],
                                 '--optimizer', "lrs",
                                 '--max_queries', "1000",
                                 '--lux', str(lux),
                                 '--mesh_count', "16",
                                 '--instances_per_class', "2" if args.subset else "150" if idx != 3 else "25",
                                 '--save_dir', os.path.join(self.path, 'results', scenario, str(run), dataset, str(lux), str(patch_count))
                                 ]
                            )

        return tasks


if __name__ == "__main__":
    gs = e3_classification()
    gs.main()