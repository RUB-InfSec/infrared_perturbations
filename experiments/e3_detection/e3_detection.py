import os

from experiments.Experiment import Experiment

class e3_detection(Experiment):

    def __init__(self):
        params = {
            "dataset": ["GTSDB", "Mapillary"],
            "lux": [10, 1000, 2000, 3000, 4000, 5000],
            "patch_count": [16, 32, 64, 96, 128, 192],
            }

        axes = ("lux", "patch_count")
        labels = ("Lux", "Patches")

        Experiment.__init__(self, params, axes, labels, tasks_per_worker=6)

    def get_tasks(self, args):
        if args.subset:
            patch_counts = [self.params['patch_count'][-1]]
            lux_values = [self.params['lux'][0]]
        else:
            patch_counts = self.params['patch_count']
            lux_values = self.params['lux']

        tasks = []
        for dataset in self.params['dataset']:
            for lux in lux_values:
                for patch_count in patch_counts:
                    tasks.append(
                        ['python3', 'infrared_perturbation_od.py',
                         '--dataset', dataset,
                         '--patch_count', str(patch_count),
                         '--target_mapping', '[("*","hide")]',
                         '--optimizer', "lrs",
                         '--max_queries', "1000",
                         '--lux', str(lux),
                         '--mesh_count', "16",
                         '--instances', "10" if args.subset else "0",
                         '--save_dir',
                         os.path.join(self.path, 'results', dataset, str(lux), str(patch_count))
                         ]
                    )

        return tasks

if __name__ == "__main__":
    gs = e3_detection()
    gs.main()