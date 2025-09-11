import os

from experiments.Experiment import Experiment

class e5(Experiment):

    def __init__(self):

        self.target_mappings = [[f'"[(12, {idx})]"' for idx in [1, 8, 13, 25]], [f'"[(13, {idx})]"' for idx in [1, 8, 12, 25]]]

        params = {
            "run": ["priority_road", "yield"],
            "map": [i for i in range(len(self.target_mappings[0]) + 1)],
            "lux": [10, 1000, 2000, 3000, 4000, 5000],
            "patch_count": [16, 32, 64, 96, 128, 192],
            }

        axes = ("lux", "patch_count")
        labels = ("Lux", "Patches")

        Experiment.__init__(self, params, axes, labels, average_axis=(1,), tasks_per_worker=12)


    def get_tasks(self, args):
        if args.subset:
            patch_counts = [self.params['patch_count'][-1]]
        else:
            patch_counts = self.params['patch_count']

        tasks = []

        for kdx, run in enumerate(self.params['run']):
            for jdx, map in enumerate(self.params['map']):
                if map > len(self.target_mappings[kdx]) - 1:
                    break

                for lux in self.params['lux']:
                    for patch_count in patch_counts:
                        tasks.append(
                            ['python3', 'infrared_perturbation.py',
                             '--dataset', "GTSRB",
                             '--target_model', 'GtsrbCNN',
                             '--patch_count', str(patch_count),
                             '--target_mapping', self.target_mappings[kdx][jdx],
                             '--optimizer', "lrs",
                             '--max_queries', "1000",
                             '--lux', str(lux),
                             '--mesh_count', "16",
                             '--instances_per_class', "25",
                             '--save_dir',
                             os.path.join(self.path, 'results', run, str(map), str(lux),
                                          str(patch_count))
                             ]
                        )

        return tasks


if __name__ == "__main__":
    gs = e5()
    gs.main()