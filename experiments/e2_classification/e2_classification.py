import os

from experiments.Experiment import Experiment

class e2_classification(Experiment):

    def __init__(self):

        params = {"optimizer": ["lrs", "ga", "es", "pso", "rnd"],
                  "patch_count": [64, 128, 256, 384, 512, 768]}

        axes = ("optimizer", "patch_count")
        labels = ("Optimizer", "Perturbation Count")

        Experiment.__init__(self, params, axes, labels, transpose_plot=True, tasks_per_worker=6)

    def get_tasks(self, args):
        if args.subset:
            patch_counts = [self.params['patch_count'][-1]]
        else:
            patch_counts = self.params['patch_count']

        tasks = []
        for optimizer in self.params['optimizer']:
            for patch_count in patch_counts:
                tasks.append(
                    ['python3', 'infrared_perturbation.py',
                     '--dataset', 'GTSRB',
                     '--target_model', 'GtsrbCNN',
                     '--patch_count', str(patch_count),
                     '--target_mapping', '[("*","*")]',
                     '--optimizer', optimizer,
                     '--max_queries', "1000",
                     '--lux', "10",
                     '--mesh_count', "32",
                     '--instances_per_class', "5" if args.subset else "25",
                     '--save_dir', os.path.join(self.path, 'results', optimizer, str(patch_count))]
                    )
        return tasks

if __name__ == "__main__":
    gs = e2_classification()
    gs.main()