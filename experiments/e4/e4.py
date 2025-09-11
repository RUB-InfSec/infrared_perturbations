import os

from experiments.Experiment import Experiment

class e4(Experiment):

    def __init__(self):
        params = {
            "mesh_count": [1, 2, 4],
            "patch_count": [64, 128, 256, 384, 512, 768],
            }

        axes = ("mesh_count", "patch_count")
        labels = ("Perturbation Width [l]", "Total Perturbation Area [Pixels]")

        Experiment.__init__(self, params, axes, labels, transpose_plot=True, tasks_per_worker=6)


    def get_tasks(self, args):
        if args.subset:
            patch_counts = [self.params['patch_count'][-1]]
        else:
            patch_counts = self.params['patch_count']

        tasks = []
        for mesh_count in self.params['mesh_count']:
            for patch_count in patch_counts:
                tasks.append(
                    ['python3', 'infrared_perturbation.py',
                     '--dataset', "GTSRB",
                     '--target_model', 'GtsrbCNN',
                     '--patch_count', str(patch_count // (mesh_count ** 2)),
                     '--target_mapping', '"[(1,0),(2,0),(3,0),(4,0),(5,0),(7,0),(8,0)]"',
                     '--optimizer', "lrs",
                     '--max_queries', "1000",
                     '--lux', "10",
                     '--mesh_count', str(32 // mesh_count),
                     '--instances_per_class', "5" if args.subset else "25",
                     '--save_dir', os.path.join(self.path, 'results', str(mesh_count), str(patch_count))
                     ]
                )

        return tasks


if __name__ == "__main__":
    gs = e4()
    gs.main()