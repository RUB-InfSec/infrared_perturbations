import os

import infrared_perturbation_od
import utils.utils
from experiments.Experiment import Experiment


class e6_detection(Experiment):

    def __init__(self):
        params = {
            "dataset": ["Mapillary", "GTSDB"],
        }

        Experiment.__init__(self, params, None, None, tasks_per_worker=6)

    def get_tasks(self, args):
        tasks = []
        for dataset in self.params['dataset']:
            tasks.append(
                ['python3', 'infrared_perturbation_od.py',
                 '--dataset', dataset,
                 '--patch_count', "192",
                 '--target_mapping', '[("*","hide")]',
                 '--optimizer', "lrs",
                 '--max_queries', "1000",
                 '--lux', "10",
                 '--instances', "10" if args.subset else "0",
                 '--mesh_count', "16",
                 '--save_dir', os.path.join(self.path, 'results', 'experiments', dataset)
                 ]
            )

        return tasks

    def evaluate(self):
        parser = utils.utils.get_parser_detection()

        for source_model in self.params['dataset']:
            for target_model in self.params['dataset']:
                if source_model == target_model:
                    continue
                args = parser.parse_args(
                    ['--dataset', source_model, '--save_dir', os.path.join(self.path, 'results', 'transferability', source_model), '--store_predictions',
                     'transferability', '--adv_dir',
                     str(os.path.join(self.path, 'results', 'experiments', target_model)), '--src_dataset', target_model])

                infrared_perturbation_od.evaluate_transferability(args)

        return None, None

    def plot(self, results, queries):
        pass


if __name__ == "__main__":
    gs = e6_detection()
    gs.main()
