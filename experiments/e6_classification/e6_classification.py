import os

from dataset.classification import gtsrb
from experiments.Experiment import Experiment


class e6_classification(Experiment):

    def __init__(self):
        params = {
            "target_model": ["GtsrbCNN", "ResNet50", "ConvNeXt", "SwinTransformer"],
        }

        Experiment.__init__(self, params, None, None, tasks_per_worker=6)

    def get_tasks(self, args):
        tasks = []
        for target_model in self.params['target_model']:
            tasks.append(
                ['python3', 'infrared_perturbation.py',
                 '--dataset', "GTSRB",
                 '--target_model', target_model,
                 '--patch_count', "192",
                 '--target_mapping', '[("*","*")]',
                 '--optimizer', "lrs",
                 '--max_queries', "1000",
                 '--lux', "10",
                 '--enforce_query_budget',
                 '--mesh_count', "16",
                 '--instances_per_class', "5" if args.subset else "25",
                 '--save_dir', os.path.join(self.path, 'results', target_model)
                 ]
            )

        return tasks

    def evaluate(self):
        parser = gtsrb.get_parser()

        print("Transferability")
        for source_model in self.params['target_model']:
            for target_model in self.params['target_model']:
                if source_model == target_model:
                    continue
                args = parser.parse_args(['--model', source_model, 'test', '--dir', str(os.path.join(self.path, 'results', target_model))])
                print(f"{source_model} -> {target_model}: {(1 - gtsrb.test_model(args)):.02f}")

        return None, None

    def plot(self, results, queries):
        pass

if __name__ == "__main__":
    gs = e6_classification()
    gs.main()
