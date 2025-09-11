import argparse
import itertools
import os
import sys
from abc import ABC, abstractmethod
from collections import Counter
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.getcwd())
from experiments.Scheduler import Worker, TaskManager

from itertools import product

class Experiment(ABC):

    def __init__(self, params=None, axes=None, labels=None, transpose_plot=False, average_axis=None, tasks_per_worker=6):
        self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.__class__.__name__)
        self.gpus = [0]
        self.tasks_per_worker = tasks_per_worker
        self.params = params
        self.axes = axes
        self.labels = labels
        self.transpose = transpose_plot
        self.average_axis = average_axis

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpus))

        import torch
        torch.cuda.device_count.cache_clear()

        os.makedirs(os.path.join(self.path, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(self.path, 'results'), exist_ok=True)

    @abstractmethod
    def get_tasks(self, args):
        pass

    def main(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--mode", type=str, choices=["run", "evaluate"])
        parser.add_argument("--subset", action="store_true", help='Only runs experiments for a representative subset.')
        parser.add_argument("--force", action="store_true", help='Overwrites existing results.')
        args = parser.parse_args()

        if args.subset:
            print("Running with reduced number of samples and subset of parameters.")

        if args.mode == "run":
            self.run(args)
        elif args.mode == "evaluate":
            results, queries = self.evaluate()
            self.plot(results, queries)

    def run(self, args):
        log_path = os.path.join(self.path, 'logs')
        os.makedirs(log_path, exist_ok=True)

        tasks = self.get_tasks(args)
        if tasks is not None:

            dir_path = os.path.join(self.path, 'results')
            if bool(os.listdir(dir_path)):
                if args.force:
                    print("Result files found. Overwriting...")
                else:
                    print("Result files found. Use --force to overwrite or delete results folder.")
                    exit()

            tm = TaskManager(tasks)

            gpu_cycle = cycle(self.gpus)
            workers = [Worker(i, next(gpu_cycle), tm, log_path) for i in range(self.tasks_per_worker * len(self.gpus))]

            for w in workers:
                w.start()
            for w in workers:
                w.join()

    def evaluate(self):

        results = np.zeros(tuple(len(self.params[key]) for key in self.params))
        queries = np.zeros_like(results)

        keys = list(self.params)
        values = [self.params[k] for k in self.params]

        for combination in product(*values):
            labeled = dict(zip(keys, combination))

            # Get the index of each value in its corresponding parameter list
            index = tuple(self.params[k].index(labeled[k]) for k in keys)

            path = os.path.join(self.path, 'results', *[str(labeled[k]) for k in keys])

            if os.path.exists(path) and os.path.isdir(path):
                files = os.listdir(path)
                if not files:  # avoid division by zero
                    continue

                results[index] = Counter(map(lambda x: x[:-4].split('_')[-1], os.listdir(path)))['True'] / len(
                    os.listdir(path))
                queries[index] = sum(list(map(lambda x: int(x[:-4].split('_')[-2]), os.listdir(path)))) / len(
                    os.listdir(path))

        if self.average_axis is not None:
            for ax in self.average_axis:
                results = self.replace_axis_with_nonzero_average(results, axis=ax)
                queries = self.replace_axis_with_nonzero_average(queries, axis=ax)

        return results, queries


    def replace_axis_with_nonzero_average(self, arr, axis):
        # Sum non-zero values along the axis
        sum_nonzero = np.sum(np.where(arr != 0, arr, 0), axis=axis, keepdims=True)

        # Count non-zero elements along the axis
        count_nonzero = np.count_nonzero(arr, axis=axis, keepdims=True)

        # Compute average safely, with shape preserved (axis size = 1)
        avg = np.divide(
            sum_nonzero,
            count_nonzero,
            out=np.zeros_like(sum_nonzero, dtype=float),
            where=count_nonzero != 0
        )

        return avg


    def plot(self, results, queries):
        param_names = list(self.params.keys())  # Ordered list of parameter names
        name_to_axis = {name: i for i, name in enumerate(param_names)}

        # Convert param names in self.slice_axes to axis indices
        slice_axes = [name_to_axis[name] for name in self.axes]
        iter_axes = [i for i in range(results.ndim) if i not in slice_axes]

        # Axis labels and tick values for slice axes
        x_axis, y_axis = slice_axes
        x_labels = self.params[param_names[x_axis]]
        y_labels = self.params[param_names[y_axis]]
        x_label = self.labels[0]
        y_label = self.labels[1]

        # Build all index combinations for non-slice axes
        iter_ranges = [
            range(min(len(self.params[param_names[ax]]), results.shape[ax]))
            for ax in iter_axes
        ]
        for idx_combo in itertools.product(*iter_ranges):
            # Build slicing tuple: keep slice_axes as ':' and fix others
            slicer = [slice(None)] * results.ndim
            for i, ax in enumerate(iter_axes):
                slicer[ax] = idx_combo[i]

            slice_result = results[tuple(slicer)].T  # Transpose for display
            slice_query = queries[tuple(slicer)]

            fig, axs = plt.subplots(1, 1, figsize=(7, 7))
            if not self.transpose:
                axs.imshow(slice_result, vmin=np.min(slice_result), vmax=1)
                axs.set_xticks(np.arange(len(x_labels)), labels=x_labels)
                axs.set_xlabel(x_label)

                axs.set_yticks(np.arange(len(y_labels)), labels=y_labels)
                axs.set_ylabel(y_label)
            else:
                axs.imshow(slice_result.T, vmin=np.min(slice_result.T), vmax=1)
                axs.set_xticks(np.arange(len(y_labels)), labels=y_labels)
                axs.set_xlabel(y_label)

                axs.set_yticks(np.arange(len(x_labels)), labels=x_labels)
                axs.set_ylabel(x_label)


            empty_run = True

            for x in range(len(x_labels)):
                for y in range(len(y_labels)):
                    # Rebuild the full index to access data points
                    full_idx = []
                    for i in range(results.ndim):
                        if i == x_axis:
                            full_idx.append(x)
                        elif i == y_axis:
                            full_idx.append(y)
                        else:
                            full_idx.append(idx_combo[iter_axes.index(i)])

                    if not np.all(results[tuple(full_idx)] == 0) and not np.all(queries[tuple(full_idx)] == 0):
                        empty_run = False

                    if not self.transpose:
                        axs.text(x, y,
                                 f'{100 * results[tuple(full_idx)]:.1f}\n{queries[tuple(full_idx)]:.1f}',
                                 ha="center", va="center", color="w", fontsize=16)
                    else:
                        axs.text(y, x,
                                 f'{100 * results[tuple(full_idx)]:.1f}\n{queries[tuple(full_idx)]:.1f}',
                                 ha="center", va="center", color="w", fontsize=16)

            fig.tight_layout()

            # Create filename with all iterated axis values
            filename_parts = [
                f"{param_names[ax]}_{self.params[param_names[ax]][idx]}"
                for ax, idx in zip(iter_axes, idx_combo)
            ]
            filename_suffix = "_".join(filename_parts)
            filename = f"results_{self.__class__.__name__}_{filename_suffix}.pdf"

            os.makedirs(os.path.join(self.path, 'plots'), exist_ok=True)
            if not empty_run:
                plt.savefig(os.path.join(self.path, 'plots', filename), bbox_inches='tight')
            plt.close(fig)