import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

import utils.utils
from experiments.Experiment import Experiment

sys.path.append("../..")
import os
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import outset as otst
from outset import tweak as otst_tweak

from sklearn.metrics import roc_curve

import pickle

class e8(Experiment):

    def get_tasks(self):
        pass

    def evaluate(self):
        sam_checkpoint = "./model/segmentation/sam_vit_l_0b3195.pth"

        sam = sam_model_registry["vit_l"](checkpoint=sam_checkpoint)
        # SAM does not like MPS
        sam.to(device=utils.utils.get_device(ignore=("mps",)))

        scenarios = ["moving/all", "static/all", "static/indoor", "static/outdoor"]
        scenarios += [f"static/longitudinal/{i}" for i in range(4, 10)]
        scenarios += ["static/lateral/15", "static/lateral/20", "static/lateral/25"]

        longitudinal_averages = [0.0, 0.0]
        lateral_averages = [0.0, 0.0]

        for idx, label in enumerate(scenarios):
            label = label.replace("/", "_")

            ben, mal = self.filter_and_count(sam, path=os.path.join(self.path, 'images', label.replace("_", "/")), label=label)
            fpr, fn, tpr, tn, f1, nu = self.compute_rates(ben, mal)

            if "lon" in label:
                longitudinal_averages[0] += (tn / 6)
                longitudinal_averages[1] += (fn / 6)
            elif "lat" in label:
                lateral_averages[0] += (tn / 3)
                lateral_averages[1] += (fn / 3)

            if ("lon" not in label and "lat" not in label) or "9" in label or "25" in label:
                if "9" in label:
                    tn = longitudinal_averages[0]
                    fn = longitudinal_averages[1]
                    label = "static_longitudinal"
                elif "25" in label:
                    tn = lateral_averages[0]
                    fn = lateral_averages[1]
                    label = "static_lateral"

                print("-------------")
                print(label)
                print(f"CA: {tn:.04f}")
                print(f"ASR: {fn:.04f}")

            if idx in [0, 1]:
                self.plotter((fpr, tpr, nu), (ben, mal), label)

        return None, None

    def plot(self, results, queries):
        pass

    def plotter(self, arg1, arg2, label):

        fpr, tpr, nu = arg1
        benign_scores, malicious_scores = arg2

        plt.rcParams.update({'font.size': 28})
        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel("Number of masks")
        ax.set_ylabel("Ratio of data")
        plt.hist(benign_scores, label='benign', alpha=0.5, density=True)
        plt.hist(malicious_scores, label='adversarial', alpha=0.5, density=True)

        plt.tight_layout()
        plt.axvline(x=nu, color='red')
        plt.savefig(os.path.join(self.path, 'plots', f"segmentation_histogram_{label}.pdf"), bbox_inches='tight')

        if "moving" not in label:
            plt.rcParams.update({'font.size': 20})
            fig = plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr)
            # 3 axes grid: source plot and two zoom frames
            grid = otst.OutsetGrid([(0.0, 0.9, 0.1, 1.0)],
                                   x="FPR",
                                   y="TPR",
                                   marqueeplot_kws={
                                       "mark_glyph_kws": {"markersize": 25},  # Use Roman badge markers
                                       "leader_tweak": otst_tweak.TweakReflect(vertical=True)
                                   }, include_sourceplot=True)  # frame coords
            griddo = grid.broadcast(
                plt.plot,  # run plotter over all axes
                fpr,
                tpr,  # line coords
                c="black"
            )  # kwargs forwarded to plt.plot
        else:
            plt.rcParams.update({'font.size': 32})
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")

            ax.spines[['right', 'top']].set_visible(False)

            plt.plot(fpr, tpr, color='black')
            plt.tight_layout()
            plt.grid()

        plt.grid()
        if "moving" not in label:
            grid.marqueeplot()
            grid.tight_layout()
        plt.grid()
        plt.savefig(os.path.join(self.path, 'plots', f"roc_curve_static_{label}.pdf"), bbox_inches="tight")

    def run(self, args):
        self.args = args
        self.evaluate()

    def segment_folder(self, sam, path, label):
        if not os.path.isdir(path):
            raise FileNotFoundError

        sign_values = []

        file_path = os.path.join(self.path, 'results', f'masks_unfiltered_{label}.npy')

        if not os.path.isfile(file_path):
            for root, d_names, f_names in os.walk(path):

                if self.args.subset:
                    f_names = f_names[:5]

                for f in f_names:
                    filena = os.path.join(root, f)
                    image = cv2.imread(f"{filena}")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    mask_generator = SamAutomaticMaskGenerator(sam)
                    all_masks = mask_generator.generate(image)
                    sign_values.append([image.shape[0], image.shape[1], all_masks])

            with open(file_path, 'wb') as fp:
                pickle.dump(sign_values, fp)
        else:
            with open(file_path, 'rb') as fp:
                sign_values = pickle.load(fp)
        return sign_values

    def filter_masks(self, x, thres):
        # Create a circular mask

        height, width = x[0], x[1]
        min_pixels = int(thres * height * width)

        Y, X = np.ogrid[:height, :width]
        center = (width // 2, height // 2)
        radius = min(center[0], center[1])  # Fit circle in image
        circle_mask = (X - center[0]) ** 2 + (Y - center[1]) ** 2 <= radius ** 2
        circle_mask = circle_mask.astype(np.bool_)

        # Filter masks to keep only content inside circle
        filtered_masks = []
        for mask in x[2]:
            m = mask["segmentation"]
            m_inside = m & circle_mask
            if np.sum(m_inside) > min_pixels:  # Ignore very small segments
                new_mask = mask.copy()
                new_mask["segmentation"] = m_inside
                filtered_masks.append(new_mask)

        return filtered_masks

    def filter_and_count(self, sam, path, label):
        benign_label = f"benign_{label}"
        mal_label = f"malicious_{label}"
        size_threshold = 0.002

        benign_signs = self.segment_folder(sam, f"{path}/benign", benign_label)
        benign_scores = []
        for x in benign_signs:
            filtered_masks = self.filter_masks(x, size_threshold)
            benign_scores.append(len(filtered_masks))

        mal_signs = self.segment_folder(sam, f"{path}/malicious", mal_label)
        malicious_scores = []
        for idx, x in enumerate(mal_signs):
            filtered_masks = self.filter_masks(x, size_threshold)
            malicious_scores.append(len(filtered_masks))

        return benign_scores, malicious_scores

    def compute_rates(self, scores_benign, scores_malicious):
        true_labels = [[0] * len(scores_benign) + [1] * len(scores_malicious)]

        fpr, tpr, thresholds = roc_curve(true_labels[0], np.concatenate((scores_benign, scores_malicious)), pos_label=1)

        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
        EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        # print(f'EER: {EER} @ nu={eer_threshold}')

        predicted_labels_benign = (np.array(scores_benign) > eer_threshold).astype(int)
        predicted_labels_malicious = (np.array(scores_malicious) > eer_threshold).astype(int)

        cm = confusion_matrix(true_labels[0], np.concatenate((predicted_labels_benign, predicted_labels_malicious)),
                              normalize="true")

        tn, fp, fn, tp = cm.ravel()
        length = len(np.concatenate((predicted_labels_benign, predicted_labels_malicious)))

        f1 = f1_score(true_labels[0], np.concatenate((predicted_labels_benign, predicted_labels_malicious)))

        return fpr, fn, tpr, tn, f1, eer_threshold


if __name__ == "__main__":
    gs = e8()
    gs.main()

