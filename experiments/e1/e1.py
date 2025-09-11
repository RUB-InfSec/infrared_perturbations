import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

from experiments.Experiment import Experiment


class e1(Experiment):

    def get_tasks(self, args):
        return None

    def run(self, args):
        print("Please use the argument --mode evaluate. This experiment only produces a result plot and hence no results need to be generated in the first place.")

    def plot(self, results, queries):
        pass

    def evaluate(self):
        trans_files = sorted(os.listdir(f"{self.path}/images/VIS"))
        brightness = []
        image_pairs = []
        points = [(45, 115),
                  (45, 115),
                  (44, 115),
                  (44, 115),
                  (43, 115),
                  (43, 115),
                  (37, 115)]

        for idx, file in enumerate(trans_files):
            if ".png" in file:
                bright = int(file.replace("lux.png", ""))
            else:
                bright = int(file.replace("lux.jpg", ""))
            brightness.append(bright)

            # resize is needed
            vis_img = cv2.resize(cv2.imread(f"{self.path}/images/VIS/{file}"), (224, 224))
            ir_img = cv2.resize(cv2.imread(f"{self.path}/images/IR/{file}"), (224, 224))
            mask_img = cv2.resize(cv2.imread(f"{self.path}/images/Mask/{file}"), (224, 224))

            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            ir_img = cv2.cvtColor(ir_img, cv2.COLOR_BGR2RGB)

            vis_img = cv2.bitwise_and(vis_img, vis_img, mask=mask_img[:, :, 0])
            ir_img = cv2.bitwise_and(ir_img, ir_img, mask=mask_img[:, :, 0])

            image_pairs.append([vis_img, ir_img])

        image_pairs = np.array(image_pairs)
        points = np.array(points)

        master_idx = np.arange(image_pairs.shape[0])

        # applies formulas from paper
        r_vals = (image_pairs[master_idx, 1, points[:, 0], points[:, 1], 0] -
                  image_pairs[master_idx, 0, points[:, 0], points[:, 1], 0]) / image_pairs[
                     master_idx, 0, points[:, 0], points[:, 1], 0]

        g_vals = (image_pairs[master_idx, 1, points[:, 0], points[:, 1], 1] -
                  image_pairs[master_idx, 0, points[:, 0], points[:, 1], 1]) / image_pairs[
                     master_idx, 0, points[:, 0], points[:, 1], 0]

        b_vals = (image_pairs[master_idx, 1, points[:, 0], points[:, 1], 2] -
                  image_pairs[master_idx, 0, points[:, 0], points[:, 1], 2]) / image_pairs[
                     master_idx, 0, points[:, 0], points[:, 1], 0]

        # do the fitting
        r_fit = np.polyfit(brightness, r_vals, 3)
        g_fit = np.polyfit(brightness, g_vals, 3)
        b_fit = np.polyfit(brightness, b_vals, 3)

        p_r = np.poly1d(r_fit)
        p_g = np.poly1d(g_fit)
        p_b = np.poly1d(b_fit)

        plt.rcParams.update({'font.size': 17})
        plt.figure()
        plt.xlabel("lux")
        plt.ylabel(r"$\rho_c$")

        plt.scatter(brightness, r_vals, color="red", linewidth=1, alpha=0.5)
        plt.scatter(brightness, g_vals, color="green", linewidth=1, alpha=0.5)
        plt.scatter(brightness, b_vals, color="blue", linewidth=1, alpha=0.5)

        ranger = np.linspace(0, max(brightness), 100)
        plt.plot(ranger, p_r(ranger), label="red (approx.)", linewidth=3, linestyle="solid",
                 color="red")
        plt.plot(ranger, p_g(ranger), label="green (approx.)", linewidth=3, linestyle="dashed",
                 color="green")
        plt.plot(ranger, p_b(ranger), label="blue (approx.)",linewidth=3,linestyle="dotted",
                 color="blue")
        plt.grid()
        plt.legend()

        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(self.path, 'plots',
                     f'results_{self.__class__.__name__}.pdf'), bbox_inches='tight')

        print("Coefficients:")
        print(p_r, p_g, p_b)

        return None, None


if __name__ == "__main__":
    gs = e1()
    gs.main()
