import os
import pickle

import cv2
import torch
from scipy import ndimage

from experiments.Experiment import Experiment
from utils import utils
from utils.cnn_utils import load_trained_classifier
from utils.utils import fix_randomness


class e7(Experiment):

    def get_tasks(self):
        return None

    def run(self, args):
        adv_img_path = "./experiments/e3_classification/results/untargeted/0/GTSRB/10/192"

        assert bool(os.listdir())
        fix_randomness(42)

        with open('./dataset/classification/GTSRB/gtsrb_test.pkl', 'rb') as dataset:
            test_data = pickle.load(dataset)
            images, labels = test_data['data'], test_data['labels']

        total_images = len(os.listdir(adv_img_path))

        params = {'no_defense': 0, 'non-local': 10, 'local': 5, 'adversarial_training': 0 }

        for name, filter_size in params.items():
            accuracy_benign, accuracy_adversarial = self.evaluate_accuracy(images, filter=name, size=filter_size,
                                                                           adv_img_path=adv_img_path)
            accuracy_benign /= total_images
            accuracy_adversarial /= total_images

            if 'local' in name:
                print(
                    f"Filter {name} with size {filter_size} - CA: {accuracy_benign:.02f}, ASR: {(1.0 - accuracy_adversarial):.02f}")
            else:
                print(
                    f"{name} - CA: {accuracy_benign:.02f}, ASR: {(1.0 - accuracy_adversarial):.02f}")

    def evaluate(self):
        print("Please use the argument --mode run to evaluate this experiment. Results are directly shown when this experiment is executed.")
        return None, None

    def plot(self, results, queries):
        pass

    def evaluate_single_image(self, img, filter, size):

        if filter != 'adversarial_training':
            pre_process, model = load_trained_classifier("GTSRB", 'GtsrbCNN', False)

            if filter == "local":
                img = ndimage.median_filter(img, size, mode='reflect')
            elif filter == "non-local":
                img = cv2.fastNlMeansDenoisingColored(img, None, size, size, 3,
                                                      11)
        else:
            pre_process, model = load_trained_classifier("GTSRB", 'GtsrbCNN', True)

        img = pre_process(img).unsqueeze(0).to(utils.get_device())
        predict = torch.softmax(model(img)[0], 0)
        return int(torch.argmax(predict).data)


    def evaluate_accuracy(self, images, filter, size, adv_img_path):
        accuracy_benign = 0
        accuracy_adversarial = 0
        for image in os.listdir(adv_img_path):
            index = int(image.title().split('_')[0])
            label = int(image.title().split('_')[1].split('-')[0])

            if self.evaluate_single_image(images[index], filter, size) == label:
                accuracy_benign += 1

            if self.evaluate_single_image(cv2.resize(cv2.imread(f'{adv_img_path}/{image}'), (32, 32)), filter, size) == label:
                accuracy_adversarial += 1

        return accuracy_benign, accuracy_adversarial

if __name__ == "__main__":
    gs = e7()
    gs.main()