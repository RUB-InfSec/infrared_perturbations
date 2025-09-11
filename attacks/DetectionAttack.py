import torch
import ultralytics

from attacks.Attack import Attack
from utils.od_utils import matching_predictions


class DetectionAttack(Attack):
    def __init__(self, adv_perturbation, attack_type, img, mask_coords, model, true_label, transform_num,
                 pre_process, iou_treshold, target_label=None):
        self.iou_treshold = iou_treshold
        self.confidence_treshold = 0.3
        self.true_label = true_label
        self.attack_type = attack_type
        self.target_label = target_label
        super().__init__(adv_perturbation, img, mask_coords, model, transform_num, pre_process, False)

    def single_img_loss_function(self, predict, mask_coord, width, height):
        best_match_ious, confidences, predicted_classes = matching_predictions(predict, mask_coord, width, height)

        if best_match_ious.numel() == 0:
            if self.attack_type == "hide":
                return 0, True, "hide"
            else:
                return 1, False, "hide"

        hidden = all(
            [best_match_iou < self.iou_treshold or confidence < self.confidence_treshold for best_match_iou, confidence
             in zip(best_match_ious, confidences)])

        if self.attack_type == "hide":
            return torch.mean(confidences).item(), hidden, "hide" if hidden else int(
                torch.mode(predicted_classes)[0].item())

    def loss_function(self, perturbed_img, mask_coords):
        with torch.no_grad():
            if isinstance(self.model, ultralytics.models.yolo.model.YOLO):
                predict = self.model(perturbed_img, show=False, verbose=False)
            else:
                predict = self.model(perturbed_img)

        # Increase number of queries that the attack took
        self.num_query += perturbed_img.shape[0]

        losses = []
        successes = []
        predicted_labels = []

        for prediction, mask_coord in zip(predict, mask_coords):
            loss, success, predicted_label = self.single_img_loss_function(prediction, mask_coord,
                                                                           perturbed_img.shape[3],
                                                                           perturbed_img.shape[2])

            losses.append(loss)
            successes.append(success)
            predicted_labels.append(predicted_label)

        # Return the average of the losses instead of computing the loss of the average, as the latter is not feasible for OD.
        # For success and predicted label return the majority vote. As the optimizer does not use these values, anyway, this does not make a huge difference.
        return sum(losses) / len(losses), sum(successes) > 0.5 * len(successes), max(set(predicted_labels),
                                                                                     key=predicted_labels.count)
