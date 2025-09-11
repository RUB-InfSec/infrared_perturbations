import torch

from attacks.Attack import Attack


class ClassificationAttack(Attack):
    def __init__(self, adv_perturbation, img, mask_coords, model, label, transform_num, pre_process, targeted,
                 target=None):
        self.label = label
        super().__init__(adv_perturbation, img, mask_coords, model, transform_num, pre_process, targeted, target)

    def untargeted_loss(self, predict):
        """
        Compute the untargeted loss given the predicted class probabilities.

        Args:
            predict: The normalized (using softmax) prediction of the neural network

        Returns:

        """
        best_other_label = torch.argmax(torch.hstack((predict[:self.label], predict[self.label + 1:])))
        if best_other_label >= self.label:
            best_other_label += 1  # Compensate for the index shift due to self.label being removed from the list
        confidence = float(predict[self.label]) - float(predict[best_other_label])

        success = torch.argmax(predict) != self.label
        return confidence, success

    def targeted_loss(self, predict):
        """
        Compute the targeted loss given the predicted class probabilities.

        Args:
            predict: The normalized (using softmax) prediction of the neural network

        Returns:

        """
        top10 = torch.topk(predict, k=10)
        summ = torch.sum(predict[top10[1][top10[1] != self.target]])
        confidence = float(summ) - float(predict[self.target])

        success = torch.argmax(predict) == self.target

        return confidence, success

    def loss_function(self, perturbed_img, mask_coords):
        """
        Loss function for gradient-free optimizers

        Args:
            mask_coords:
            perturbed_img: The adversarial example to evaluate

        Returns:
            confidence: The lower the value, the more successful the attack.
            success: Whether the attack is successful.
            torch.argmax(predict): The model's predicted label
        """
        with torch.no_grad():
            predict = torch.softmax(self.model(perturbed_img), 1)
            predict = torch.mean(predict, dim=0)

        if self.targeted:
            confidence, success = self.targeted_loss(predict)
        else:
            confidence, success = self.untargeted_loss(predict)

        self.num_query += perturbed_img.shape[0]
        return confidence, success, torch.argmax(predict)
