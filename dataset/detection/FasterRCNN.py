import copy
import cv2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN

import torch
import torchvision

class Collator(object):
    def __init__(self, device):
        self.device = device

    def __call__(self, batch):
        imgs = torch.stack([item[0] for item in batch])
        targets = []
        for item in batch:
            dictionary = {}
            for key, value in item[1].items():
                dictionary[key] = value.to(self.device)
            targets.append(dictionary)
        return imgs.to(self.device), targets

def build_model(num_classes):
    # Choosing the v2 model as it seems to perform better than the regular one
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,
                                                      num_classes + 1)  # Class label 0 is reserved for background in Faster R-CNN

    # Define a post-inference hook to adjust predictions
    def post_inference_hook(predictions):
        adjusted_predictions = []
        for prediction in predictions:
            # Filter out predictions that do not exceed the detection threshold (setting it to 0.25 to make it equal to the way it's handled in YOLO)
            prediction['boxes'] = prediction['boxes'][prediction['scores'] >= 0.25]
            prediction['labels'] = prediction['labels'][prediction['scores'] >= 0.25]
            prediction['scores'] = prediction['scores'][prediction['scores'] >= 0.25]
            for i, (box, label, score) in enumerate(
                    zip(prediction['boxes'], prediction['labels'], prediction['scores'])):
                if label == 0:
                    prediction['boxes'][i] = torch.stack((prediction['boxes'][:i], prediction['boxes'][i + 1:]))
                    prediction['labels'][i] = torch.stack((prediction['labels'][:i], prediction['labels'][i + 1:]))
                    prediction['scores'][i] = torch.stack((prediction['scores'][:i], prediction['scores'][i + 1:]))
                else:
                    prediction['labels'][i] = prediction['labels'][i] - 1
            adjusted_predictions.append(prediction)
        return adjusted_predictions

    # Override the forward method to apply the post-inference hook
    def forward(self, images, targets=None):
        if len(images.size()) == 3:  # Add batch dimension if missing
            images = torch.unsqueeze(images, 0)
        if targets:
            return super(FasterRCNN, self).forward(images, targets)
        else:
            raw_predictions = super(FasterRCNN, self).forward(images)
            adjusted_predictions = post_inference_hook(raw_predictions)
            return adjusted_predictions

    # Attach the modified forward method to the model
    model.forward = forward.__get__(model)

    return model


def plot_predictions_faster_rcnn(img, result):
    image = copy.deepcopy(img)
    for w, bbox in enumerate(result['boxes']):
        x1, y1, x2, y2 = bbox.cpu().detach().numpy()
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(image, str(result['labels'][w].item()), (int(x1) - 20, int(y1) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(image, str(result['scores'][w].item()), (int(x1) - 20, int(y1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image