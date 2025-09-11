import numpy as np
import torch
import torchvision.ops


def best_matching_prediction(predict, mask_coord, width, height):
    y_min, y_max, x_min, x_max = np.min(mask_coord[0]), np.max(mask_coord[0]), np.min(mask_coord[1]), np.max(
        mask_coord[1])
    bbox = torch.tensor([x_min / width, y_min / height, x_max / width, y_max / height])
    if isinstance(predict, dict):  # Faster RCNN
        if torch.numel(predict['boxes']) == 0:
            return -1, -1, "hide"
        predicted_bbox = predict['boxes'].to('cpu')
        for i in range(predicted_bbox.shape[0]):
            predicted_bbox[i] = torch.tensor(
                [predicted_bbox[i][0] / width, predicted_bbox[i][1] / height, predicted_bbox[i][2] / width,
                 predicted_bbox[i][3] / height])
        best_match_iou, best_match_index = torch.max(torchvision.ops.box_iou(bbox.unsqueeze(0), predicted_bbox), dim=1)
        best_match_iou, best_match_index = best_match_iou.item(), best_match_index.item()
        confidence = predict['scores'][best_match_index].item()
        predicted_cls = int(predict['labels'][best_match_index].item())
        return best_match_iou, confidence, predicted_cls
    else:  # YOLO
        if predict.boxes.xyxyn.nelement() == 0:
            return -1, -1, "hide"
        best_match_iou, best_match_index = torch.max(
            torchvision.ops.box_iou(bbox.unsqueeze(0), predict.boxes.xyxyn.to('cpu')), dim=1)
        best_match_iou, best_match_index = best_match_iou.item(), best_match_index.item()
        confidence = predict.boxes.conf[best_match_index].item()
        predicted_cls = int(predict.boxes.cls[best_match_index].item())
        return best_match_iou, confidence, predicted_cls


def matching_predictions(predict, mask_coord, width, height):
    y_min, y_max, x_min, x_max = np.min(mask_coord[0]), np.max(mask_coord[0]), np.min(mask_coord[1]), np.max(
        mask_coord[1])
    bbox = torch.tensor([x_min / width, y_min / height, x_max / width, y_max / height])
    if isinstance(predict, dict):  # Faster RCNN
        if torch.numel(predict['boxes']) == 0:
            return torch.Tensor(), torch.Tensor(), "hide"
        predicted_bbox = predict['boxes'].to('cpu')
        for i in range(predicted_bbox.shape[0]):
            predicted_bbox[i] = torch.tensor(
                [predicted_bbox[i][0] / width, predicted_bbox[i][1] / height, predicted_bbox[i][2] / width,
                 predicted_bbox[i][3] / height])

        ious = torchvision.ops.box_iou(bbox.unsqueeze(0), predicted_bbox).squeeze(0)
        matching_indices = torch.argwhere(ious).squeeze(1)
        return ious[matching_indices], predict['scores'][matching_indices], predict['labels'][matching_indices]
    else:  # YOLO
        if predict.boxes.xyxyn.nelement() == 0:
            return torch.Tensor(), torch.Tensor(), "hide"
        ious = torchvision.ops.box_iou(bbox.unsqueeze(0), predict.boxes.xyxyn.to('cpu')).squeeze(0)
        matching_indices = torch.argwhere(ious).squeeze(1)
        return ious[matching_indices], predict.boxes.conf[matching_indices], predict.boxes.cls[matching_indices]
