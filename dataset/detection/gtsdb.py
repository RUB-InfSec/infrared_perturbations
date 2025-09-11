import os

import torch.utils.data
from torchvision.io import read_image


class GTSDBDataset(torch.utils.data.Dataset):
    def __init__(self, root, train, num_classes):
        self.root = root
        self.train = train
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images", "train" if self.train else "val"))))
        self.num_classes = num_classes

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = read_image(os.path.join(self.root, "images", "train" if self.train else "val", self.imgs[idx]))

        label_file_path = os.path.join(self.root, "labels", "train" if self.train else "val",
                                       f"{self.imgs[idx].split('.')[0]}.txt")
        if os.path.exists(label_file_path) and os.path.isfile(label_file_path):
            with open(label_file_path, 'r') as label_file:
                labels = []
                boxes = []
                for line in label_file:
                    labels.append(int(line.split()[0]))

                    # Convert from YOLO format to Faster R-CNN format
                    x_center, y_center, width, height = [float(val) for val in line.split()[1:]]
                    boxes.append([
                        (x_center - width / 2) * img.shape[2],
                        (y_center - height / 2) * img.shape[1],
                        (x_center + width / 2) * img.shape[2],
                        (y_center + height / 2) * img.shape[1]
                    ])

            target = {
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64)
            }
        else:  # If label file does not exist, there is no bounding box in the corresponding image
            target = {
                'boxes': torch.empty((0, 4), dtype=torch.float32),
                'labels': torch.empty((0,), dtype=torch.int64)
            }

        # img: has to be an image tensor (C, H, W)
        # target: has to be a dictionary with the following entries:
        #   - boxes (FloatTensor[N, 4]): [x1, y1, x2, y2]
        #   - labels (Int64Tensor[N]): the class label for each box

        return img / 255.0, target
