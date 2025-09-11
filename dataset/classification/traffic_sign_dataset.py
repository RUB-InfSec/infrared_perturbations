import torch
from torch.utils.data import Dataset


class TrafficSignDataset(Dataset):

    def __init__(self, x, y, transform):
        self.x = x
        if isinstance(y, list):
            self.y = y
        else:
            self.y = torch.LongTensor(y)

        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        if self.transform is not None:
            _x = self.transform(self.x[item])
        else:
            _x = self.x[item]

        _y = self.y[item]
        return _x, _y
