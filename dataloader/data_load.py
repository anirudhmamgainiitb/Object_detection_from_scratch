import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class UnderwaterDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=416, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.transform = transform
        self.imgs = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.imgs)

    def load_image(self, index):
        img_path = os.path.join(self.img_dir, self.imgs[index])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img

    def load_yolo_labels(self, index, grid_size=52, num_classes=7):
        label_name = self.imgs[index].replace(".jpg", ".txt").replace(".png", ".txt")
        label_path = os.path.join(self.label_dir, label_name)

        target = torch.zeros((grid_size, grid_size, 5 + num_classes))

        if not os.path.exists(label_path):
            return target

        with open(label_path, "r") as f:
            for line in f.readlines():
                cls, x, y, w, h = map(float, line.strip().split())

                gx = int(x * grid_size)
                gy = int(y * grid_size)

                gx = min(gx, grid_size - 1)
                gy = min(gy, grid_size - 1)

                cell_x = x * grid_size - gx
                cell_y = y * grid_size - gy
                cell_w = w * grid_size
                cell_h = h * grid_size

                target[gy, gx, 0:4] = torch.tensor([cell_x, cell_y, cell_w, cell_h])
                target[gy, gx, 4] = 1.0
                target[gy, gx, 5 + int(cls)] = 1.0

        return target


    def __getitem__(self, index):
        img = self.load_image(index)
        target = self.load_yolo_labels(index)

        if self.transform:
            img = self.transform(img)

        return img, target
