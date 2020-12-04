import os
import warnings
import xml.etree.ElementTree as ET
from os import listdir
from os.path import join

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def collate_fn(batch):
    return tuple(zip(*batch))


class MaskDatasetRetriever(Dataset):
    def __init__(self, root_path, transform=None, target_transform=None):
        self.anno_paths = sorted([join(root_path, x) for x in listdir(root_path) if x.split(".")[-1] in ["xml"]])
        self.labels = ['face_mask', 'face']
        self.transform = transform
        self.target_transform = target_transform

    def _get_annotations(self, path):

        tree = ET.parse(path)

        boxes = []
        labels = []

        for child in tree.getroot():
            if child.tag != 'object':
                continue

            bndbox = child.find('bndbox')
            box = [
                float(bndbox.find(t).text)
                for t in ['xmin', 'ymin', 'xmax', 'ymax']
            ]

            if child.find('name').text in self.labels:
                label = self.labels.index(child.find('name').text) + 1
            else:
                warnings.warn("label {} not found, using default 1".format(child.find('name').text))
                label = 1

            boxes.append(box)
            labels.append(label)

        return np.array(boxes), np.array(labels)

    def _read_content(self, xml_file: str):
        # experimental

        tree = ET.parse(xml_file)
        root = tree.getroot()

        list_with_all_boxes = []
        labels = []

        for boxes in root.iter('object'):

            filename = root.find('filename').text
            if root.find('name').text in self.labels:
                label = self.labels.index(root.find('name').text) + 1
            else:
                label = 1

            ymin, xmin, ymax, xmax = None, None, None, None

            ymin = int(boxes.find("bndbox/ymin").text)
            xmin = int(boxes.find("bndbox/xmin").text)
            ymax = int(boxes.find("bndbox/ymax").text)
            xmax = int(boxes.find("bndbox/xmax").text)

            list_with_single_boxes = [xmin, ymin, xmax, ymax]
            list_with_all_boxes.append(list_with_single_boxes)
            labels.append(label)

        return filename, list_with_all_boxes, labels

    def __getitem__(self, index: int):
        anno_path = self.anno_paths[index]
        if os.path.exists(anno_path.replace("xml", "jpg")):
            img_path = anno_path.replace("xml", "jpg")
        elif os.path.exists(anno_path.replace("xml", "png")):
            img_path = anno_path.replace("xml", "png")
        else:
            return self.__getitem__(min(index + 1, self.__len__() - 1))

        boxes, labels = self._get_annotations(anno_path)  # ok

        image = cv2.imread(img_path).copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if len(boxes) == 0:
            return self.__getitem__(min(index + 1, self.__len__() - 1))

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
            # target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample[1])))).permute(1, 0)
            # target['boxes'][:, [0, 1, 2, 3]] = target['boxes'][:, [1, 0, 3, 2]]  # yxyx: be warning

        if self.target_transform:
            boxes, labels = self.target_transform(torch.Tensor(boxes),
                                                  torch.Tensor(labels))

        return image, boxes, labels

    def __len__(self):
        return len(self.anno_paths)
