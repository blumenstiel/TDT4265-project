import os
import torch.utils.data
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image

from ssd.container import Container


class RDDDataset(torch.utils.data.Dataset):
    class_names = ('__background__',
                   'D00',
                   'D10',
                   'D20',
                   'D40')

    def __init__(self, data_dir, split, transform=None, target_transform=None):
        """Dataset for VOC data.
        Args:
            data_dir: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        image_sets_file = os.path.join(self.data_dir, "ImageSets", f"{self.split}.txt")
        self.image_ids = RDDDataset._read_image_ids(image_sets_file)
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}
        print(f"Dataset loaded. Subset: {split}, number of images: {len(self)}")

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        boxes, labels = self._get_annotation(image_id)
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        targets = Container(
            boxes=boxes,
            labels=labels,
        )
        return image, targets, index

    def get_annotation(self, index):
        image_id = self.image_ids[index]
        return self._get_annotation(image_id)

    def __len__(self):
        return len(self.image_ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):
        annotation_file = os.path.join(self.data_dir, "Annotations", f"{image_id}.xml")
        boxes = []
        labels = []
        objects = ET.parse(annotation_file).findall("object")
        for obj in objects:
            class_name = obj.find('name').text.strip()
            if class_name in self.class_names:
                bbox = obj.find('bndbox')
                # VOC dataset format follows Matlab, in which indexes start from 0
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                boxes.append([x1, y1, x2, y2])
                labels.append(self.class_dict[class_name])

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64))

    def get_img_info(self, index):
        img_id = self.image_ids[index]
        annotation_file = os.path.join(self.data_dir, "Annotations", f"{img_id}.xml")
        anno = ET.parse(annotation_file).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def _read_image(self, image_id):
        image_file = os.path.join(self.data_dir, f"JPEGImages/{image_id}.jpg")
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image
