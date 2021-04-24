import torch
import pathlib
import numpy as np
import json
from ssd.container import Container
from PIL import Image



class TDT4265Dataset_sampling(torch.utils.data.Dataset):

    class_names = ('__background__',
                   'D00 - Linear Longitudinal Crack',
                   'D10 - Linear Lateral Crack',
                   'D20 - Alligator and Other Complex Cracks',
                   'D40 - Pothole')
    validation_percentage = 0.2

    def __init__(self, data_dir: str, split: str, transform=None, target_transform=None):
        data_dir = pathlib.Path(data_dir)
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        if split == "test":
            self.labels = self.read_labels(data_dir.parent.joinpath("labels_test.json"))
        else:
            self.split = split
            self.labels = self.read_labels(data_dir.parent.joinpath("labels.json"))
            self.image_ids.sort(key=lambda x: int(x))
            self.image_ids = self.split_dataset(split)
            self.image_ids = self.filter_image_ids()
            if split == "train":
                self.image_ids = self.remove_empty_images()
                self.sampling()


        print(f"Dataset loaded. Subset: {split}, number of images: {len(self)}")

    def sampling(self):
        print('Number images old:', len(self.image_ids))
        d00 = []
        d10 = []
        d20 = []
        d40 = []
        drop_d00 = []
        for id in self.image_ids:
            box, label = self._get_annotation(id)
            if len(label) == 1 and label[0] == 1:
                # find images that only contain D00
                drop_d00.append(id)
            # get lists of image ids for each label
            for l in label:
                if l == 1:
                    d00.append(id)
                if l == 2:
                    d10.append(id)
                if l == 3:
                    d20.append(id)
                if l == 4:
                    d40.append(id)
        print('D00:', len(d00))
        print('D10:', len(d10))
        print('D20:', len(d20))
        print('D40:', len(d40))
        print('drop_d00:', len(drop_d00))

        # drop 10,000 images that only contain D00
        drop = drop_d00[:7500]
        for id in drop:
            self.image_ids.remove(id)

        # empirically determined values that lead to a more or less balanced dataset
        d20 = d20 * 2
        d40 = d40 * 2
        self.image_ids.extend(d10)
        self.image_ids.extend(d20)
        self.image_ids.extend(d40)
        print('Number images new:', len(self.image_ids))
        d00 = []
        d10 = []
        d20 = []
        d40 = []
        for id in self.image_ids:
            box, label = self._get_annotation(id)
            for l in label:
                if l == 1:
                    d00.append(id)
                if l == 2:
                    d10.append(id)
                if l == 3:
                    d20.append(id)
                if l == 4:
                    d40.append(id)
        print('D00:', len(d00))
        print('D10:', len(d10))
        print('D20:', len(d20))
        print('D40:', len(d40))

    def filter_image_ids(self):
        image_ids_with_image_file = []
        for image_id in self.image_ids:
            image_path = self.data_dir.joinpath(f"{image_id}.jpg")
            if not image_path.is_file():
                continue
            image_ids_with_image_file.append(image_id)
        return image_ids_with_image_file

    def remove_empty_images(self):
        """
            Removes any images without objects for training
        """
        keep_idx = []
        for idx in range(len(self)):
            boxes, labels = self.get_annotation(idx)
            if len(labels) == 0:
                continue
            keep_idx.append(idx)
        image_ids = [self.image_ids[idx] for idx in keep_idx]
        return image_ids


    def get_categoryId_to_label_idx(self, labels):
        categories = labels["categories"]
        mapping = {}
        for category in categories:
            name = category["name"]
            category_id = category["id"]
            assert category_id not in mapping
            mapping[category_id] = self.class_names.index(name)
            assert mapping[category_id] != -1
        return mapping

    def read_labels(self, label_path):
        assert label_path.is_file(), \
            f"Did not find label file: {label_path.absolute()}"
        with open(label_path, "r") as fp:
            labels = json.load(fp)
        self.image_ids = list(set([x["id"] for x in labels["images"]]))
        labels_processed = {
            image_id: {"bboxes": [], "labels": []}
            for image_id in self.image_ids
        }
        self.image_info = {
            x["id"]: x for x in labels["images"]
        }
        category2label = self.get_categoryId_to_label_idx(labels)
        for label in labels["annotations"]:
            category = label["category_id"]
            label_cls = category2label[category]
            x0, y0, w, h = label["bbox"]
            x1 = x0 + w
            y1 = y0 + h
            box = [x0, y0, x1, y1]
            labels_processed[label["image_id"]]["bboxes"].append(box)
            labels_processed[label["image_id"]]["labels"].append(label_cls)
        return labels_processed

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        boxes, labels = self.get_annotation(index)
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        target = Container(boxes=boxes, labels=labels)
        return image, target, index

    def __len__(self):
        return len(self.image_ids)

    def split_dataset(self, split):
        num_train_images = int(len(self.image_ids) * (1-self.validation_percentage))
        if split == "train":
            image_ids = self.image_ids[:num_train_images]
        else:
            image_ids = self.image_ids[num_train_images:]
        return image_ids

    def _get_annotation(self, image_id):
        annotations = self.labels[image_id]
        bbox_key = "bboxes"
        boxes = np.zeros((len(annotations[bbox_key]), 4), dtype=np.float32)
        labels = np.zeros((len(annotations[bbox_key])), dtype=np.int64)
        for idx in range(len(boxes)):
            boxes[idx] = annotations["bboxes"][idx]
            labels[idx] = annotations["labels"][idx]
        return boxes, labels

    def get_annotation(self, index):
        image_id = self.image_ids[index]
        return self._get_annotation(image_id)

    def _read_image(self, image_id):
        image_path = self.data_dir.joinpath(f"{image_id}.jpg")
        assert image_path.is_file(), image_path
        image = Image.open(str(image_path)).convert("RGB")
        image = np.array(image)
        return image

    def get_img_info(self, index):
        image_id = self.image_ids[index]
        return self.image_info[image_id]