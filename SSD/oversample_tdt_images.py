import xml.etree.ElementTree as ET
import json
import numpy as np


class_names = ('__background__',
               'D00 - Linear Longitudinal Crack',
               'D10 - Linear Lateral Crack',
               'D20 - Alligator and Other Complex Cracks',
               'D40 - Pothole')

def read_labels(label_path):
    global image_ids, labels_processed
    with open(label_path, "r") as fp:
        labels = json.load(fp)
    image_ids = list(set([x["id"] for x in labels["images"]]))
    labels_processed = {
        image_id: {"bboxes": [], "labels": []}
        for image_id in image_ids
    }
    image_info = {
        x["id"]: x for x in labels["images"]
    }
    category2label = get_categoryId_to_label_idx(labels)
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

def get_categoryId_to_label_idx(labels):
    categories = labels["categories"]
    mapping = {}
    for category in categories:
        name = category["name"]
        category_id = category["id"]
        assert category_id not in mapping
        mapping[category_id] = class_names.index(name)
        assert mapping[category_id] != -1
    return mapping

def _get_annotation(labels, image_id):
    annotations = labels[image_id]
    bbox_key = "bboxes"
    boxes = np.zeros((len(annotations[bbox_key]), 4), dtype=np.float32)
    labels = np.zeros((len(annotations[bbox_key])), dtype=np.int64)
    for idx in range(len(boxes)):
        boxes[idx] = annotations["bboxes"][idx]
        labels[idx] = annotations["labels"][idx]
    return boxes, labels

def get_annotation(labels, index):
    image_id = image_ids[index]
    return _get_annotation(labels, image_id)

if __name__ == "__main__":
    labels = read_labels("datasets/tdt4265/labels.json")
    image_ids.sort(key=lambda x: int(x))
    num_train_images = int(len(image_ids) * (1 - 0.2))
    #image_ids = image_ids[:num_train_images]
    print(len(image_ids))

    d00 = []
    d10 = []
    d20 = []
    d40 = []
    drop = []
    print('Number images old:', len(image_ids))

    for idx in range(len(image_ids)):
        box, label = get_annotation(labels, idx)
        if len(label) == 1 and label[0]==1:
            drop.append(idx)
        for l in label:
            if l == 1:
                d00.append(idx)
            if l == 2:
                d10.append(idx)
            if l == 3:
                d20.append(idx)
            if l == 4:
                d40.append(idx)
    print('D00:', len(d00))
    print('D10:', len(d10))
    print('D20:', len(d20))
    print('D40:', len(d40))
    print(len(drop))

    print('Total labels:', len(d00) + len(d10) + len(d20) + len(d40))

    #image_ids_copy = image_ids.copy()
    #for im in drop:
    #    image_ids_copy.remove(image_ids[im])
#
    #d00 = []
    #d10 = []
    #d20 = []
    #d40 = []
    #drop = []
    #print('Number images old:', len(image_ids_copy))
#
    #for idx in range(len(image_ids_copy)):
    #    box, label = get_annotation(labels, idx)
    #    if len(label) == 1 and label[0] == 1:
    #        drop.append(idx)
    #    for l in label:
    #        if l == 1:
    #            d00.append(idx)
    #        if l == 2:
    #            d10.append(idx)
    #        if l == 3:
    #            d20.append(idx)
    #        if l == 4:
    #            d40.append(idx)
    #print('D00:', len(d00))
    #print('D10:', len(d10))
    #print('D20:', len(d20))
    #print('D40:', len(d40))
#
#

    #d10 = list(set(d10))
    #content_list.extend(d10)
#
    #print(len(content_list))
    #with open("datasets/RDD2020_filtered/ImageSets/train_oversampling.txt", 'w') as f:
    #    for sample in content_list:
    #        f.write(sample)