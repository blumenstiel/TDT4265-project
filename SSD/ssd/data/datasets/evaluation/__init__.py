import pathlib
import logging
import os
from datetime import datetime
from ssd.data.datasets import VOCDataset
from ssd.data.datasets import MNISTDetection, TDT4265Dataset, RDDDataset
from .voc import voc_evaluation
from .eval_detection_voc import eval_detection_voc


def evaluate(dataset, predictions, output_dir: pathlib.Path, **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[(boxes, labels, scores)]): Each item in the list represents the
            prediction results for one image. And the index should match the dataset index.
        output_dir: output folder, to save evaluation files or results.
    Returns:
        evaluation result
    """
    args = dict(
        dataset=dataset, predictions=predictions, output_dir=output_dir, **kwargs,
    )
    if isinstance(dataset, VOCDataset):
        return voc_evaluation(**args)
    if isinstance(dataset, MNISTDetection):
        return voc_detection_evaluation(**args)
    elif isinstance(dataset, (RDDDataset, TDT4265Dataset)):
        return voc_detection_evaluation(**args)
    else:
        raise NotImplementedError


def voc_detection_evaluation(dataset, predictions, output_dir, iteration=None):
    class_names = dataset.class_names

    pred_boxes_list = []
    pred_labels_list = []
    pred_scores_list = []
    gt_boxes_list = []
    gt_labels_list = []

    for i in range(len(dataset)):
        annotation = dataset.get_annotation(i)
        gt_boxes, gt_labels = annotation
        gt_boxes_list.append(gt_boxes)
        gt_labels_list.append(gt_labels)

        img_info = dataset.get_img_info(i)
        prediction = predictions[i]
        prediction = prediction.resize((img_info['width'], img_info['height'])).numpy()
        boxes, labels, scores = prediction['boxes'], prediction['labels'], prediction['scores']

        pred_boxes_list.append(boxes)
        pred_labels_list.append(labels)
        pred_scores_list.append(scores)
    result = eval_detection_voc(pred_bboxes=pred_boxes_list,
                                pred_labels=pred_labels_list,
                                pred_scores=pred_scores_list,
                                gt_bboxes=gt_boxes_list,
                                gt_labels=gt_labels_list,
                                iou_thresh=0.5,
                                use_07_metric=True)
    logger = logging.getLogger("SSD.inference")
    result_str = "mAP: {:.4f}\n".format(result["map"])
    metrics = {'mAP': result["map"]}
    for i, ap in enumerate(result["ap"]):
        if i == 0:  # skip background
            continue
        metrics[class_names[i]] = ap
        result_str += "{:<16}: {:.4f}\n".format(class_names[i], ap)
    logger.info(result_str)

    if iteration is not None:
        result_path = os.path.join(output_dir, 'result_{:07d}.txt'.format(iteration))
    else:
        result_path = os.path.join(output_dir, 'result_{}.txt'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    with open(result_path, "w") as f:
        f.write(result_str)

    return dict(metrics=metrics)