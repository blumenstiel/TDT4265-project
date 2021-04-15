from ssd.modeling.box_head.prior_box import PriorBox
from .target_transform import SSDTargetTransform
from .transforms import *


def build_transforms(cfg, is_train=True):
    if is_train:
        transform = [
            ConvertFromInts(),
            ToPercentCoords(),
        ]
        if cfg.DATA_AUGMENTATION.RANDOMCROP:
            transform.append(RandomSampleCrop())
        transform.append(Resize(cfg.INPUT.IMAGE_SIZE))
        transform.append(SubtractMeans(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD))
        if cfg.DATA_AUGMENTATION.INVERT:
            transform.append(InvertImage())
        if cfg.DATA_AUGMENTATION.MIRROR:
            transform.append(RandomMirror())
        if cfg.DATA_AUGMENTATION.COLORJITTER:
            transform.append(ColorJitter)
        transform.append(ToTensor())
    else:
        transform = [
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD),
        ]
        if cfg.DATA_AUGMENTATION.INVERT:
            transform.append(InvertImage())
            transform.append(ToTensor())

    transform = Compose(transform)
    return transform


def build_target_transform(cfg):
    transform = SSDTargetTransform(PriorBox(cfg)(),
                                   cfg.MODEL.CENTER_VARIANCE,
                                   cfg.MODEL.SIZE_VARIANCE,
                                   cfg.MODEL.THRESHOLD)
    return transform
