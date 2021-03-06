from ssd.modeling.box_head.prior_box import PriorBox
from .target_transform import SSDTargetTransform
from .transforms import *


def build_transforms(cfg, is_train=True):
    if is_train:
        transform = []
        if cfg.DATA_AUGMENTATION.INVERT:
            transform.append(InvertImage())
        transform.append(ConvertFromInts())
        if cfg.DATA_AUGMENTATION.RANDOMCROP and cfg.DATA_AUGMENTATION.RATIOCROP:
            transform.append(RandomSampleCrop(image_size=cfg.INPUT.IMAGE_SIZE))
        elif cfg.DATA_AUGMENTATION.RANDOMCROP:
            transform.append(RandomSampleCrop())
        elif cfg.DATA_AUGMENTATION.RATIOCROP:
            transform.append(ImageRatioCrop(cfg.INPUT.IMAGE_SIZE))
        elif cfg.DATA_AUGMENTATION.PADDING:
            transform.append(PadImage(cfg.INPUT.IMAGE_SIZE))
        if cfg.DATA_AUGMENTATION.MIRROR:
            transform.append(RandomMirror())
        if cfg.DATA_AUGMENTATION.COLORJITTER:
            transform.append(ColorJitter())
        transform.append(ToPercentCoords())
        transform.append(Resize(cfg.INPUT.IMAGE_SIZE))
        transform.append(SubtractMeans(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD))
        transform.append(ToTensor())
    else:
        transform = []
        if cfg.DATA_AUGMENTATION.INVERT:
            transform.append(InvertImage())
        transform.append(Resize(cfg.INPUT.IMAGE_SIZE))
        transform.append(SubtractMeans(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD))
        transform.append(ToTensor())

    transform = Compose(transform)
    return transform


def build_target_transform(cfg):
    transform = SSDTargetTransform(PriorBox(cfg)(),
                                   cfg.MODEL.CENTER_VARIANCE,
                                   cfg.MODEL.SIZE_VARIANCE,
                                   cfg.MODEL.THRESHOLD)
    return transform
