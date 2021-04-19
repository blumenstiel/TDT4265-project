from torch import nn
from ssd.modeling.backbone.vgg import VGG
from ssd.modeling.backbone.basic import BasicModel
from ssd.modeling.backbone.baseline1 import Baseline1
from ssd.modeling.backbone.mobilenetv3 import MobileNetV3
from ssd.modeling.backbone.improved_d import ImprovedModel_d
from ssd.modeling.backbone.improved_d import Resnext
from ssd.modeling.backbone.improved_avgpool import AvgPoolModel
from ssd.modeling.box_head.box_head import SSDBoxHead
from ssd.utils.model_zoo import load_state_dict_from_url
from ssd import torch_utils


class SSDDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.box_head = SSDBoxHead(cfg)
        print(
            "Detector initialized. Total Number of params: ",
            f"{torch_utils.format_params(self)}")
        print(
            f"Backbone number of parameters: {torch_utils.format_params(self.backbone)}")
        print(
            f"SSD Head number of parameters: {torch_utils.format_params(self.box_head)}")

    def forward(self, images, targets=None):
        features = self.backbone(images)
        detections, detector_losses = self.box_head(features, targets)
        if self.training:
            return detector_losses
        return detections


def build_backbone(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    if backbone_name == "basic":
        model = BasicModel(cfg)
        return model
    if backbone_name == "baseline1":
        model = Baseline1(cfg)
        return model
    if backbone_name == "vgg":
        model = VGG(cfg)
        if cfg.MODEL.BACKBONE.PRETRAINED:
            state_dict = load_state_dict_from_url(
                "https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth")
            model.init_from_pretrain(state_dict)
        return model
    if backbone_name == "mobilenetv3":
        model = MobileNetV3()
        if cfg.MODEL.BACKBONE.PRETRAINED:
            model.load_state_dict(load_state_dict_from_url(
                'https://github.com/d-li14/mobilenetv3.pytorch/raw/master/pretrained/mobilenetv3-large-1cd25616.pth'),
                strict=False)
        return model
    if backbone_name == "improved_d":
        model = ImprovedModel_d(cfg)
        return model
    if backbone_name == "improved_minpool":
        model = AvgPoolModel(cfg)
        return model
    if backbone_name == "resnext":
        model = Resnext(cfg)
        return model