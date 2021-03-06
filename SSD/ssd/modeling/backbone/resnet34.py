import torch
import torchvision.models as models
from torch import nn


class Resnet34(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = models.resnet34(pretrained=cfg.MODEL.BACKBONE.PRETRAINED, progress=True)

        # p value for dropout
        self.dropout = cfg.MODEL.BACKBONE.DROPOUT

        self.add_lay5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )
        self.add_lay6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )
        self.add_lay7 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out_features = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        out_features.append(nn.Dropout(p=self.dropout)(x))
        x = self.model.layer3(x)
        out_features.append(nn.Dropout(p=self.dropout)(x))
        x = self.model.layer4(x)
        out_features.append(nn.Dropout(p=self.dropout)(x))
        x = self.add_lay5(x)
        out_features.append(nn.Dropout(p=self.dropout)(x))
        x = self.add_lay6(x)
        out_features.append(nn.Dropout(p=self.dropout)(x))
        x = self.add_lay7(x)
        out_features.append(nn.Dropout(p=self.dropout)(x))

        return tuple(out_features)

if __name__ == '__main__':
    # Testing resnet layers
    # for testing: set dropout manually and replace cfg.MODEL.BACKBONE.PRETRAINED with "True"
    model = Resnet34('_')
    print(model)
    out_features = model.forward(torch.zeros((1, 3, 300, 300)))
    for i in out_features:
        print(i.shape)