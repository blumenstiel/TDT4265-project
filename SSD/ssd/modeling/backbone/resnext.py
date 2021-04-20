import torch
import torchvision.models as models
from torch import nn


class Resnext(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = models.resnext50_32x4d(pretrained=cfg.MODEL.BACKBONE.PRETRAINED, progress=True)

        self.add_lay5 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, stride=1),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1),
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
        out_features.append(x)
        x = self.model.layer3(x)
        out_features.append(x)
        x = self.model.layer4(x)
        out_features.append(x)
        x = self.add_lay5(x)
        out_features.append(x)
        x = self.add_lay6(x)
        out_features.append(x)
        x = self.add_lay7(x)
        out_features.append(x)

        #for i in out_features:
        #    print(i.shape)

        return tuple(out_features)


#model = Resnext('_')
#print(model)
#model.forward(torch.zeros((1, 3, 300, 300)))

