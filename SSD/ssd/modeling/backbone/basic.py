from torch import nn


class BasicModel(nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """
    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS

        self.feature_extractor0 = nn.Sequential(
            nn.Conv2d(in_channels=image_channels, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=output_channels[0], kernel_size=3, padding=1, stride=2),
        )

        self.feature_extractor1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=output_channels[0], out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=output_channels[1], kernel_size=3, padding=1, stride=2),
        )

        self.feature_extractor2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=output_channels[1], out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=output_channels[2], kernel_size=3, padding=1, stride=2),
        )

        self.feature_extractor3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=output_channels[2], out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=output_channels[3], kernel_size=3, padding=1, stride=2),
        )

        self.feature_extractor4 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=output_channels[3], out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=output_channels[4], kernel_size=3, padding=1, stride=2),
        )

        self.feature_extractor5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=output_channels[4], out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=output_channels[5], kernel_size=3, padding=0),
        )

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        out = self.feature_extractor0(x)
        out_features.append(out)
        out = self.feature_extractor1(out)
        out_features.append(out)
        out = self.feature_extractor2(out)
        out_features.append(out)
        out = self.feature_extractor3(out)
        out_features.append(out)
        out = self.feature_extractor4(out)
        out_features.append(out)
        out = self.feature_extractor5(out)
        out_features.append(out)

        for idx, feature in enumerate(out_features):
            w, h = self.output_feature_shape[idx]
            expected_shape = (self.output_channels[idx], h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)

