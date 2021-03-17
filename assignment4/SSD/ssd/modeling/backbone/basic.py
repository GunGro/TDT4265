import torch
from torch import nn

def create_block(output_channels, i):
    print(output_channels)
    if i == len(output_channels) - 2:
        print('hei')
        return nn.Sequential(
              nn.ReLU()
             ,nn.Conv2d(output_channels[i], 128, kernel_size=3, stride=1, padding=1)
             ,nn.ReLU()
             ,nn.Conv2d(128, output_channels[i+1], kernel_size=3, stride=1, padding=0)
             )
    elif i == 1:
        return nn.Sequential(
              nn.ReLU()
             ,nn.Conv2d(output_channels[i], 256, kernel_size=3, stride=1, padding=1)
             ,nn.ReLU()
             ,nn.Conv2d(256, output_channels[i+1], kernel_size=3, stride=2, padding=1)
             )
    else:
        return nn.Sequential(
              nn.ReLU() 
             ,nn.Conv2d(output_channels[i], 128, kernel_size=3, stride=1, padding=1)
             ,nn.ReLU()
             ,nn.Conv2d(128, output_channels[i+1], kernel_size=3, stride=2, padding=1)
             )


class BasicModel(torch.nn.Module):
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
        
        self.start = nn.Sequential(
          nn.Conv2d(image_channels, 32, kernel_size=3,stride=1,padding=1)
         ,nn.MaxPool2d(2, 2)
         ,nn.ReLU()
         
         ,nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
         ,nn.MaxPool2d(2, 2)
         ,nn.ReLU()

         ,nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1)
         ,nn.ReLU()
         ,nn.Conv2d(64, output_channels[0], kernel_size=3, stride=2, padding=1)
         )
        
        self.blocks = [0]*(len(output_channels)-1)
        for i in range(len(output_channels)-1):
            self.blocks[i] = create_block(output_channels, i)



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
        x = self.start(x)
        out_features = []
        for block in self.blocks:
            with torch.no_grad():
                out_features.append(x)
            x = block(x)
        with torch.no_grad():
            out_features.append(x)
        for idx, feature in enumerate(out_features):
            w, h = self.output_feature_shape[idx]
            expected_shape = (self.output_channels[idx], h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"


       

        return tuple(out_features)

