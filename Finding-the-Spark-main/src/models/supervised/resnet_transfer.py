from torchvision.models.segmentation import fcn_resnet101
import torch
from torch import nn
import torch.nn.functional as F

class FCNResnetTransfer(nn.Module):
    def __init__(self, input_channels, output_channels, scale_factor=50, **kwargs):
        """
        Loads the fcn_resnet101 model from torch hub,
        then replaces the first and last layer of the network
        in order to adapt it to our current problem, 
        the first convolution of the fcn_resnet must be changed
        to an input_channels -> 64 Conv2d with (7,7) kernel size,
        (2,2) stride, (3,3) padding and no bias.

        The last layer must be changed to be a 512 -> output_channels
        conv2d layer, with (1,1) kernel size and (1,1) stride. 

        A final pooling layer must then be added to pool each 50x50
        patch down to a 1x1 image, as the original FCN resnet is trained to
        have the segmentation be the same resolution as the input.
        
        Input:
            input_channels: number of input channels of the image
            of shape (batch, input_channels, width, height)
            output_channels: number of output channels of prediction,
            prediction is shape (batch, output_channels, width//scale_factor, height//scale_factor)
            scale_factor: number of input pixels that map to 1 output pixel,
            for example, if the input is 800x800 and the output is 16x6
            then the scale factor is 800/16 = 50.
        """
        super(FCNResnetTransfer, self).__init__()
        # Load the model fcn_resnet101 from segmentation model weights from online like torch.hub
        FCNmodel = fcn_resnet101(pretrained = True, progress = True)
        # Replace the first and last layer of the network so that the number of channels fits with
        # the number of channels of the image and the number of classes we are predicting
        firstlayer = nn.Conv2d(input_channels, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        FCNmodel.backbone.conv1 = firstlayer
        lastlayer = nn.Conv2d(512, output_channels, kernel_size=1, stride=1)
        FCNmodel.classifier[4] = lastlayer
        # A final pooling layer must then be added to pool each 50x50
        # patch down to a 1x1 image, as the original FCN resnet is trained to
        # have the segmentation be the same resolution as the input.
        self.finallayer = nn.AdaptiveAvgPool2d(1)
        self.scale_factor = scale_factor
        self.FCNmodel = FCNmodel
        

      
    def forward(self, x):
        """
        Runs predictions on the modified FCN resnet
        followed by pooling

        Input:
            x: image to run a prediction of, of shape
            (batch, self.input_channels, width, height)
            with width and height divisible by
            self.scale_factor
        Output:
            pred_y: predicted labels of size
            (batch, self.output_channels, width//self.scale_factor, height//self.scale_factor)
        """
        # Be careful of the output data structure of model you loaded (e.g. dict, tuple, etc.)
        #running predictions on the FCN resnet
        x = self.FCNmodel(x)['out']
        #do this to get the resolution (pooling)
        x = self.finallayer(x)
        #resize output
        return F.interpolate(x, scale_factor = self.scale_factor, mode = 'bilinear', align_corners = False)