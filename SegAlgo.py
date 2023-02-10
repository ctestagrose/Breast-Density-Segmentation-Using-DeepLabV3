import torchvision.models.segmentation
import torch

class Model_Definition():
    """
    Model definition for segmentation algorithm

    All models will be loaded and saved as DataParallel models allowing for the use of mulitple GPUS or just 1.

    Args:
        pretrained: keep this set to True when training and evaluating. PyTorch seems to require it.
        input: default = 256. This is the input into the added layer to the algorithm
        output: default = 3. Set this to be the number of pixel classes you want to segment

    """
    def __init__(self, pretrained = True, input = 256, output = 3):
        self.pretrained = pretrained
        self.input = input
        self.output = output

    def segmentationModel(self, device='cpu'):
        model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=self.pretrained)
        model.classifier[4] = torch.nn.Conv2d(self.input, self.output, kernel_size=(1, 1), stride=(1, 1))
        model = torch.nn.DataParallel(model)
        return model
