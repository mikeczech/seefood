import torch
import torch.nn as nn

torch.hub.list("rwightman/gen-efficientnet-pytorch")

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class EfficientNetFeatureExtractor(nn.Module):
    """ Extracts features using EfficientNet """

    def __init__(self, efficientnet_model_name):
        super(FeatureExtractor, self).__init__()
        self.model = torch.hub.load(
            "rwightman/gen-efficientnet-pytorch", efficientnet_model_name, pretrained=True
        )
        set_parameter_requires_grad(self.model, True)

    def forward(self, x):
        x = self.model.features(x)
        x = x.mean([2, 3])
        return x
