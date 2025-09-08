import abc
import torch.nn as nn

"""
Abstract Model that inherits from torch and ABC. Ensures that all deriving specific models at least define the listed functions.
It also provides the _create_backbone_and_head() function which creates the backbone/head model architecture of a client.
"""


class Model(nn.Module, abc.ABC):
    def __init__(self):
        super(Model, self).__init__()

    def _create_backbone_and_head(self, model, backbone_layers, num_classes):
        """Creates the backbone/head model architeture by taking assigning the backbone and
        head with the specified amount of layer blocks.

        Args:
            model (model): The model. Here Resnet18
            backbone_layers (str): How many layer blocks the backbone shall include. (full, minus1, minus2)
            num_classes (int): The number of output features

        Returns:
            backbone, head: Returns the created backbone and head
        """
        if backbone_layers == "full":
            backbone = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
            )

            head = nn.Sequential(
                model.avgpool,
                nn.Flatten(),
                nn.Linear(in_features=model.fc.in_features, out_features=num_classes, bias=True)
            )
        elif backbone_layers == "minus1":
            backbone = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
            )

            head = nn.Sequential(
                model.layer4,
                model.avgpool,
                nn.Flatten(),
                nn.Linear(in_features=model.fc.in_features, out_features=num_classes, bias=True)
            )
    
        elif backbone_layers == "minus2":
            backbone = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
            )

            head = nn.Sequential(
                model.layer3,
                model.layer4,
                model.avgpool,
                nn.Flatten(),
                nn.Linear(in_features=model.fc.in_features, out_features=num_classes, bias=True)
            )

        else:
            raise "Please specify a valid amount of backbone layers."

        return backbone, head

    @abc.abstractmethod
    def forward(self):
        pass

    @abc.abstractmethod
    def train_model(self):
        pass

    @abc.abstractmethod
    def validate_model(self):
        pass

    @abc.abstractmethod
    def test_model(self):
        pass
