import torch.nn as nn
import torch.nn.functional as f
import pytorch_lightning as pl
import torchvision


class CNN_lightning3(pl.LightningModule):
    def __init__(self, num_classes=10, name=None):
        super(CNN_lightning3, self).__init__()
        self.name = name
        self.num_classes = num_classes
        self.dropout = nn.Dropout(0.1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding='same')
        self.act0 = nn.ReLU()
        self.MP1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same')
        self.act1 = nn.ReLU()
        self.MP2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten(1, -1)
        self.dense1 = nn.Linear(in_features=64 * 7 ** 2, out_features=128)
        self.actd1 = nn.ReLU()
        self.dense2 = nn.Linear(in_features=128, out_features=64)
        self.actd2 = nn.ReLU()
        self.dense2_1 = nn.Linear(in_features=64, out_features=32)
        self.act2 = nn.ReLU()
        self.dense3 = nn.Linear(in_features=32, out_features=self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act0(x)
        x = self.MP1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.act1(x)
        x = self.MP2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.actd1(x)
        x = self.dense2(x)
        x = self.actd2(x)
        x = self.dense2_1(x)
        x = self.act2(x)
        x = self.dense3(x)
        return f.softmax(x, dim=1)

class ResNet18(pl.LightningModule):
    def __init__(self, num_classes=10, name='ResNet'):
        super().__init__()
        self.name = name
        self.model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model.maxpool = nn.Identity()
    def forward(self, x):
        return self.model(x)

## The following is ResNet18 with pretrained weights for ImageNet. This is used for the Imagenette experiments.
class ResNet18_pl(pl.LightningModule):
    def __init__(self, num_classes=1000, name='ResNet', type='Imagenette'):
        super().__init__()
        self.name = name
        self.model = torchvision.models.resnet18(weights='IMAGENET1K_V1', num_classes=num_classes)
        if type == 'Imagenette':
            self.select_classes = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
    def forward(self, x):
        predictions = self.model(x)
        projected_predictions = predictions[:, self.select_classes]
        return projected_predictions