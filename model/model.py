from torch import nn
from model.decoder import Decoder
from model.resnet import ResNet18_Classifier, ResNet18_Encoder
from torchvision import models
import torch


class Spoofing(nn.Module):
    def __init__(self, in_channels=3, pretrained=True, dropout=0.5):
        super(Spoofing, self).__init__()
        # self.encoder = ResNet18_Encoder(out_indices=[2, 3, 4])
        # self.decoder = Decoder(in_channels=(64, 128, 256, 512), out_channels=(512, 256, 128, 64, 3))
        self.encoder = ResNet18_Encoder(pretrained=pretrained, in_channels=3)
        self.decoder = Decoder()
        self.classifier = ResNet18_Classifier(pretrained=pretrained, dropout=dropout, in_channels=in_channels)

    def forward(self, x):
        outs = self.encoder(x)
        outs = self.decoder(outs)
        # return outs
        # s = x[:, :-1, :, :] + outs[-1]
        # label_score = self.classifier(torch.cat((s, x[:, -1, :, :].unsqueeze(1)), dim=1))
        label_score = self.classifier(torch.cat((x, outs[-1]), dim=1))
        return outs, label_score

class Classifier(nn.Module):
    def __init__(self, pretrained=True, dropout=0.5):
        super(Classifier, self).__init__()
        # self.resnet18 = models.resnet18(pretrained=True)
        # self.resnet18.fc = nn.Linear(
        #     in_features=self.resnet18.fc.in_features, out_features=1
        # )
        self.linear = nn.Sequential(
            nn.Linear(512*7*7, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = self.resnet18.avgpool(x)
        # x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear(x)
        # x = self.resnet18.fc(x)
        return x
