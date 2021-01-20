import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from model.model import Spoofing, Classifier
from dataset import faceDataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model.network import ResnetUnetHybrid

if __name__ == '__main__':
    
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    test_data = np.load('test_data.npy')
    test_dataset = faceDataset('oulu_test', './oulu/test', data=test_data, sequence=True)

    batch_size = 2
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = Spoofing(in_channels=4).cuda()
    model.load_state_dict(torch.load('spoofer.pth'))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    depth_generator = ResnetUnetHybrid.load_pretrained(device = device)

    model.eval()
    depth_generator.eval()
    with torch.no_grad():
        output = np.zeros(1)
        for i, data in enumerate(test_loader):
            batch = data.shape[0]
            data = data.view(-1, 3, test_dataset.new_size, test_dataset.new_size)
            # depth_map = depth_generator(data.cuda())
            # depth_map = F.interpolate(depth_map, size=(224, 224), mode='bicubic', align_corners=False)
            # RGBD_data = torch.cat((data.cuda(), depth_map), 1)
            _, score = model(data.cuda())

            score = torch.sigmoid(score)
            score = score.cpu().numpy().reshape(batch, 11)

            output = np.concatenate((output, np.mean(score, axis=1)))

    output = output[1:]
    with open('oulu_output.csv', 'w') as f:
        titles = 'video_id,label'
        f.write(titles + '\n')
        for i, y in enumerate(output):
            pred = '{},'.format(i)
            pred += str(y)
            f.write(pred + '\n')
