import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from model.model import Spoofing, Classifier
from dataset import faceDataset
from loss import TripletLoss
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

if __name__ == '__main__':
    
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    test_data = np.load('siw_test_data.npy')
    test_dataset = faceDataset('siw_test', './siw_test', data=test_data, sequence=True)

    batch_size = 3
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = Spoofing().cuda()
    model.load_state_dict(torch.load('spoofer.pth'))
    classifier = Classifier().cuda()
    classifier.load_state_dict(torch.load('classifier.pth'))
    model.eval()
    classifier.eval()
    cue_threshold = 0.01
    with torch.no_grad():
        output = np.zeros(1)
        for i, data in enumerate(test_loader):
            batch = data.shape[0]
            data = data.view(-1, 3, test_dataset.new_size, test_dataset.new_size)
            # output_cue, score = model(data.cuda())
            output_cue = model(data.cuda())
            score = classifier(model.encoder(output_cue[-1].detach() + data.cuda())[-1])

            # output_cue = output_cue[-1].cpu().numpy()
            # output_cue = np.abs(output_cue).sum(axis=(1, 2, 3)) / (3 * (test_dataset.new_size**2))
            # output_cue[output_cue > cue_threshold] = 1
            # output_cue[output_cue <= cue_threshold] = 0
            # output_cue = output_cue.reshape(batch, 10)
            # mean = np.mean(output_cue, axis=1).reshape(-1, 1)
            # for j in range(batch):
            #     if mean[j] <= (4/10):
            #         output_cue[j] = -output_cue[j] + 1
            # score = torch.sigmoid(score)
            # non_zero = (output_cue != 0)
            # score = score.cpu().numpy().reshape(batch, 10)
            # output = np.concatenate((output, np.sum(output_cue * score, axis=1) / non_zero.sum(axis=1)))

            score = torch.sigmoid(score)
            score = score.cpu().numpy().reshape(batch, 10)
            output = np.concatenate((output, np.mean(score, axis=1)))
    output = output[1:]
    with open('Siw_output.csv', 'w') as f:
        titles = 'video_id,label'
        f.write(titles + '\n')
        for i, y in enumerate(output):
            pred = '{},'.format(i)
            pred += str(y)
            f.write(pred + '\n')
