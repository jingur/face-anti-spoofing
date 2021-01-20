import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from model.model import Spoofing
from model.model import Classifier
from dataset import faceDataset
from loss import TripletLoss
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from model.network import ResnetUnetHybrid
  
if __name__ == '__main__':
    
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Read data from numpy file
    train_data = np.load('train_data.npy')
    train_label = np.load('train_label.npy')
    train_dataset = faceDataset('oulu_train', './oulu/train', data=train_data, label=train_label)
    print(train_data.shape)
    val_data = np.load('val_data.npy')
    val_label = np.load('val_label.npy')
    val_dataset = faceDataset('oulu_train', './oulu/val', data=val_data, label=val_label, sequence=True)
    print(val_data.shape)
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = Spoofing(in_channels=4).cuda()
    classifier = Classifier(pretrained=True).cuda()
    depth_generator = ResnetUnetHybrid.load_pretrained(device = device)

    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer_classifier = torch.optim.Adam(classifier.parameters(), lr=lr)
    
    decay_rate = 0.95
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=decay_rate)
    lr_scheduler_classifier = torch.optim.lr_scheduler.StepLR(optimizer_classifier, 1, gamma=decay_rate)
    
    triplet_loss = TripletLoss()
    bce_loss_cuda = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4]).cuda())
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4]))
    MSE_loss = nn.MSELoss()
    normal_bce = nn.BCEWithLogitsLoss()
    
    num_epoch = 40
    best_val_auc = 0
    best_loss = 9999
    
    print('Start train')
    depth_generator.eval()
    for epoch in range(num_epoch):
        model.train()
        classifier.train()
        train_loss = 0

        for i, (data, label) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_classifier.zero_grad()
            # data : [batch, 3, size, size]
            # label : [batch, 1]
            # cue_output, score = model(data.cuda())
            depth_map = depth_generator(data.cuda() * label.view(-1, 1, 1, 1).cuda())
            depth_map = F.interpolate(depth_map, size=(224, 224), mode='bicubic', align_corners=False)
            # RGBD_data = torch.cat((data.cuda(), depth_map), 1)
            cue_output, score = model(data.cuda())
            
            # binary loss 
            classifier_loss = bce_loss_cuda(score, label.cuda())
            # classifier_loss = cross_loss(score, label.view(-1).long().cuda())

            # cue_map loss (L2_loss)
            # cup_output[-1] shape: [batch, 1, size, size]

            # cue_map = cue_output[-1] * label.view(-1, 1, 1, 1).cuda()
            # live_num = torch.sum(label)
            # reg_loss = torch.sum(torch.abs(cue_map)) / (live_num + 1e-9)
            reg_loss = MSE_loss(cue_output[-1], depth_map)

            # Triplet loss
            trip_loss = 0
            for feat in cue_output[:-1]:
                feat = F.adaptive_avg_pool2d(feat, [1, 1]).view(cue_output[-1].shape[0], -1)
                trip_loss += triplet_loss(feat, label.view(-1, 1).cuda())

            total_loss = 1*classifier_loss + 2*reg_loss + 1*trip_loss
            total_loss.backward()
            optimizer.step()
            optimizer_classifier.step()
            train_loss += total_loss.item()
            print('\r[{}/{}] {}/{} train_loss: {:9.5f}'.format( \
                epoch+1, num_epoch, i+1, len(train_loader), train_loss/(i+1)), end='')

        torch.cuda.empty_cache()
        lr_scheduler.step()
        lr_scheduler_classifier.step()

        # Validation
        cue_threshold = 0.01
        model.eval()
        classifier.eval()
        with torch.no_grad():
            output = np.zeros(1)
            valid_loss = 0
            for i, (data, label) in enumerate(val_loader):
                batch = data.shape[0]
                data = data.view(-1, 3, val_dataset.new_size, val_dataset.new_size)
                # depth_map = depth_generator(data.cuda())
                # depth_map = F.interpolate(depth_map, size=(224, 224), mode='bicubic', align_corners=False)
                # RGBD_data = torch.cat((data.cuda(), depth_map), 1)
                output_cue, score = model(data.cuda())
                # score = classifier(model.encoder(output_cue[-1].detach() + data.cuda())[-1])

                # for cross entropy
                # score = torch.softmax(score, dim=1)
                # max_score = torch.max(score, dim=1)[0]
                # index = torch.max(score, dim=1)[1]
                # max_score[index == 0] = -max_score[index == 0] + 1
                # score = max_score.cpu().numpy().reshape(batch, 11)
                # output = np.concatenate((output, np.mean(score, axis=1)))

                score = torch.sigmoid(score)

                # non_zero = (output_cue != 0)
                # score = score.cpu().numpy().reshape(batch, 11)
                # output = np.concatenate((output, np.sum(output_cue * score, axis=1) / non_zero.sum(axis=1)))
                score = score.cpu().numpy().reshape(batch, 11)
                output = np.concatenate((output, np.mean(score, axis=1)))
            output = output[1:]
            groundtruth = np.mean(val_label, axis=1).astype(np.uint8)
            valid_loss = bce_loss(torch.Tensor(output), torch.Tensor(groundtruth)).item()
            valid_auc = roc_auc_score(groundtruth, output)
            print('\nAUC socre: {}  Classifier loss: {}'.format(valid_auc, valid_loss))
            if valid_loss < best_loss and epoch > 10:
                best_loss = valid_loss
                torch.save(model.state_dict(), 'spoofer.pth')
                print('Save model !')
                # torch.save(classifier.state_dict(), 'classifier.pth')
        torch.cuda.empty_cache()


