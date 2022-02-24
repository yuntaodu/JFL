import os
import random
import copy
import warnings
warnings.filterwarnings('ignore')
import argparse

import torch

from preprocess.data_utils import load_data_only
from model import models


def testOurs(test_mode, model, device, use_cuda, summarywriter, dataset_name, history, dataloader, epoch):
    lambda_g, lambda_f = 0, 0
    # evaluate
    model.eval()
    if use_cuda:
        model = model.to(device)
        
    n_correct = 0
    n_total = 0
    for (imgs, labels) in dataloader:
        batch_size, image_size = imgs.shape[0], imgs.shape[-1]
        imgs = torch.FloatTensor(batch_size, 3, image_size, image_size).copy_(imgs)
        if use_cuda:
            imgs = imgs.to(device)
            labels = labels.to(device)
        
        label_output_Fs, label_output_Ft, _, _ = model(imgs, imgs, lambda_g, lambda_f)

        pred = torch.max(label_output_Fs, label_output_Ft).max(1, keepdim=True)[1]

        n_correct += pred.eq(labels.view_as(pred)).cpu().sum()
        n_total += batch_size

    acc = n_correct.data.numpy() * 1.0 / n_total
    
    # write test result to tensorboard
    summarywriter.add_scalar('test_acc', acc, epoch)

    history['accs'].append(acc)
    if acc >= history['best_acc']:
        history['best_model'] = copy.deepcopy(history['current_model'])
        history['best_acc'] = acc
        history['best_epoch'] = epoch
    print('epoch: %d, accuracy of the %s: %.4f' % (epoch, dataset_name, acc))
