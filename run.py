import copy
import os
import random
import warnings
warnings.filterwarnings('ignore')
import argparse

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from preprocess.data_utils import load_data_module, load_data_only
from model import models
import train


parser = argparse.ArgumentParser()

parser.add_argument('-d', '--domain', type=str, default='both')  # don't change!
parser.add_argument('-data_root', type=str, default='/root/dataset_DA')  # data root directory
parser.add_argument('-cuda', '--cuda', type=int, help='the gpu device id for training', default=1)
parser.add_argument('-use', '--use_cuda', type=bool, help='whether use gpu', default=True)
parser.add_argument('-s', '--source', type=str, help='the source dataset name', default='mnist')
parser.add_argument('-t', '--target', type=str, help='the target dataset name', default='mnist_m')
parser.add_argument('-lr', '--lr', type=float, help='learning rate', default=1e-3)
parser.add_argument('-epoch', '--num_epoch', help='the num of epoches for training', default=200)
parser.add_argument('-batch', '--batch_size', help='the size of one batch', type=int, default=128)
parser.add_argument('-gamma', '--gamma', type=float, help='gamma for gradient reversal loss', default=10)  # default 10
parser.add_argument('-modeldir', '--model_root', type=str, help='the directory for saving model', default='./models/final')
parser.add_argument('-logdir', '--log_root', type=str, help='the directory for saving tensorboard', default='./runs/final')  # tensorboard path
parser.add_argument('-lr_step', '--lr_scheduler_step_size', help='step for adjust learning rate', type=int, default=40)
parser.add_argument('-lr_gamma', '--lr_scheduler_gamma', help='gamma for adjust learning rate', type=float, default=0.8)

args = parser.parse_args()

train_mode = 'revgrad'
domain = args.domain
source, target = args.source, args.target

data_root = args.data_root
model_root = args.model_root
log_root = args.log_root

if not os.path.exists(data_root):
    assert 'Data directory does not exist !'
if not os.path.exists(model_root):
    os.makedirs(model_root)
if not os.path.exists(log_root):
    os.makedirs(log_root)

num_epoch = args.num_epoch
batch_size = args.batch_size
gamma = args.gamma
lr_scheduler_step_size = args.lr_scheduler_step_size
lr_scheduler_gamma = args.lr_scheduler_gamma
lr0 = args.lr
use_cuda = args.use_cuda
device = torch.device('cuda:{}'.format(args.cuda)) if torch.cuda.is_available() else torch.device('cpu')
if use_cuda:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

# load data and module
if domain == 'both':
    module, train_loader_source, train_loader_target, image_size, num_classes = load_data_module(data_root, source, target, batch_size, train=True)
else:
    module, train_dataloader, image_size, num_classes = load_data_only(data_root, source, target, domain, batch_size, train=True)

_, test_loader_target, _, _ = load_data_only(data_root, source, target, 'target', batch_size, train=False)

SEED = 2020
random.seed(SEED)
torch.manual_seed(SEED)

Model = models.OurModel(module.FeatureExtractor(), module.LabelClassifier(), module.LabelClassifier(), module.DomainDiscriminator(), module.PredictionDiscriminator())

# setup loss and optimizer
label_criterion = nn.NLLLoss()
domain_criterion = nn.NLLLoss()
prediction_criterion = nn.NLLLoss()
history = {'accs': [], 'current_model':None, 'best_model': None, 'best_acc': 0., 'best_epoch': None}

optimizer = optim.Adam(Model.parameters(), lr=lr0)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma)

if use_cuda:
    Model = Model.to(device)
    label_criterion = label_criterion.to(device)
    domain_criterion = domain_criterion.to(device)
    prediction_criterion = prediction_criterion.to(device)


save_path = os.path.join(source+'_'+target, train_mode)

log_dir = os.path.join(log_root, save_path)
model_path = os.path.join(model_root, save_path)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(model_path):
    os.makedirs(model_path)
summarywriter = SummaryWriter(log_dir=log_dir)


print('Training OurModel by Reverse Gradient...')
print('cuda=%d, base_lr=%.4f, batch_size=%d, gamma=%.1f, num_epoch=%d, lr_step_size=%d, lr_gamma=%.3f,' % (
    args.cuda, lr0, batch_size, gamma, num_epoch, lr_scheduler_step_size, lr_scheduler_gamma), 'discriminative loss: 2000-5000, LAMBDA=30, prediction by Fs&Ft loss_s * 5 & loss_t * 0.5')
for epoch in range(num_epoch):
    train.trainOurModel(Model, device, use_cuda, summarywriter, train_loader_source, train_loader_target, test_loader_target, 
            optimizer, scheduler, label_criterion, domain_criterion, prediction_criterion, history, target, 
            epoch, num_epoch, batch_size, num_classes, image_size, gamma, need_test=True)
            
for epoch in range(num_epoch):
    print('[epoch: {} / {}] accuracy of the {}: {:.4f}'.format(epoch, num_epoch, target, history['accs'][epoch]))

print('best accuracy of {} (target): {:.4f}(epoch {})'.format(target, history['best_acc'], history['best_epoch']))
torch.save(history['best_model'], os.path.join(model_path, 'epoch_' + str(history['best_epoch']) + '.pth'))
print('Training Finished')
