import os
import random
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch

from preprocess import dataset
from preprocess.dataset import CustomDataset, MatDataset


def load_data_module(data_root, source, target, batch_size, train=True):
    print('>> Source: {}, Target: {}'.format(source, target))
    module, dataloader_source, dataloader_target, image_size, num_classes = None, None, None, 32, 10
    source_path, target_path = os.path.join(data_root, source), os.path.join(data_root, target)
    if source == 'mnist' and target == 'mnist_m':

        from model import svhn_module
        module = svhn_module

        image_size, num_classes = 32, 10
        dataloader_source = dataset.load_mnist(source_path, image_size, batch_size, train=train)
        dataloader_target = dataset.load_mnist_m(target_path, image_size, batch_size, train=train)
    elif (source == 'svhn' and target == 'mnist') or (source == 'mnist' and target == 'svhn'):

        from model import svhn_module
        module = svhn_module
        
        image_size, num_classes = 32, 10
        if source == 'svhn' and target == 'mnist':
            dataloader_source = dataset.load_svhn(source_path, image_size, batch_size, train=train)
            dataloader_target = dataset.load_mnist(target_path, image_size, batch_size, train=train)
        else:
            dataloader_source = dataset.load_mnist(source_path, image_size, batch_size, train=train)
            dataloader_target = dataset.load_svhn(target_path, image_size, batch_size, train=train)
    elif source.find('office31') != -1 and target.find('office31') != -1:
        from model import office31_module
        module = office31_module
        image_size, num_classes = 224, 31
        source, target =  source.split('_')[-1], target.split('_')[-1]  # use customDataset
        source_path, target_path = os.path.join(data_root, source), os.path.join(data_root, target)  # use customDataset
        dataloader_source = dataset.load_office31(source_path, image_size, batch_size, train=train)
        dataloader_target = dataset.load_office31(target_path, image_size, batch_size, train=train)
    elif source == 'digits' and target == 'svhn':
        digits_path = 'synth_train_32x32.mat' if train else 'synth_test_32x32.mat'

        from model import svhn_module
        module = svhn_module
        
        image_size, num_classes = 32, 10
        dataloader_source = dataset.load_digits(os.path.join(source_path, digits_path), image_size, batch_size, train=train)
        dataloader_target = dataset.load_svhn(target_path, image_size, batch_size, train=train)
    else:
        print('unexpected transfer task!')
    return module, dataloader_source, dataloader_target, image_size, num_classes


def load_data_only(data_root, source, target, domain, batch_size, train=True):
    dataset_name = source if domain == 'source' else target
    print('>> {}-{} loading Dataset: {}, {} only'.format(source, target, dataset_name, domain))
    module, dataloader, image_size, num_classes = None, None, 32, 10
    dataset_path = os.path.join(data_root, dataset_name)
    if source == 'mnist' and target == 'mnist_m':

        from model import svhn_module
        module = svhn_module

        image_size, num_classes = 32, 10
        if dataset_name == 'mnist':
            dataloader = dataset.load_mnist(dataset_path, image_size, batch_size, train=train)
        else:
            dataloader = dataset.load_mnist_m(dataset_path, image_size, batch_size, train=train)
    elif (source == 'svhn' and target == 'mnist') or (source == 'mnist' and target == 'svhn'):

        from model import svhn_module
        module = svhn_module
        
        image_size, num_classes = 32, 10
        if dataset_name == 'svhn':
            dataloader = dataset.load_svhn(dataset_path, image_size, batch_size, train=train)
        else:
            dataloader = dataset.load_mnist(dataset_path, image_size, batch_size, train=train)
    elif source.find('office31') != -1 and target.find('office31') != -1:
        from model import office31_module
        module = office31_module
        image_size, num_classes = 224, 31
        dataset_name =  dataset_name.split('_')[-1] # use customDataset
        dataset_path = os.path.join(data_root, dataset_name)  # use customDataset
        dataloader = dataset.load_office31(dataset_path, image_size, batch_size, train=train)
    elif source == 'digits' and target == 'svhn':
        digits_path = 'synth_train_32x32.mat' if train else 'synth_test_32x32.mat'

        from model import svhn_module
        module = svhn_module

        image_size, num_classes = 32, 10
        if dataset_name == 'digits':
            dataloader = dataset.load_digits(os.path.join(dataset_path, digits_path), image_size, batch_size, train=train)
        else:
            dataloader = dataset.load_svhn(dataset_path, image_size, batch_size, train=train)
    else:
        print('unexpected transfer task!')
    return module, dataloader, image_size, num_classes


def get_mean_std(device, image_root, dataset_name, image_size):
    image_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()])
    root = os.path.join(image_root, dataset_name)
    if dataset_name == 'mnist':
        dataset = datasets.MNIST(root=root, train=True, transform=image_transform, download=True)
    elif dataset_name == 'mnist_m':
        data_root = os.path.join(root, 'mnist_m_train')
        data_list = os.path.join(root, 'mnist_m_train_labels.txt')
        dataset = CustomDataset(data_root=data_root, data_list=data_list, transform=image_transform)
    elif  dataset_name == 'svhn':
        dataset = datasets.SVHN(root=root, split='train', transform=image_transform, download=True)
    elif  dataset_name == 'digits':
        digits_path = 'synth_train_32x32.mat'
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        dataset = MatDataset(os.path.join(root, digits_path), transform=image_transform)
    elif dataset_name.find('office31') != -1:
        data_list = dataset_name+'_list.txt'
        dataset = CustomDataset(data_root=image_root, data_list=data_list, transform=image_transform)
    else:
        print('unexpected dataset!')
        return
    if dataset_name == 'mnist_m':
        data = torch.zeros(len(dataset), 3, image_size, image_size)
        for idx in range(len(dataset)):
            data[idx] = dataset[idx][0]
    else:
        dataloader = DataLoader(dataset, batch_size=len(dataset), num_workers=8)
        data = iter(dataloader).next()[0]
        data = data.float().to(device)
    # cop for gray scale
    data = torch.FloatTensor(data.shape[0], 3, image_size, image_size).copy_(data)
    
    std, mean = torch.std_mean(data.permute(1, 0, 2, 3).reshape(3, -1), dim=1)

    std, mean = np.round(std.cpu().numpy(),decimals=4), np.round(mean.cpu().numpy(), decimals=4)
    print('{}: image_size={}, mean={}, std={}'.format(dataset_name, image_size, mean, std))


def generate_image_list(image_root, dataset_name, save_path):
    class_label = {}
    image_list_save_name = dataset_name + '_list.txt'
    image_list = []
    idx = 0
    for class_name in os.listdir(image_root):
        class_label[class_name] = idx
        class_root = os.path.join(image_root, class_name)
        for img_name in os.listdir(class_root):
            image_list.append('{} {}'.format(os.path.join(class_root, img_name), idx))
        idx += 1
    random.shuffle(image_list)  # shuffle sample
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, image_list_save_name), mode='w') as f:
        for image in image_list:
            f.write(image+"\n")

