from PIL import Image
from scipy.io import loadmat

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import os


def default_loader(path):
    return Image.open(path).convert('RGB')


def array2Img(array):
    return Image.fromarray(array, mode='RGB').convert('RGB')


class CustomDataset(Dataset):
    def __init__(self, data_root, data_list, transform=None, loader=default_loader):
        super(CustomDataset, self).__init__()
        with open(data_list, mode='r', encoding='utf8') as f:
            imgs = []
            for line in f:
                words = line.strip().split()
                imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.data_root = data_root
        self.transform = transform
        self.loader = default_loader

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        f, label = self.imgs[idx]
        img = self.loader(os.path.join(self.data_root, f))
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class MatDataset(Dataset):
    def __init__(self, data_root, transform=None):
        super(MatDataset, self).__init__()
        data = loadmat(data_root)
        self.X = data['X']
        if self.X.shape[-1] != 3 and self.X.shape[-1] != 1:  # (32, 32, 3, )  -> (, 32, 32, 3)
            self.X = self.X.transpose(3, 0, 1, 2)
        self.y = data['y'].flatten()
        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        img, label = array2Img(self.X[idx, :, :, :]), int(self.y[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def load_mnist(image_root, image_size, batch_size, train=True):
    if image_size == 28:
        mean, std = (0.1307, ), (0.3081, )
    elif image_size == 32:
        mean, std = (0.1309, ), (0.2893, )
    else:
        mean, std = (0.5, ), (0.5, )
    image_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])
    mnist = datasets.MNIST(root=image_root, train=train, transform=image_transform, download=True)
    batch_size = batch_size if batch_size != 'all' else len(mnist)
    dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=train, drop_last=train, num_workers=8)
    return dataloader


def load_mnist_m(image_root, image_size, batch_size, train=True):
    if image_size == 28:
        mean, std = (0.4582, 0.4623, 0.4085), (0.2386, 0.2239, 0.2444)
    elif image_size == 32:
        mean, std = (0.4579, 0.4621, 0.4082), (0.2519, 0.2368, 0.2587)
    else:
        mean, std = (0.5, ), (0.5, )
    image_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    if train:
        data_root = os.path.join(image_root, 'mnist_m_train')
        data_list = os.path.join(image_root, 'mnist_m_train_labels.txt')
    else:
        data_root = os.path.join(image_root, 'mnist_m_test')
        data_list = os.path.join(image_root, 'mnist_m_test_labels.txt')
    mnist_m = CustomDataset(data_root=data_root, data_list=data_list, transform=image_transform)
    dataloader = DataLoader(mnist_m, batch_size=batch_size, shuffle=train, drop_last=train, num_workers=8)
    return dataloader


def load_svhn(image_root, image_size, batch_size, train=True):
    image_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4377, 0.4438, 0.4728), std=(0.198, 0.201, 0.197))
    ])
    split = 'train' if train else 'test'
    def process_label(label):  # change label 10 to label 0
        return 0 if label == 10 else label
    svhn = datasets.SVHN(root=image_root, split=split, transform=image_transform, target_transform=process_label, download=True)
    dataloader = DataLoader(svhn, batch_size=batch_size, shuffle=train, drop_last=train, num_workers=8)
    return dataloader


def load_digits(image_root, image_size, batch_size, train=True):
        # TODO: DIGITS dataset mean and std; load style
    image_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    digits = MatDataset(image_root, transform=image_transform)
    dataloader = DataLoader(digits, batch_size=batch_size, shuffle=train, drop_last=train, num_workers=8)
    return dataloader


def load_office31(image_root, image_size, batch_size, train=True):
    domain = image_root.split('/')[-1]
    if domain == 'amazon':
        mean, std = (0.7924, 0.7862, 0.7842), (0.315, 0.3175, 0.3194)
    elif domain == 'dslr':
        mean, std = (0.4709, 0.4487, 0.4064), (0.204, 0.192, 0.1996)
    else:  # webcam 
        mean, std = (0.612, 0.6188, 0.6173), (0.2506, 0.2555, 0.2577)
    image_transform = transforms.Compose([
        transforms.Resize(image_size),
        # transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # office31 = datasets.ImageFolder(root=image_root, transform=image_transform, loader=default_loader)  # image_root是数据集地址
    office31 = CustomDataset(data_root='', data_list=image_root+'_list.txt', transform=image_transform)  # image_root是dataset文件夹下的list.txt
    batch_size = batch_size if batch_size != 'all' else len(office31)
    dataloader = DataLoader(office31, batch_size=batch_size, shuffle=train, drop_last=train, num_workers=8)
    return dataloader
