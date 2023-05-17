# This is a script for training NN architectures with TD-optimization.

import os
from pathlib import Path
from tqdm import tqdm
import argparse
import torch
import torch.nn.functional as f

import utils.utils_optimization as uo
import utils.utils_architectures as ua
from utils.utils_optimization import TD_optimizer_lightning

import numpy as np

import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, SVHN
from pytorch_metric_learning.samplers import MPerClassSampler
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
import pytorch_lightning as pl

import matplotlib.pyplot as plt
plt.switch_backend('Agg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e',
        '--epochs',
        required=False,
        type=int,
        default=50,
        help='Number of epochs to train the model.'
    )
    parser.add_argument(
        '-bs',
        '--batch_size',
        required=False,
        type=int,
        default=128,
        help='Batch size.'
    )

    parser.add_argument(
        '-lr',
        '--learning_rate',
        type=float,
        required=False,
        default=0.001,
        help='Learning rate for usual training.',
    )

    parser.add_argument(
        '-arc',
        '--architecture',
        type=str,
        required=False,
        default='CNN_lightning3',
        help='Name of the NN architecture, currently supported: "CNN_lightning3", "ResNet18".',
    )
    parser.add_argument(
        '-idn',
        '--in_distribution_data',
        type=str,
        required=False,
        default='MNIST',
        help='Name of dataset to be used for training, currently supported: "MNIST", "FMNIST", "CIFAR10", "SVHN".',
    )

    args = parser.parse_args()

    data_dir = os.path.join(os.getcwd(), 'data')
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    curr_dir = os.getcwd()

    data_dir_mnist = os.path.join(data_dir, 'mnist')
    data_dir_fmnist = os.path.join(data_dir, 'fmnist')
    data_dir_cifar10 = os.path.join(data_dir, 'cifar10')
    data_dir_svhn = os.path.join(data_dir, 'svhn')

    use_cuda = True

    epochs = args.epochs
    lr = args.learning_rate

    batch_size = args.batch_size
    data_name = args.in_distribution_data

    arc = args.architecture

    if data_name in ['CIFAR10', 'SVHN']:
        arc = 'ResNet18'
        lr = 0.1
    if data_name == 'SVHN':
        epochs = 22

    nlabel = 10

    data_normalization = False

    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    print('Num threads: ', torch.get_num_threads(), '\nNum CPUs: ', os.cpu_count())

    ## Prepare architecture:
    if arc == 'CNN_lightning3':
        net = ua.CNN_lightning3(num_classes=nlabel)
    elif arc == 'ResNet18':
         net = ua.ResNet18(num_classes=nlabel)
    else:
        RuntimeError('Chosen architecture not supported!')

    ## Save dir for weights of trained models:
    model_save_dir = os.path.join(curr_dir, 'models', data_name)
    Path(os.path.join(model_save_dir)).mkdir(parents=True, exist_ok=True)

    # preparation of training data and test data:
    data_name_dict = {'MNIST': MNIST, 'FMNIST': FashionMNIST, 'CIFAR10': CIFAR10, 'SVHN': SVHN}
    data_dirs_dict = {'MNIST': data_dir_mnist, 'FMNIST': data_dir_fmnist, 'CIFAR10': data_dir_cifar10, 'SVHN': data_dir_svhn}

    transformation = transforms.Compose([transforms.ToTensor()])
    if data_name in ['CIFAR10', 'SVHN']:
        train_transforms = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    else:
        train_transforms = transformation

    if data_name != 'SVHN':
        train_data = data_name_dict[data_name](data_dirs_dict[data_name], train=True, download=True, transform=train_transforms)
        test_data = data_name_dict[data_name](data_dirs_dict[data_name], train=False, download=True, transform=transformation)
    else:
        train_data = data_name_dict[data_name](data_dirs_dict[data_name], split='train', download=True, transform=train_transforms)
        test_data = data_name_dict[data_name](data_dirs_dict[data_name], split='test', download=True, transform=transformation)


    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)
    num_batches_train = len(train_loader)

    if data_name != 'SVHN':
        num_classes = len(train_data.classes)
        train_data_targets = train_data.targets
    else:
        num_classes = 10
        train_data_targets = train_data.labels

    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=False,
                             pin_memory=True)

    ## Setting up and training the model:
    history = {
        'train_loss': [],
        'test_loss': [],
        'test_acc': [],
    }

    ## Setting up checkpoint callback:
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_save_dir,
        filename="weights-{epoch}",#-{epoch:02d}",
        save_weights_only=True,
        save_top_k=-1,
    )

    ## Defining optimizers and loss:
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer1 = optim.Adam(params=net.parameters(), lr=lr)
    model = TD_optimizer_lightning(
        net,
        learning_rate=lr,
        loss_criterion=criterion,
        ).to(device)

    # model.to(device)
    print(f'\nModel is on device: {model.device}\n')
    num_devices = None
    accelerator = 'auto'
    if device == torch.device('cpu'):
        num_devices = 1
        accelerator = None
    trainer = pl.Trainer(accelerator=accelerator,
                         devices=num_devices,
                         #gpus=-1,
                         max_epochs=epochs,
                         callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=200)])
    trainer.fit(model, train_loader, test_loader)
    trainer.test(model, test_loader)
    ## clear GPU memory:
    torch.cuda.empty_cache()