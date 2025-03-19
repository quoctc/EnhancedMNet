#!/usr/bin/env python3

import argparse
import sys
import os
import csv

import torch
import torch.nn
import torch.optim
import torch.optim.lr_scheduler
import torch.utils.data
import torchvision.transforms
import torchvision.datasets

from utils.cross_entropy import LabelSmoothingCrossEntropy
from utils import datasets
from models.utils.datasets import StanfordDogs

from models.MobileNets.mobilenets_org_filters import build_mobilenet_v2, build_mobilenet_v3, build_mobilenet_v1

def get_args():
    """
    Parse the command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Training script for CIFAR and fine-grained datasets.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-r', '--data-dir', type=str, default='./dataset/', help='Dataset root path.')
    parser.add_argument('-d', '--dataset', type=str, choices=['CIFAR10', 'CIFAR100', 'dogs'], default='cifar10', help='Dataset name.')
    parser.add_argument('--save-path', type=str, default='.cl/saved-model/', help='Path to save the model.')
    parser.add_argument('--download', action='store_true', help='Download the specified dataset before running the training.')
    parser.add_argument('-a', '--architecture', type=str, default='v1', choices=['v1', 'v2', 'v3'], help='Model architecture name.')
    parser.add_argument('--mode', type=str, default='large', help='large or small MobileNetV3')
    parser.add_argument('-g', '--gpu-id', default=1, type=int, help='ID of the GPU to use. Set to -1 to use CPU.')
    parser.add_argument('-j', '--num_workers', default=4, type=int, help='Number of data loading workers.')
    parser.add_argument('-b', '--batch-size', default=64, type=int, help='Batch size.')        
    parser.add_argument('-e', '--epochs', default=200, type=int, help='Number of total epochs to run.')
    parser.add_argument('-l', '--learning-rate', default=0.1, type=float, help='Initial learning rate.')
    parser.add_argument('-s', '--schedule', nargs='+', default=[100, 150, 180], type=int, help='Learning rate schedule (epochs after which the learning rate should be dropped).')    
    parser.add_argument('-m', '--momentum', default=0.9, type=float, help='SGD momentum.')
    parser.add_argument('-w', '--weight-decay', default=1e-4, type=float, help='SGD weight decay.')
    parser.add_argument('-gb', '--gb-filters', default=8, type=int, help='# of gabor filters')
    parser.add_argument('--alpha', default=1.0, type=float, help='Network weighting coefficient for the regularization loss.')
    parser.add_argument('--filter', type=str, default='Gabor+Img', choices=['Sobel+Img', 'LoG+Img', 'Gabor+Img'], help='Select filter to use')
    parser.add_argument('--gb_filters', type=int, default=2, help='number of gabor filter to use')
    parser.add_argument('--kernel_size', type=int, default=5, help='Select filter kernel size to use')
    parser.add_argument('--group', type=int, default=1, help='Select group to use')

    return parser.parse_args()

def get_device(args):
    """
    Determine the device to use for the given arguments.
    """
    if args.gpu_id >= 0:
        return torch.device('cuda:{}'.format(args.gpu_id))
    else:
        return torch.device('cpu')

def get_input_size_and_classes(args):
    """
    Return the input size for the given dataset.
    """
    if args.dataset == 'imagenet':
        input_size = 224
        num_class = 1000
    elif args.dataset == 'dogs':
        input_size = 224
        num_class = 120
    elif args.dataset == 'tinyimagenet':
        input_size = 56
        num_class = 200
    elif args.dataset == 'cifar100':
        input_size = 32
        num_class = 100
    elif args.dataset == 'cifar10' or args.dataset == 'svhn':
        input_size = 32
        num_class = 10
    return input_size, num_class

def get_data_loader(args, train):
    """
    Return the data loader for the given arguments.
    """
    if args.dataset in ('cifar10', 'cifar100'):
        # select transforms based on train/val
        if train:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
            ])
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
    
        # cifar10 vs. cifar100
        if args.dataset == 'cifar10':
            dataset_class = torchvision.datasets.CIFAR10
        else:
            dataset_class = torchvision.datasets.CIFAR100
            
    elif args.dataset in ('dogs',):
        # select transforms based on train/val
        print('Using dogs dataset new version')
        if train:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(256, 256)), #OLD
                torchvision.transforms.RandomCrop(224), #OLD
                #torchvision.transforms.RandomResizedCrop(224), #NEW
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ColorJitter(0.4),
                torchvision.transforms.ToTensor()
            ])
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(256, 256)),
                torchvision.transforms.CenterCrop(224),
                #torchvision.transforms.Resize((224, 224)), #NEW
                torchvision.transforms.ToTensor()
            ])
    
        dataset_class = StanfordDogs
    
    elif args.dataset in ('ImageNet'):
        # select transforms based on train/val
        traindir = os.path.join(args.data_root, 'train')
        valdir = os.path.join(args.data_root, 'val')
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if train:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(256, 256)),
                torchvision.transforms.RandomCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ColorJitter(0.4),
                torchvision.transforms.ToTensor()
                ,normalize
            ])
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(256, 256)),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor()
                ,normalize
            ])

    else:
        raise NotImplementedError('Can\'t determine data loader for dataset \'{}\''.format(args.dataset))
    
    # trigger download only once
    if args.download:
        dataset_class(root=args.data_dir, train=train, download=True, transform=transform)

    # instantiate dataset class and create data loader from it
    dataset = dataset_class(root=args.data_dir, train=train, download=False, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True if train else False, num_workers=args.num_workers)

def calculate_accuracy(output, target):
    """
    Top-1 classification accuracy.
    """
    with torch.no_grad():
        batch_size = output.shape[0]
        prediction = torch.argmax(output, dim=1)
        return torch.sum(prediction == target).item() / batch_size

def run_epoch(train, data_loader, model, criterion, optimizer, n_epoch, args, device):
    """
    Run one epoch. If `train` is `True` perform training, otherwise validate.
    """
    if train:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)
    
    batch_count = len(data_loader)
    losses = []
    accs = []    
    for (n_batch, (images, target)) in enumerate(data_loader):
        images = images.to(device)
        target = target.to(device)

        output = model(images)
        loss = criterion(output, target)
        
        # record loss and measure accuracy
        loss_item = loss.item()
        losses.append(loss_item)
        acc = calculate_accuracy(output, target)
        accs.append(acc)

        # compute gradient and do SGD step
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        DEBUG = False
        if DEBUG and (n_batch % 10) == 0:
            print('[{}]  epoch {}/{},  batch {}/{},  loss_{}={:.5f},  acc_{}={:.2f}%'.format('train' if train else ' val ', n_epoch + 1, args.epochs, n_batch + 1, batch_count, "train" if train else "val", loss_item, "train" if train else "val", 100.0 * acc))
    
    return (sum(losses) / len(losses), sum(accs) / len(accs))
            
def main():
    import warnings
    warnings.filterwarnings('ignore')

    """
    Run the complete model training.
    """
    # Get the arguments
    args = get_args()
    
    args.dataset = args.dataset.lower()
    args.architecture = args.architecture.lower()

    # set device
    device = get_device(args)

    # initialize model
    input_size, num_class = get_input_size_and_classes(args)
    if args.architecture == 'v2':
        print('Using MobileNetV2')
        model = build_mobilenet_v2(num_classes=num_class, 
                                    width_multiplier=args.alpha, 
                                    cifar=(input_size < 100),
                                    use_filter=args.filter,
                                    device_id=args.gpu_id,
                                    num_gabor_filters=args.gb_filters)
    elif args.architecture == 'v1':
        print('Using MobileNetV1')
        model = build_mobilenet_v1(num_classes=num_class, 
                                    width_multiplier=args.alpha, 
                                    cifar=(input_size < 100),
                                    use_filter=args.filter,
                                    device_id=args.gpu_id,
                                    num_gabor_filters=args.gb_filters)
    elif args.architecture == 'v3':
        print('Using MobileNetV3')
        model = build_mobilenet_v3(num_classes=num_class,
                                    version=args.mode,
                                    width_multiplier=args.alpha, 
                                    cifar=(input_size < 100),
                                    use_lightweight_head=True,
                                    use_filter=args.filter,
                                    device_id=args.gpu_id,
                                    num_gabor_filters=args.gb_filters)

    print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))

    # prepare save path
    folder_name = [args.architecture, args.mode, args.dataset, 'wm'+str(args.alpha), 'bs'+str(args.batch_size), str(args.weight_decay), 'epochs'+str(args.epochs), 'filter'+str(args.filter)]
    folder_name = '-'.join(folder_name)
    args.save_path = os.path.join(args.save_path, folder_name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # get data loaders
    train_loader = get_data_loader(args=args, train=True)
    val_loader = get_data_loader(args=args, train=False)
    
    # define loss function and optimizer
    train_loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.schedule, gamma=0.1)
             
    # for each epoch...
    acc_val_max = None
    acc_val_argmax = None
    for n_epoch in range(args.epochs):
        current_learning_rate = optimizer.param_groups[0]['lr']
        print('Starting epoch {}/{},  learning_rate={}'.format(n_epoch + 1, args.epochs, current_learning_rate))
        
        # train
        (loss_train, acc_train) = run_epoch(train=True, data_loader=train_loader, model=model, criterion=train_loss_fn, optimizer=optimizer, n_epoch=n_epoch, args=args, device=device)
        # validate
        (loss_val, acc_val) = run_epoch(train=False, data_loader=val_loader, model=model, criterion=criterion, optimizer=None, n_epoch=n_epoch, args=args, device=device)
        if (acc_val_max is None) or (acc_val > acc_val_max):
            acc_val_max = acc_val
            acc_val_argmax = n_epoch
            # save the best model
            torch.save(model.state_dict(), os.path.join(args.save_path, 'best_model.pth'))
    
        # adjust learning rate
        scheduler.step()
        
        # print epoch summary
        line = 'Epoch {}/{} summary:  loss_train={:.5f},  acc_train={:.2f}%,  loss_val={:.2f},  acc_val={:.2f}% (best: {:.2f}% @ epoch {}), model: {}, dataset: {}, filter: {}'.format(n_epoch + 1, args.epochs, loss_train, 100.0 * acc_train, loss_val, 100.0 * acc_val, 100.0 * acc_val_max, acc_val_argmax + 1, args.architecture, args.dataset, args.filter)
        print('=' * len(line))
        print(line)
        print('=' * len(line))
        
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Stopped')
        sys.exit(0)
    #except Exception as e:
        #print('Error: {}'.format(e))
        #sys.exit(1)
