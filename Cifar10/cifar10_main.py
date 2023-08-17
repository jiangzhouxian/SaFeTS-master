'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torchvision.transforms as transforms
from utils.advDataloader import *
from utils.spectrum_utils import *
from utils.cluster_selec_utils import *

from Models_cifar10.backbones.ResNet import ResNet18
from Models_cifar10.backbones.VGG import VGG16
from Models_cifar10.backbones.DenseNet import DenseNet121
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os
import time
import datetime
import argparse



# GPU ruinning time
def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

# Training
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    train_acc = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # inputs:[b,3,32,32], targets:[b]
        # train_outputs:[b,10]
        inputs, targets = inputs.to(device), targets.to(device)
        # print(inputs.shape)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        # Computing loss
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # Computing accuracy
        train_acc = correct / total
        
    # Computing accuracy in each epoach
    total_train_acc.append(train_acc)
    total_train_losses.append(loss.item())
    print('[INFO] Epoch-{}: Train: Loss:{:.4f}, Accuracy:{:.4f}'.format(epoch + 1,loss.item(),train_acc))
    
# Testing
def test_adv(epoch, ckpt):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_acc = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(adv_testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        test_acc = correct / total
        print(
            '[INFO] Epoch-{}-Adv_Test Accurancy: {:.3f}'.format(epoch + 1, test_acc), '\n')

    total_adv_test_acc.append(test_acc)

    # Saving weights.
    acc = 100 * correct / total
    if acc > best_acc:
        #print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, ckpt)
        best_acc = acc

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_acc = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        test_acc = correct / total
        print(
            '[INFO] Epoch-{}-Original_Test Accurancy: {:.3f}'.format(epoch + 1, test_acc), '\n')

    total_test_acc.append(test_acc)


if __name__ == '__main__':
    # Setting hyper-parameters
    epochs = 100
    batch_size =128
    T_max = 100
    lr = 0.1
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training') 
    parser.add_argument('device', help="GPU device", type=str, default='0')
    parser.add_argument('model', help="target model", choices=['ResNet18', 'VGG16', 'DenseNet121'])
    parser.add_argument('sample_source', help="data sample source", choices=['PGD','CW'])
    parser.add_argument('selec_type', help="select type", choices=['phase', 'highpass', 'residual','quaternion_fourier','deepgini','robot'])
    parser.add_argument('cluster_type', help="cluster type", choices=['kmeans','gmm'])
    parser.add_argument('sample', help="sample between (0,1)", type=float, default = 0.8)
    parser.add_argument('strategy', help="sample strategy", choices=['high_uc', 'low_uc', 'uniform'], default = 'uniform')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    #parser.add_argument('--checkpoint', type=str, default='checkpoint/VGG16-CIFAR10.pth')
    args = parser.parse_args()

    # 设置相关参数
    # 指定GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    total_train_losses,total_train_acc,total_test_acc,total_adv_test_acc = [],[],[],[]
    
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_sample = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    
    save_sample_path = 'Sample_data_20/cifar10_{}_{}_{}_{}_{}_{}_samples.pkl'.format(args.model,args.sample_source,args.selec_type,args.cluster_type,str(args.sample),args.strategy)
    
    # Loading the CIFAR10 training set
    trainset = torchvision.datasets.CIFAR10(
        root='Models_cifar10/Dataset', train=True, download=True, transform=transform_train)
    
    with open(save_sample_path, 'rb') as f:
        adv_sample_data = pickle.load(f)

    adv_sample_samples = adv_sample_data['data']
    adv_sample_labels = adv_sample_data['labels']
    adv_sample_dataset = CIFAR10AdvDataset(adv_sample_samples, adv_sample_labels, transform=transform_sample)
    
    # Adding samples selected for spectral characterization to the CIFAR-10 training set
    combined_train_dataset = ConcatDataset([trainset, adv_sample_dataset])
    trainloader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    # Loading the CIFAR10 test set
    testset = torchvision.datasets.CIFAR10(root='Models_cifar10/Dataset', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Loading the CIFAR10 Adversarial Sample Set
    test_adv_path = 'Cifar10_adv/cifar10_{}_{}_testset.pkl'.format(args.model,args.sample_source)
    adv_testloader = Cifar10advLoader(test_adv_path,train = 'test', batch_size=batch_size, shuffle=False, num_workers=2)

    #print('==> Building model..')
    if args.model == 'ResNet18':
        model = ResNet18().to(device)
    elif args.model == 'VGG16':
        model = VGG16().to(device)
    elif args.model == 'DenseNet121':
        model = DenseNet121().to(device)

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    # Setting up the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # Cosine Annealing Ordered Adjustment of Learning Rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    total_train_losses,total_train_acc,total_test_acc,total_adv_test_acc = [],[],[],[]
    # Recording of training time
    tic = time_sync()

    # Start training
    for epoch in range(epochs):
        train(epoch)
        save_model = 'checkpoint_20/{}_{}_{}_{}_{}_{}_retrain.pth'.format(args.model,args.sample_source,args.selec_type,args.cluster_type,str(args.sample),args.strategy)
        test_adv(epoch, save_model)
        # Dynamic adjustment of learning rates
        scheduler.step()
        test(epoch)

    # data visualization
    plt.figure()
    plt.plot(range(epochs), total_train_acc, label='Train Accurancy')
    plt.plot(range(epochs), total_test_acc, label='Test Accurancy')
    plt.xlabel('Epoch')
    plt.ylabel('Accurancy')
    plt.title('{}_{}_{}_{}_{}_{}_Accurancy'.format(args.model,args.sample_source,args.selec_type,args.cluster_type,str(args.sample),args.strategy))
    
    plt.legend()
    plt.savefig('checkpoint_20/{}_{}_{}_{}_{}_{}_Accurancy.jpg'.format(args.model,args.sample_source,args.selec_type,args.cluster_type,str(args.sample),args.strategy))  # 自动保存plot出来的图片
    plt.show()

    # Output best_acc
    print(f'Best Acc: {best_acc}')
    toc = time_sync()
    # Calculate this running time
    t = (toc - tic) / 3600
    retrain_results = {
        'total_train_losses': total_train_losses,
        'total_train_acc': total_train_acc,
        'total_test_acc':total_test_acc,
        'total_adv_test_acc':total_adv_test_acc
    }
    save_result_path = 'checkpoint_20/{}_{}_{}_{}_{}_{}_retrain.npy'.format(args.model,args.sample_source,args.selec_type,args.cluster_type,str(args.sample),args.strategy)
    np.save(save_result_path,retrain_results)