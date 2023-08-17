'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.optim import Adam
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from torchvision.datasets import SVHN
import torchvision.transforms as transforms
from utils.advDataloader import *
from utils.spectrum_utils import *
from utils.cluster_selec_utils import *
from Models.resnet50 import *
from Models.squeezenet import *
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os
import time
import datetime
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch SVHN Training') 
    parser.add_argument('device', help="GPU device", type=str, default='0')
    parser.add_argument('model', help="target model", choices=['resnet50'])
    parser.add_argument('sample_source', help="data sample source", choices=['cw','pgd','all'])
    parser.add_argument('selec_type', help="select type", choices=['orig','phase', 'highpass', 'residual','quaternion_fourier','deepgini','entropy','robot'])
    parser.add_argument('cluster_type', help="cluster type", choices=['kmeans','gmm'])
    parser.add_argument('sample', help="sample between (0,1)", type=float, default = 0.8)
    parser.add_argument('strategy', help="sample strategy", choices=['high_uc', 'low_uc', 'uniform'], default = 'uniform')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()
    
    # 设置超参
    #nb_epochs = 15
    #accuracy = []
    #eps = 0.01
    #eps_iter = 0.001
    #steps = 20
    #batch_size = 64
    # 指定GPU
    
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 20
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_sample = transforms.Compose([transforms.ToTensor()])
    
    save_sample_path = 'Sample_data/SVHN_{}_{}_{}_{}_{}_{}_samples.pkl'.format(args.model,args.sample_source,args.selec_type,args.cluster_type,str(args.sample),args.strategy)
    
    # 加载SVHN训练集
    trainset = SVHN(root='./Dataset', split='train',download=True, transform=transform)    
    svhn_adv = SVHNAdvDataset(pickle_file=save_sample_path, transform=ToTensor())

    # 将频谱特征选择的样本加入 SVHN 训练集
    combined_train_dataset = ConcatDataset([trainset, svhn_adv])
    trainloader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    # 加载SVHN测试集
    testset = SVHN(root='./Dataset', split='test',download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=2)
    
    # 加载SVHN对抗样本集
    test_adv_path = 'SVHN_adv/SVHN_{}_{}_testset.pkl'.format(args.model,args.sample_source)
    svhn_adv_test = SVHNAdvDataset(pickle_file=test_adv_path, transform=ToTensor())    
    adv_testloader = DataLoader(svhn_adv_test, batch_size=1000, shuffle=False, num_workers=2)
    #加载模型
    model = r50(resn50)
    model = model.to(device)

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    
        # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 定义训练函数
    def train(model, train_loader, criterion, optimizer):
        model.train()
        train_loss = 0.0
        train_correct = 0
        for images, labels in train_loader:
            images = images.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
        train_loss /= len(train_loader.dataset)
        train_accuracy = 100.0 * train_correct / len(train_loader.dataset)
        return train_loss, train_accuracy

    # 定义测试函数
    def test(model, test_loader, criterion):
        model.eval()
        test_loss = 0.0
        test_correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.cuda()
                labels = labels.cuda()
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                test_correct += (predicted == labels).sum().item()
        test_loss /= len(test_loader.dataset)
        test_accuracy = 100.0 * test_correct / len(test_loader.dataset)
        return test_loss, test_accuracy

    # 训练模型并保存最佳模型
    best_accuracy = 0.0
    train_losses,train_accuracies,test_accuracies,adv_accuracies = [],[],[],[]
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, trainloader, criterion, optimizer)
        test_loss, test_accuracy = test(model, testloader, criterion)
        adv_test_loss, adv_test_accuracy = test(model, adv_testloader, criterion)
        print('Epoch: [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.2f}%, Test Accuracy: {:.4f}, Adv Test Accuracy: {:.2f}%'.format(epoch+1, num_epochs, train_loss, train_accuracy, test_accuracy,adv_test_accuracy))
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        adv_accuracies.append(adv_test_accuracy)
        if adv_test_accuracy > best_accuracy:
            save_retrain_model = 'checkpoint/{}_{}_{}_{}_{}_{}_retrain.pth'.format(args.model,args.sample_source,args.selec_type,args.cluster_type,str(args.sample),args.strategy)
            torch.save(model.state_dict(), save_retrain_model)
            best_accuracy = adv_test_accuracy
    retrain_results = {
        'total_train_losses': train_losses,
        'total_train_acc': train_accuracies,
        'total_test_acc':test_accuracies,
        'total_adv_test_acc':adv_accuracies
    }
    save_result_path = 'checkpoint/{}_{}_{}_{}_{}_{}_retrain.npy'.format(args.model,args.sample_source,args.selec_type,args.cluster_type,str(args.sample),args.strategy)
    np.save(save_result_path,retrain_results)