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
from utils.baselines import *
from Models_cifar10.backbones.ResNet import ResNet18
from Models_cifar10.backbones.VGG import VGG16
from Models_cifar10.backbones.DenseNet import DenseNet121
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
import datetime
import os
import argparse

parser = argparse.ArgumentParser(description='Main function for difference-inducing input selection for retrain on CIFAR10 dataset')
parser.add_argument('model', help="target model", choices=['ResNet18', 'VGG16', 'DenseNet121'])
parser.add_argument('sample_source', help="data sample source", choices=['CW','PGD','all'])
parser.add_argument('selec_type', help="select type", choices=['orig','phase', 'highpass', 'residual','quaternion_fourier','deepgini','entropy','robot'])
parser.add_argument('cluster_type', help="cluster type", choices=['kmeans','dbscan','gmm'])
#parser.add_argument('sample', help="sample between (0,1)", type=float, default = 0.8)
parser.add_argument('strategy', help="sample strategy", choices=['high_uc', 'low_uc', 'uniform'], default = 'uniform')
args = parser.parse_args()
#采样的样本来源读取
Sample_Path = 'Cifar10_adv_20/{}_{}_adv_train.pkl'.format(args.model,args.sample_source)
with open(Sample_Path, 'rb') as f:
    adv_data = pickle.load(f)

advloader = Cifar10advLoader(Sample_Path,train='sample',batch_size=50000,shuffle=False,num_workers=0)
# 从advloader中获取对抗样本
adv_samples = []
adv_labels = []
for inputs, labels in advloader:
    adv_samples.extend(inputs.numpy())
    adv_labels.extend(labels.numpy())

adv_samples = np.array(adv_samples)
adv_labels = np.array(adv_labels)
adv_dataset = CIFAR10AdvDataset(adv_samples, adv_labels, transform='test')
#print(adv_samples.shape)

#计时开始
start_time = datetime.datetime.now()
#选择频谱类型
saliency = []
for inputs,_ in advloader:
    batch_saliency = spec_type(inputs.numpy(),args.selec_type)
    #print(batch_saliency.shape)
    batch_saliency = batch_saliency.numpy()
    saliency.extend(batch_saliency)

time_spen_1 = (datetime.datetime.now() - start_time).total_seconds()
print('spectrum compute time:',time_spen_1)
saliency = np.array(saliency)
#print(saliency.shape)
samples = [0.2]
for sample in samples:
    #聚类和选择
    start_time = datetime.datetime.now()
    sampled_indices = batch_cluster_dataset(saliency,adv_labels,args.cluster_type,sample,args.strategy,50000)
    time_spen_2 = (datetime.datetime.now() - start_time).total_seconds()
    print('cluster time of ',sample,':',time_spen_2)
    save_sample_path = selection_dataset(sampled_indices,adv_dataset,args.model,args.sample_source,args.selec_type,args.cluster_type,str(sample),args.strategy)
