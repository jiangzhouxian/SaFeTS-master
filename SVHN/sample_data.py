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
from Models.resnet50 import *
from Models.squeezenet import *
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
import datetime
import os
import argparse
    
#os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
parser = argparse.ArgumentParser(description='Main function for difference-inducing input selection for retrain on CIFAR10 dataset')
parser.add_argument('model', help="target model", choices=['resnet50'])
parser.add_argument('sample_source', help="data sample source", choices=['cw','pgd','all'])
parser.add_argument('selec_type', help="select type", choices=['orig','phase', 'highpass', 'residual','quaternion_fourier','deepgini','entropy','robot'])
parser.add_argument('cluster_type', help="cluster type", choices=['kmeans','gmm'])
#parser.add_argument('sample', help="sample between (0,1)", type=float, default = 0.8)
parser.add_argument('strategy', help="sample strategy", choices=['high_uc', 'low_uc', 'uniform'], default = 'uniform')
args = parser.parse_args()
#采样的样本来源读取
Sample_Path = 'SVHN_adv/SVHN_{}_{}_trainset.pkl'.format(args.model,args.sample_source)
adv_dataset = SVHNAdvDataset(pickle_file=Sample_Path, transform=ToTensor())
advloader = DataLoader(adv_dataset, batch_size=10000, shuffle=False, num_workers=2)
#计时开始
start_time = datetime.datetime.now()
#选择频谱类型

saliency = []
adv_labels = []
for inputs,labels in advloader:
    batch_saliency = spec_type(inputs.numpy(),args.selec_type)
    #print(batch_saliency.shape)
    batch_saliency = batch_saliency.numpy()
    saliency.extend(batch_saliency)
    adv_labels.extend(labels.numpy())

time_spen_1 = (datetime.datetime.now() - start_time).total_seconds()
print('spectrum compute time:',time_spen_1)
saliency = np.array(saliency)
adv_labels = np.array(adv_labels)
#print(saliency.shape)

samples = [0.1]

for sample in samples:
    
    #聚类和选择
    start_time = datetime.datetime.now()
    sampled_indices = batch_cluster_dataset(saliency,adv_labels,args.cluster_type,sample,args.strategy,60000)
    time_spen_2 = (datetime.datetime.now() - start_time).total_seconds()
    print('cluster time of ',sample,':',time_spen_2)
    '''
    start_time = datetime.datetime.now()
    model = LeNet5(num_classes=10).cuda()
    model.load_state_dict(torch.load('Models/LeNet5-model.pth'))
    sampled_indices = baselines(model,advloader,args.selec_type,sample)
    time_spen_2 = (datetime.datetime.now() - start_time).total_seconds()
    print('time of ',sample,':',time_spen_2)
    '''
    save_sample_path = selection_dataset(sampled_indices,adv_dataset,args.model,args.sample_source,args.selec_type,args.cluster_type,str(sample),args.strategy)
