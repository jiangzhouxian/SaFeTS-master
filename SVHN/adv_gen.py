import torch
from torch.utils.data import Dataset
from torchvision.datasets import SVHN
from torchvision.transforms import ToPILImage, Normalize,transforms
import numpy as np
import torchvision
import torchattacks
#from Models.ResNet import *
from PIL import Image
import pickle

from Models.resnet50 import *
from Models.wide_resnet50 import *
from Models.squeezenet import *
from torchvision.models import resnet50,wide_resnet50_2,squeezenet1_0,vgg16

# 加载SVHN测试集
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = SVHN(root='./Dataset', split='train',download=False, transform=transform)
test_dataset = SVHN(root='./Dataset', split='test',download=False, transform=transform)

batch_size = 5000  # 设置批次个数
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=False)
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

#max-min归一化和反归一化
def max_min_norm(images):
    normalized_imgs = torch.zeros_like(images)
    min_max_values = torch.zeros((images.shape[0], 2))
    for i, img in enumerate(images):
        max_value = img.max()
        min_value = img.min()
        min_max_values[i] = torch.tensor([min_value, max_value])
        normalized_imgs[i] = (img - min_value) / (max_value - min_value)
    return min_max_values,normalized_imgs

def load_model(mod):
    if mod == 'resnet50':
        model = r50(resn50)
        model.load_state_dict(torch.load('./Models/svhn_resnet50.pth'))
    elif mod == 'wide_resnet50':
        model = wr50(wresn50)
        model.load_state_dict(torch.load('./Models/svhn_wide_resnet50.pth'))
    elif mod == 'squeezenet':  
        model = squeezenet(squ)
        model.load_state_dict(torch.load('./Models/svhn_squeezenet.pth'))
    return model

def load_attack(att):
    if att == 'pgd':
        adversary = torchattacks.PGD(model, eps=0.01, alpha=0.01, steps=10)
    elif att == 'cw':
        adversary = torchattacks.CW(model, c=1, kappa=0, steps=20, lr=0.01)
    return adversary

mods = ['squeezenet']
attacks = ['pgd']
# adv gen for test set
for mod in mods:
    model = load_model(mod)
    model = model.eval()
    for att in attacks:
        adversary = load_attack(att)
        adv_samples = []
        adv_labels = []
        
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            min_max_values,normalized_imgs = max_min_norm(images)
            # Generate adversarial examples
            adv_ex = adversary(normalized_imgs, labels)
            print(adv_ex.shape)

            # Append values to lists
            adv_samples.extend(adv_ex.cpu().detach().numpy())
            adv_labels.extend(labels.cpu().numpy())
        
        np.save('SVHN_adv/test_{}_adv_{}.npy'.format(mod,att), adv_samples)
        np.save('SVHN_adv/test_labels.npy',adv_labels)
        # Convert Tensors to PIL images
        
for mod in mods:
    for att in attacks:        
        # 从npy文件中加载对抗样本集
        adv_samples = np.load('SVHN_adv/test_{}_adv_{}.npy'.format(mod,att))
        # 将numpy数组转换为PyTorch Tensor
        adv_samples = torch.from_numpy(adv_samples)
        adv_labels = np.load('SVHN_adv/test_labels.npy')
        to_pil = transforms.ToPILImage()
        adv_images = [to_pil(adv_sample) for adv_sample in adv_samples]

        # 创建一个字典，包含图像和标签
        adv_data = {
            'data': adv_images,
            'labels': adv_labels.tolist()
        }

        # 保存到文件
        with open('SVHN_adv/SVHN_{}_{}_testset.pkl'.format(mod,att), 'wb') as f:
            pickle.dump(adv_data, f)


# adv gen for train set
for mod in mods:
    model = load_model(mod)
    model = model.eval()
    for att in attacks:
        adversary = load_attack(att)
        adv_samples = []
        adv_labels = []
        for index in range(10):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                min_max_values,normalized_imgs = max_min_norm(images)
                # Generate adversarial examples
                adv_ex = adversary(normalized_imgs, labels)
                print(adv_ex.shape)

                # Append values to lists
                adv_samples.extend(adv_ex.cpu().detach().numpy())
                adv_labels.extend(labels.cpu().numpy())
        
        np.save('SVHN_adv/train_{}_adv_{}.npy'.format(mod,att), adv_samples)
        np.save('SVHN_adv/train_labels.npy',adv_labels)
        # Convert Tensors to PIL images
        
for mod in mods:
    for att in attacks:        
        # 从npy文件中加载对抗样本集
        adv_samples = np.load('SVHN_adv/train_{}_adv_{}.npy'.format(mod,att))
        # 将numpy数组转换为PyTorch Tensor
        adv_samples = torch.from_numpy(adv_samples)
        adv_labels = np.load('SVHN_adv/train_labels.npy')
        to_pil = transforms.ToPILImage()
        adv_images = [to_pil(adv_sample) for adv_sample in adv_samples]

        # 创建一个字典，包含图像和标签
        adv_data = {
            'data': adv_images,
            'labels': adv_labels.tolist()
        }

        # 保存到文件
        with open('SVHN_adv/SVHN_{}_{}_trainset.pkl'.format(mod,att), 'wb') as f:
            pickle.dump(adv_data, f)
            '''