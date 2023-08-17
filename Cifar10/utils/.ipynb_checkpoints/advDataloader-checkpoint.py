import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pickle

# 创建自定义数据集类
class CIFAR10AdvDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
def Cifar10advLoader(advPath,train,batch_size,shuffle,num_workers):
    # 加载pickle文件中的数据
    with open(advPath, 'rb') as f:
        adv_data = pickle.load(f)

    adv_images = adv_data['data']
    adv_labels = adv_data['labels']
    
    # 创建数据转换

    transform_train = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if train =='train':
        # 创建对抗样本集的数据集
        adv_dataset = CIFAR10AdvDataset(adv_images, adv_labels, transform=transform_train)
    elif train == 'test':
        adv_dataset = CIFAR10AdvDataset(adv_images, adv_labels, transform=transform_test)
    elif train =='sample':
        adv_dataset = CIFAR10AdvDataset(adv_images, adv_labels, transform=transform_test)

    # 创建DataLoader
    adv_dataloader = DataLoader(adv_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return adv_dataloader
