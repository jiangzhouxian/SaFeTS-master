import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchattacks import PGD,FGSM,CW
#from Models.ResNet import *
from torchvision.models import resnet18,resnet50,wide_resnet50_2
from PIL import Image
import pickle
from Models_cifar10.backbones.ResNet import ResNet18
from Models_cifar10.backbones.VGG import VGG16
from Models_cifar10.backbones.DenseNet import DenseNet121
import os
# Generate adversarial examples for CIFAR-10 testset
# Load CIFAR-10 testset
transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
testset = torchvision.datasets.CIFAR10(root='./Models_cifar10/Dataset', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=2)
testloader1 = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=0)

train_dataset = torchvision.datasets.CIFAR10(root='./Models_cifar10/Dataset', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, shuffle=False, num_workers=2)


#max-min normalization
def max_min_norm(images):
    normalized_imgs = torch.zeros_like(images)
    min_max_values = torch.zeros((images.shape[0], 2))
    for i, img in enumerate(images):
        max_value = img.max()
        min_value = img.min()
        min_max_values[i] = torch.tensor([min_value, max_value])
        normalized_imgs[i] = (img - min_value) / (max_value - min_value)
    return min_max_values,normalized_imgs
#Save label
for i, data in enumerate(testloader1, 0):
    images, labels = data
    adv_labels = labels
np.save('Cifar10_adv/adv_test_labels.npy', adv_labels.numpy())
mods = ['ResNet18','DenseNet121','VGG16']
attacks = ['PGD','CW']

# Generate adversarial examples for CIFAR-10 testset.
for mod in mods:
    if mod == 'ResNet18':
        model = ResNet18()
    elif mod == 'VGG16':
        model = VGG16()
    elif mod == 'DenseNet121':
        model = DenseNet121()
    # load model's checkpoint
    checkpoint_path = './Models_cifar10/checkpoint/{}-CIFAR10.pth'.format(mod)  
    checkpoint = torch.load(checkpoint_path)
    #device = "cuda" if torch.cuda.is_available() else "cpu"

    model.load_state_dict(checkpoint['net'])
    model = model.cuda().eval()
    for att in attacks:
        if att == 'PGD':
            adversary = PGD(model, eps=4/255, steps=10, alpha=8/255)
        elif att == 'CW':
            adversary = CW(model, c=1e-4, kappa=0, steps=1000, lr=0.01)
        
        adv_images = []
        for i, data in enumerate(testloader, 0):
            images, labels = data
            images = images.cuda()
            min_max_values,normalized_imgs = max_min_norm(images)
            adv_images = adversary(normalized_imgs, labels)
            adv_images.extend(adv_images.cpu().detach().numpy())
            
        #Save adversarial examples
        np.save('Cifar10_adv/{}_adv_{}.npy'.format(att,mod), adv_images)

        adv_samples = np.load('Cifar10_adv/{}_adv_{}.npy'.format(att,mod))
        adv_samples = torch.from_numpy(adv_samples)
        adv_labels = np.load('Cifar10_adv/adv_test_labels.npy')
        to_pil = transforms.ToPILImage()
        adv_images = [to_pil(adv_sample) for adv_sample in adv_samples]
        adv_data = {
            'data': adv_images,
            'labels': adv_labels.tolist()
        }
        with open('Cifar10_adv/cifar10_{}_{}_testset.pkl'.format(mod,att), 'wb') as f:
            pickle.dump(adv_data, f)

# Generate adversarial examples for CIFAR-10 training set.
for mod in mods:
    if mod == 'ResNet18':
        model = ResNet18()
    elif mod == 'VGG16':
        model = VGG16()
    elif mod == 'DenseNet121':
        model = DenseNet121()

    checkpoint_path = './Models_cifar10/checkpoint/{}-CIFAR10.pth'.format(mod) 
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['net'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    for att in attacks:
        if att == 'PGD':
            # Create PGD adversary
            adversary = PGD(model, eps=4/255, steps=10, alpha=8/255)
        elif att == 'CW':
            # Create CW adversary
            adversary = CW(model, c=1e-4, kappa=0, steps=1000, lr=0.01)
        adv_images = []
        adv_labels = []
        for k in range(20):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                min_max_values,normalized_imgs = max_min_norm(images)
                # Generate adversarial examples
                adv_ex = adversary(normalized_imgs, labels)
                adv_images.extend(adv_ex.cpu().numpy())
                adv_labels.extend(labels.cpu().numpy())
        adv_images = np.array(adv_images)
        adv_labels = np.array(adv_labels)
        # Save adversarial data to npy file
        np.save('Cifar10_adv_20/{}_{}_adv_train.npy'.format(mod,att), adv_images)
        np.save('Cifar10_adv_20/{}_{}_adv_label.npy'.format(mod,att), adv_labels)
        
        # Load adversarial data from npy file
        adv_samples = np.load('Cifar10_adv_20/{}_{}_adv_train.npy'.format(mod,att))
        adv_labels = np.load('Cifar10_adv_20/{}_{}_adv_label.npy'.format(mod,att))
        # Convert Tensors to PIL images
        adv_samples = torch.from_numpy(adv_samples)
        to_pil = transforms.ToPILImage()
        adv_images = [to_pil(adv_sample) for adv_sample in adv_samples]

        # Create a dictionary containing images and labels
        adv_data_pkl = {
            'data': adv_images,
            'labels': adv_labels.tolist()
        }
        pkl_file_path = 'Cifar10_adv_20/{}_{}_adv_train.pkl'.format(mod,att)
        # Save the data to a pkl file
        with open(pkl_file_path, 'wb') as f:
            pickle.dump(adv_data_pkl, f)