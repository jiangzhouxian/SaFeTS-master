import torch
import pickle
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class SVHNAdvDataset(Dataset):
    def __init__(self, pickle_file, transform=ToTensor()):
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        
        self.images = data['data']
        self.labels = data['labels']
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label
