import numpy as np
import torch
import torchvision.transforms as transforms

from torch.fft import fft2
from sklearn.cluster import KMeans,DBSCAN
from sklearn.mixture import GaussianMixture
from collections import defaultdict
import pickle
import random

#按照不同聚类方法聚类，并按一定比例采样保存到文件中
def cluster_dataset(data,label,clu_type,sample,strategy):
    if clu_type == 'kmeans':
        kmeans = KMeans(n_clusters=10, init='k-means++', random_state=42)
        cluster_labels = kmeans.fit_predict(data.reshape(len(data), -1))
        cluster_centers = kmeans.cluster_centers_
        n_clusters = 10

    elif clu_type == 'dbscan':
        dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
        cluster_labels = dbscan.fit_predict(data.reshape(len(data), -1))
        unique_labels = np.unique(cluster_labels)
        cluster_centers = [np.mean(data[cluster_labels == label], axis=0) for label in unique_labels if label != -1]
        n_clusters = len(cluster_centers)

    elif clu_type == 'gmm':
        gmm = GaussianMixture(n_components=10, random_state=42)
        cluster_labels = gmm.fit_predict(data.reshape(len(data), -1))
        cluster_centers = gmm.means_
        cluster_covariances = gmm.covariances_
        n_clusters = 10

    sampled_indices = []
    for cluster_idx in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_idx)[0]
        samples_per_cluster = int(len(cluster_indices) * sample)
        if clu_type == 'gmm':
            distances = np.array([np.trace(cluster_covariances[cluster_idx]) for _ in cluster_indices])
        elif clu_type == 'kmeans':
            #distances = np.linalg.norm(data[cluster_indices] - cluster_centers[cluster_idx], axis=1)
            distances = np.linalg.norm(data[cluster_indices].reshape(len(cluster_indices), -1) - cluster_centers[cluster_idx], axis=1)
        elif clu_type == 'dbscan':
            distances = np.linalg.norm(data[cluster_indices] - cluster_centers[cluster_idx], axis=1)

        if strategy == 'high_uc':
            sorted_indices = np.argsort(distances)[::-1]
        elif strategy == 'low_uc':
            sorted_indices = np.argsort(distances)
        elif strategy == 'uniform':
            sorted_indices = np.arange(len(cluster_indices))
            np.random.shuffle(sorted_indices)
            
        sampled_indices.extend(cluster_indices[sorted_indices[:samples_per_cluster]])

    return sampled_indices

def selection_datatset(sampled_indices,testset,model,sample_source,selec_type,cluster_type,sample,strategy):
    
    sampled_data = []
    sampled_labels = []

    #unsampled_data = []
    #unsampled_labels = []

    for i in range(len(testset)):
        if i in sampled_indices:
            sampled_data.append(testset.images[i])
            sampled_labels.append(testset.labels[i])
        #else:
            #unsampled_data.append(testset.images[i])
            #unsampled_labels.append(testset.labels[i])

    # 将采样数据和标签转换为 numpy 数组
    sampled_data = np.array(sampled_data)
    sampled_labels = np.array(sampled_labels)

    # 将未采样数据和标签转换为 numpy 数组
    #unsampled_data = np.array(unsampled_data)
    #unsampled_labels = np.array(unsampled_labels)

    sampled_data = torch.from_numpy(sampled_data)
    # 将Tensor转换为PIL图像
    to_pil = transforms.ToPILImage()
    adv_images = [to_pil(adv_sample) for adv_sample in sampled_data]

    # 创建一个字典，包含图像和标签
    sample_data = {
        'data': adv_images,
        'labels': sampled_labels.tolist()
    }

    # 保存到文件
    save_samples_path = 'Sample_data/cifar10_{}_{}_{}_{}_{}_{}_samples.pkl'.format(model,sample_source,selec_type,cluster_type,str(sample),strategy)
    
    with open(save_samples_path, 'wb') as f:
        pickle.dump(sample_data, f)
    '''
    # 将numpy数组转换为PyTorch Tensor
    remaining_samples = torch.from_numpy(unsampled_data)
    # 将Tensor转换为PIL图像
    to_pil = transforms.ToPILImage()
    remaining_images = [to_pil(remaining_sample) for remaining_sample in remaining_samples]

    # 创建一个字典，包含图像和标签
    remaining_data = {
        'data': remaining_images,
        'labels': unsampled_labels.tolist()
    }

    # 保存到文件
    save_remain_path = 'Sample_data/cifar10_{}_{}_{}_{}_{}_remain.pkl'.format(model,sample_source,selec_type,cluster_type,str(sample))
    with open(save_remain_path, 'wb') as f:
        pickle.dump(remaining_data, f)
'''
    return save_samples_path