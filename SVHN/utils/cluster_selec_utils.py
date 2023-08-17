import numpy as np
import torch
import torchvision.transforms as transforms

from torch.fft import fft2
from sklearn.cluster import KMeans,DBSCAN
from sklearn.mixture import GaussianMixture
from collections import defaultdict
import pickle
import random
import os
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

def batch_cluster_dataset(data, label, clu_type, sample, strategy, batch_size):
    num_batches = len(data) // batch_size
    sampled_indices = []

    for batch_idx in range(num_batches):
        batch_data = data[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        batch_label = label[batch_idx * batch_size: (batch_idx + 1) * batch_size]

        batch_sampled_indices = cluster_dataset(batch_data, batch_label, clu_type, sample, strategy)

        original_indices = [batch_idx * batch_size + idx for idx in batch_sampled_indices]
        
        sampled_indices.extend(original_indices)

    return sampled_indices

def selection_dataset(sampled_indices, testset, model, sample_source, selec_type, cluster_type, sample, strategy):
    
    sampled_data = [testset.images[i] for i in sampled_indices]
    sampled_labels = [testset.labels[i] for i in sampled_indices]

    # 将采样数据和标签转换为 numpy 数组
    #sampled_data = np.array(sampled_data).astype(np.float32)
    sampled_labels = np.array(sampled_labels)

    #sampled_data = torch.from_numpy(sampled_data)
    # 将Tensor转换为PIL图像
    #to_pil = transforms.ToPILImage()
    #adv_images = [to_pil(adv_sample) for adv_sample in sampled_data]

    # 创建一个字典，包含图像和标签
    sample_data = {
        'data': sampled_data,
        'labels': sampled_labels.tolist()
    }

    # 保存到文件
    save_samples_path = 'Sample_data/SVHN_{}_{}_{}_{}_{}_{}_samples.pkl'.format(model, sample_source, selec_type, cluster_type, str(sample), strategy)
    
    with open(save_samples_path, 'wb') as f:
        pickle.dump(sample_data, f)

    return save_samples_path