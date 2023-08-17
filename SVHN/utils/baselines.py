import torch
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
def baselines(model, advloader, selec_type, sample):
    # 计算样本总数
    total_samples = sum([len(batch_labels) for _, batch_labels in advloader])
    metric = []
    model = model.to(device)
    model = model.eval()
    for images, labels in advloader:
        images, labels = images.to(device), labels.to(device)
        if selec_type == 'robot':
            images.requires_grad = True
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            grads = torch.autograd.grad(loss, images)[0]
            fol_l2 = torch.norm(grads.view(grads.shape[0], -1), dim=1, p=2).cpu().detach().numpy()
            #print(fol_l2.shape)
            metric.extend(fol_l2)
        elif selec_type == 'deepgini':
            # Compute Gini values
            logits = model(images)
            probs = F.softmax(logits, dim=1).cpu().detach().numpy()
            ginis_val = np.sum(np.square(probs), axis=1)
            #print(ginis_val.shape)
            metric.extend(ginis_val)
        elif selec_type == 'entropy':
            # Compute prediction entropy
            logits = model(images)
            probs = F.softmax(logits, dim=1).cpu().detach().numpy()
            entropy = -np.sum(probs * np.log2(probs + 1e-8), axis=1)
            #print(entropy.shape)
            metric.extend(entropy)

    sample_size = int(sample * total_samples)
    metric_tensor = torch.tensor(metric)
    sorted_indices = torch.argsort(metric_tensor, descending=True)
    sampled_indices = sorted_indices[:sample_size]

    return sampled_indices

