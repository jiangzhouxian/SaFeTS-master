import torch
import torch.nn.functional as F
import torch.autograd as autograd

def baselines(model, advloader, selec_type, sample):
    total_samples = sum([len(batch_labels) for _, batch_labels in advloader])
    probs_list = []
    metric_list = []

    # Iterate through each batch
    for batch_data, batch_labels in advloader:
        outputs = model(batch_data)
        batch_probs = F.softmax(outputs, dim=1)
        probs_list.append(batch_probs)

        batch_metric = []
        
        for i in range(len(batch_data)):
            input_sample = batch_data[i].unsqueeze(0)
            target_sample = batch_labels[i].unsqueeze(0)
            output_sample = model(input_sample)
            batch_prob_sample = batch_probs[i]
            
            if selec_type == 'robot':
                loss_sample = F.cross_entropy(output_sample, target_sample)
                gradients = autograd.grad(loss_sample, model.parameters(), retain_graph=True)
                grad_norm = sum([grad.data.norm(2).item() ** 2 for grad in gradients]) ** 0.5
                batch_metric.append(grad_norm)

            elif selec_type == 'deepgini':
                gini = 1 - torch.sum(batch_prob_sample**2)
                batch_metric.append(gini.item())

        metric_list.append(torch.tensor(batch_metric))

    probs = torch.cat(probs_list, dim=0)
    metric = torch.cat(metric_list, dim=0)
    
    sample_size = int(sample * total_samples)
    sorted_indices = torch.argsort(metric, descending=True)
    sampled_indices = sorted_indices[:sample_size]

    return sampled_indices
