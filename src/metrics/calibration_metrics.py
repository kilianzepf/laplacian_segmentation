import numpy as np
import torch

def entropy_prob(probs):
    p = probs
    eps = 1e-12
    logp = torch.log(p + eps)
    plogp = p * logp
    entropy = -torch.sum(plogp, dim=1)
    return entropy


def total_entropy(probs, img_size=64):
    predictive_entropy = entropy_prob(probs)
    avg_total_entropy = predictive_entropy.view(-1, img_size * img_size).mean(dim=1)
    return avg_total_entropy

