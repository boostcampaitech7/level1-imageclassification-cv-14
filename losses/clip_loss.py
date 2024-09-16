import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPLoss(nn.Module):
    '''
    CLIP의 손실함수
    '''
    def __init__(self):
        super(CLIPLoss, self).__init__()

    def loss_fn(self, logits_i, logits_t, device):
        labels = torch.arange(logits_i.shape[0], device=device)
        loss_i = F.cross_entropy(logits_i, labels)
        loss_t = F.cross_entropy(logits_t, labels)
        loss = (loss_i + loss_t) / 2
        
        return loss
    
    def forward(self, logits_i, logits_t, device):
        return self.loss_fn(logits_i, logits_t, device)