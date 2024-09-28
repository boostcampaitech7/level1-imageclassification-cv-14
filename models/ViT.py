from transformers import ViTForImageClassification
import torch.nn as nn
import torch

class ViTModel(nn.Module):
    def __init__(
        self, 
        model_name: str,
        num_classes: int):
        super(ViTModel, self).__init__()
        self.model = ViTForImageClassification.from_pretrained(model_name, 
                                                               num_labels=num_classes,
                                                               ignore_mismatched_sizes=True)
        for name, p in self.model.named_parameters():
            if not name.startswith('classifier'):
                p.requires_grad = False
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).logits 