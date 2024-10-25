import torch.nn as nn

from transformers import DeiTForImageClassification

class DeitCustomModel(nn.Module):
    def __init__(self,
                 model_name: str,
                 num_labels: int):
        super(DeitCustomModel, self).__init__()
        self.model = DeiTForImageClassification.from_pretrained(model_name, num_labels=num_labels)
    
    def forward(self, **kwargs):
        res = self.model(**kwargs)
        return res