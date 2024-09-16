import torch.nn as nn
from transformers import CLIPModel

class ClipCustomModel(nn.Module):
    def __init__(self,
                 model_name: str):
        super(ClipCustomModel, self).__init__()
        self.model = CLIPModel.from_pretrained(model_name)
    
    def forward(self, **kwargs):
        res = self.model(**kwargs)
        return res