import torch.nn as nn

from transformers import DeiTForImageClassificationWithTeacher

class DeitCustomModel(nn.Module):
    def __init__(self,
                 model_name: str):
        super(DeitCustomModel, self).__init__()
        self.model = DeiTForImageClassificationWithTeacher.from_pretrained(model_name)
    
    def forward(self, **kwargs):
        res = self.model(**kwargs)
        return res