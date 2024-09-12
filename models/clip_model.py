import torch.nn as nn
from transformers import CLIPModel

class ClipCutomModel(nn.Module):
    def __init__(self,
                 model_name: str):
        super(ClipCutomModel, self).__init__()
        self.model = CLIPModel.from_pretrained(model_name)
    
    def forward(self, x):
        # with torch.no_grad():
        #     features = self.model.encode_image(x).float()  # Convert to float32
        # return self.classifier(features)
        return self.model(x)