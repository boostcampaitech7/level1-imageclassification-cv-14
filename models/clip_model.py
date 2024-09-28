import torch.nn as nn
from transformers import CLIPModel


class ClipCustomModel(nn.Module):
    def __init__(self,
                 model_name: str):
        super(ClipCustomModel, self).__init__()
        self.model = CLIPModel.from_pretrained(model_name)

        # for name, p in self.model.named_parameters():
        #     if not name.startswith('clas')

    def forward(self, x):
        # with torch.no_grad():
        #     features = self.model.encode_image(x).float()  # Convert to float32
        # return self.classifier(features)
        res = self.model(x)
        print("forward", res)
        return res
