from transformers import ViTForImageClassification
import torch.nn as nn
import torch

class ViTModel(nn.Module):
    """
    개선된 ViTModel
    """
    def __init__(
        self, 
        model_name: str,
        num_classes: int,
        fine_tune: bool = False,
        dropout_rate: float = 0.1
    ):
        super(ViTModel, self).__init__()
        self.model = ViTForImageClassification.from_pretrained(model_name, 
                                                               num_labels=num_classes,
                                                               ignore_mismatched_sizes=True)
        
        # 미세 조정 옵션 추가
        if not fine_tune:
            for name, param in self.model.named_parameters():
                if not name.startswith('classifier'):
                    param.requires_grad = False
        
        # 드롭아웃 레이어 추가
        self.dropout = nn.Dropout(dropout_rate)
        
        # 분류기 헤드 수정
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        return self.dropout(outputs.logits)