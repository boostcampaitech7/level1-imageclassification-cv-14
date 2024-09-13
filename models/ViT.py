import torch
import torch.nn as nn
from timm import create_model


class ViTModel(nn.Module):
    """
    Vision Transformer (ViT) 구조를 사용한 이미지 분류 모델
    """

    def __init__(self, num_classes: int, pretrained: bool = True):
        super(ViTModel, self).__init__()

        # timm 라이브러리를 사용하여 사전 훈련된 ViT 모델 로드
        self.vit = create_model('vit_base_patch16_224', pretrained=pretrained)

        # 분류기 헤드 수정
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)

# 사용 예시:
# model = ViTModel(num_classes=config.num_classes, pretrained=True)
# model.to(config.device)
