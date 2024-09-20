import torch
import torch.nn as nn
from timm import create_model


class ViTModel(nn.Module):
    """
    ViT-G (Vision Transformer Giant) 구조를 사용한 이미지 분류 모델
    """

    def __init__(self, num_classes: int, pretrained: bool = True):
        super(ViTModel, self).__init__()

        # timm 라이브러리를 사용하여 사전 훈련된 ViT-G 모델 로드
        self.vit = create_model('vit_giant_patch14_224', pretrained=pretrained)

        # 분류기 헤드 수정
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)
