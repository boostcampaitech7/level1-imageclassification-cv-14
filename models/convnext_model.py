import torch
import torch.nn as nn
import timm

class Convnext_Model(nn.Module):
    """
    Timm 라이브러리를 사용하여 다양한 사전 훈련된 모델을 제공하는 클래스.
    """
    def __init__(
        self, 
        model_name: str, 
        num_classes: int, 
        pretrained: bool,
        fine_tune_layers: int = 2,  # 학습할 마지막 N개의 블록
        **kwargs
    ):
        super(Convnext_Model, self).__init__()
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
        )

        # 모든 레이어 동결
        for param in self.model.parameters():
            param.requires_grad = False

        # 최종 분류기 레이어 수정
        in_features = self.model.get_classifier().in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

        # 분류기 레이어를 학습 가능하게 설정
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        # 마지막 N개의 블록을 학습 가능하게 설정 (fine-tune)
        for layer in list(self.model.children())[-fine_tune_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.model(x)