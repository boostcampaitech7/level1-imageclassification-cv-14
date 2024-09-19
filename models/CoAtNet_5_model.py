import torch
import torch.nn as nn
import timm

class CoAtNetV5FineTune(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super(CoAtNetV5FineTune, self).__init__()
        
        # `timm` 라이브러리를 사용하여 CoAtNet v5 모델 로드
        # timm 모델 중 "coatnet_5" 이름을 사용하여 사전 학습된 모델 로드
        self.backbone = timm.create_model('coatnet_5', pretrained=pretrained, features_only=True)

        # 마지막 레이어에 새로운 클래스 수로 분류 레이어를 추가
        in_features = self.backbone.feature_info[-1]['num_chs']
        
        # 학습할 FC Layer
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # AdaptiveAvgPool로 공간 차원 축소
            nn.Flatten(),  # 1차원으로 변환
            nn.Linear(in_features, num_classes)  # 새로운 FC 레이어
        )
    
    def forward(self, x):
        # Backbone을 통해 특징 추출
        features = self.backbone(x)[-1]
        
        # 추출된 특징을 사용하여 분류
        out = self.classifier(features)
        
        return out

def get_model(num_classes: int, pretrained: bool = True):
    return CoAtNetV5FineTune(num_classes, pretrained)

