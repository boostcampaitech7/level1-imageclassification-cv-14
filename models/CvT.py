from transformers import AutoImageProcessor, CvtForImageClassification
import torch.nn as nn
import torch

class CvTModel(nn.Module):

    def __init__(
        self, 
        model_name: str,
        num_classes: int,
        fine_tune: bool = False,
        dropout_rate: float = 0.1
    ):
        super(CvTModel, self).__init__()
        self.model = CvtForImageClassification.from_pretrained(model_name, 
                                                               num_labels=num_classes,
                                                               ignore_mismatched_sizes=True)
        
        # 미세 조정 옵션 추가
        if not fine_tune:
            for name, param in self.model.named_parameters():
                if not name.startswith('classifier'):
                    param.requires_grad = False
        

        print(self.model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x).logits
        return outputs
    
class EnhancedCvTModel(nn.Module):
    def __init__(
        self, 
        model_name: str,
        num_classes: int,
        fine_tune_mode: str = 'full',
        freeze_layers: int = 0,
        dropout_rate: float = 0.1
    ):
        super(EnhancedCvTModel, self).__init__()
        self.model = CvtForImageClassification.from_pretrained(model_name, 
                                                               num_labels=num_classes,
                                                               ignore_mismatched_sizes=True)
        
        # dropout을 classifier 이전에 추가
        self.dropout = nn.Dropout(dropout_rate)
        
        # classifier를 새로 정의 (dropout 포함)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            self.dropout,
            nn.Linear(in_features, num_classes)
        )
        
        # Fine-tuning 모드 설정
        self.set_fine_tuning_mode(fine_tune_mode, freeze_layers)
        
        # 레이어 수 계산
        self.num_layers = self.count_layers()
        
        print(self.model)
        print(f"Total number of layers: {self.num_layers}")

    def count_layers(self):
        def count_conv_layers(module):
            return sum(1 for m in module.modules() if isinstance(m, (nn.Conv2d, nn.Linear)))
        
        return count_conv_layers(self.model.cvt) + 1  # +1 for the classifier

    def set_fine_tuning_mode(self, mode: str, freeze_layers: int):
        if mode == 'full':
            # 전체 모델 학습
            for param in self.model.parameters():
                param.requires_grad = True
        elif mode == 'last_layer':
            # 마지막 레이어만 학습
            for name, param in self.model.named_parameters():
                if name.startswith('classifier'):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif mode == 'partial':
            # 지정된 수의 레이어만 고정
            total_layers = self.count_layers()
            if freeze_layers >= total_layers:
                raise ValueError(f"freeze_layers ({freeze_layers}) must be less than total layers ({total_layers})")
            
            layers = list(self.model.cvt.named_modules())
            frozen_count = 0
            for name, module in layers:
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    if frozen_count < freeze_layers:
                        for param in module.parameters():
                            param.requires_grad = False
                        frozen_count += 1
                    else:
                        for param in module.parameters():
                            param.requires_grad = True
            # classifier는 항상 학습 가능하게 설정
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        else:
            raise ValueError("Invalid fine_tune_mode. Choose 'full', 'last_layer', or 'partial'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).logits