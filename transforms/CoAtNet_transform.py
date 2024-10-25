import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.dropout import CoarseDropout

class AlbumentationsTransform:
    def __init__(self, is_train: bool = True):
        # 공통 변환 설정: 이미지 리사이즈, 정규화, 텐서 변환
        common_transforms = [
            A.Resize(224, 224),  # 이미지를 224x224 크기로 리사이즈
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 정규화
            ToTensorV2()  # albumentations에서 제공하는 PyTorch 텐서 변환
        ]
        
        if is_train:
            # 훈련용 변환: 랜덤 수평 뒤집기, 랜덤 회전, 랜덤 밝기 및 대비 조정 추가
            self.transform = A.Compose(
                [
                    A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
                    CoarseDropout(max_holes=4, max_height=16, max_width=16, min_holes=1, min_height=4, min_width=4, p=0.3),  # CoarseDropout으로 대체
                    A.GaussianBlur(blur_limit=3, p=0.1),  # 가우시안 블러 추가
                    A.HorizontalFlip(p=0.5),  # 50% 확률로 이미지를 수평 뒤집기
                    A.Rotate(limit=15),  # 최대 15도 회전
                    A.RandomBrightnessContrast(p=0.2),
                    A.GaussianBlur(blur_limit=3, p=0.1),# 밝기 및 대비 무작위 조정
                                        A.OneOf([
                        A.Emboss(p=0.3),  # 이미지 윤곽 강조
                        A.Sharpen(p=0.3),  # 이미지 샤프닝
                        A.Blur(blur_limit=3, p=0.3),  # 블러 효과
                    ], p=0.3)
                ] + common_transforms
            )
        else:
            # 검증/테스트용 변환: 공통 변환만 적용
            self.transform = A.Compose(common_transforms)

    def __call__(self, image) -> torch.Tensor:
        # 이미지가 NumPy 배열인지 확인
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a NumPy array (OpenCV format).")
        
        # 이미지에 변환 적용 및 결과 반환
        transformed = self.transform(image=image)  # 이미지에 설정된 변환을 적용
        
        return transformed['image']  # 변환된 이미지의 텐서를 반환