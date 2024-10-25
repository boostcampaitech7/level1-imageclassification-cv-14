import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.dropout import CoarseDropout
import cv2

class AlbumentationsTransform:
    def __init__(self, is_train: bool = True):
        # 공통 변환 설정: 이미지 리사이즈, 정규화, 텐서 변환
        common_transforms = [
            A.Resize(256,256),  # 이미지를 320,320 크기로 리사이즈
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 정규화
            ToTensorV2()  # albumentations에서 제공하는 PyTorch 텐서 변환
        ]
        
        if is_train:
            # 훈련용 변환: 랜덤 수평 뒤집기, 랜덤 회전, 랜덤 밝기 및 대비 조정 추가
            self.transform = A.Compose(
                [
                    A.RandomResizedCrop(256,256, scale=(0.8, 1.0)),
                    CoarseDropout(max_holes=4, max_height=16, max_width=16, min_holes=1, min_height=4, min_width=4, p=0.2),  # CoarseDropout으로 대체
                    A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2.0), p=0.1),
                    A.HorizontalFlip(p=0.5),  # 50% 확률로 이미지를 수평 뒤집기
                    A.Rotate(limit=15),  # 최대 15도 회전
                    A.RandomBrightnessContrast(p=0.2),
                    A.OneOf([
                        A.Emboss(p=0.3),  # 이미지 윤곽 강조
                        A.Sharpen(p=0.3),  # 이미지 샤프닝
                        A.Blur(blur_limit=3, p=0.3),  # 블러 효과
                    ], p=0.3),
                    A.Lambda(image=self.morphological_transform, p=0.2)  # 20% 확률로 형태학적 변환 적용
                ] + common_transforms
            )
        else:
            # 검증/테스트용 변환: 공통 변환만 적용
            self.transform = A.Compose(common_transforms)
            
    def morphological_transform(self, img, kernel_size=3, iterations=1, *args, **kwargs):
        """
        형태학적 변환 (침식 또는 팽창)을 이미지에 적용합니다.
        """
        # 랜덤하게 erosion 또는 dilation 선택
        if np.random.rand() > 0.5:
            # Erosion (침식): 선을 얇게 만듭니다.
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            img = cv2.erode(img, kernel, iterations=iterations)
        else:
            # Dilation (팽창): 선을 두껍게 만듭니다.
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            img = cv2.dilate(img, kernel, iterations=iterations)
        
        return img

    def __call__(self, image) -> torch.Tensor:
        # 이미지가 NumPy 배열인지 확인
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a NumPy array (OpenCV format).")
        
        # 이미지에 변환 적용 및 결과 반환
        transformed = self.transform(image=image)  # 이미지에 설정된 변환을 적용
        
        return transformed['image']  # 변환된 이미지의 텐서를 반환