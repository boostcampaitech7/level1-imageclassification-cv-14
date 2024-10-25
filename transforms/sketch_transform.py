from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter
import torch
import cv2
import numpy as np

class SketchTransform:
    def __init__(self, is_train=True):
        if is_train:
            # 훈련 데이터에 대한 전처리 파이프라인 (라인 강조 포함)
            self.transform = transforms.Compose([
                transforms.Lambda(lambda img: self.enhance_lines(img)),  # 라인 강조 필터 적용
                transforms.RandomResizedCrop(320, scale=(0.8, 1.0)),  # 랜덤 크롭 및 리사이즈
                transforms.RandomHorizontalFlip(p=0.5),  # 50% 확률로 이미지를 수평 뒤집기
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 이동, 회전, 스케일 변환
                transforms.ToTensor(),  # PIL 이미지 -> torch.Tensor로 변환
                transforms.ConvertImageDtype(torch.float32),  # Tensor를 float32로 변환
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # RGB 이미지의 정규화
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random')  # 랜덤한 부분 삭제
            ])
        else:
            # 검증 및 테스트 데이터에 대한 전처리 파이프라인 (기본 전처리만 적용)
            self.transform = transforms.Compose([
                transforms.Resize((320, 320)),  # 모델에 맞는 크기로 조정
                transforms.ToTensor(),  # PIL 이미지 -> torch.Tensor로 변환
                transforms.ConvertImageDtype(torch.float32),  # Tensor를 float32로 변환
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # RGB 이미지의 정규화
            ])

    def enhance_lines(self, img):
        """
        이미지의 라인을 강조하는 필터 적용.
        """
        # PIL 이미지를 OpenCV 형식으로 변환
        img_cv = np.array(img)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)  # 그레이스케일로 변환 (에지 검출 위해)
        img_cv = cv2.GaussianBlur(img_cv, (3, 3), 0)  # 가우시안 블러로 노이즈 제거

        # 라플라시안 필터로 엣지 검출
        edges = cv2.Laplacian(img_cv, cv2.CV_64F)
        edges = cv2.convertScaleAbs(edges)

        # 원본 이미지에 에지 강조를 더하기 위해 RGB로 변환 후 합성
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        enhanced_img = cv2.addWeighted(np.array(img), 0.7, edges_colored, 0.3, 0)  # 원본과 에지 합성

        # 다시 PIL 이미지로 변환
        return Image.fromarray(enhanced_img)

    def __call__(self, image):
        # 이미지를 PIL 포맷으로 변환
        if not isinstance(image, Image.Image):  # 만약 이미 PIL 이미지가 아니라면 변환
            image = Image.fromarray(image)

        # 이미지에 전처리 적용
        transformed_image = self.transform(image).unsqueeze(0)  # 배치 차원을 추가하여 반환 (N, C, H, W 형태)
        return transformed_image
