from torchvision import transforms
from PIL import Image
import torch

class ConvnextTransform:
    def __init__(self, is_train=True):
        if is_train:
            # 훈련 데이터에 대한 전처리 파이프라인 (데이터 증강 포함)
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),  # 50% 확률로 이미지를 수평 뒤집기
                transforms.RandomRotation(15),  # 최대 15도 회전
                transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 밝기 및 대비 조정
                transforms.Resize((320, 320)),  # 모델에 맞는 크기로 조정
                transforms.ToTensor(),  # PIL 이미지 -> torch.Tensor로 변환
                transforms.ConvertImageDtype(torch.float32),  # Tensor를 float32로 변환
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
            ])
        else:
            # 검증 데이터에 대한 전처리 파이프라인 (기본 전처리만 적용)
            self.transform = transforms.Compose([
                transforms.Resize((320, 320)),  # 모델에 맞는 크기로 조정
                transforms.ToTensor(),  # PIL 이미지 -> torch.Tensor로 변환
                transforms.ConvertImageDtype(torch.float32),  # Tensor를 float32로 변환
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
            ])
    
    def __call__(self, image):
        # 이미지를 PIL 포맷으로 변환
        if isinstance(image, Image.Image):  # 이미 PIL 이미지인 경우
            pil_image = image
        else:
            pil_image = Image.fromarray(image)  # numpy 배열을 PIL 이미지로 변환
        
        # 이미지에 전처리 적용
        transformed_image = self.transform(pil_image).unsqueeze(0)  # 배치 차원을 추가하여 반환 (N, C, H, W 형태)
        return transformed_image