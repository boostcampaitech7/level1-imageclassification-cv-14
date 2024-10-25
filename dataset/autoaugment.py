
class AlbumentationsTransform:
    def __init__(self, is_train: bool = True):
        # 공통 변환 설정: 이미지 리사이즈, 정규화, 텐서 변환
        self.common_transforms = A.Compose([
            A.Resize(224, 224),  # 이미지를 224x224 크기로 리사이즈
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 정규화
            ToTensorV2()  # PyTorch 텐서 변환
        ])

        if is_train:
            # PyTorch AutoAugment 설정 (ImageNet 정책 사용)
            self.autoaugment = transforms.Compose([
                transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET)
            ])
        else:
            self.autoaugment = None  # AutoAugment는 검증 시 사용하지 않음

    def __call__(self, image) -> torch.Tensor:
        # 이미지가 NumPy 배열인지 확인
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a NumPy array (OpenCV format).")
        
        # 1. 원본 이미지를 변환
        original_transformed = self.common_transforms(image=image)['image']

        # 2. 증강된 이미지도 반환 (학습 모드일 경우)
        if self.autoaugment:
            pil_image = Image.fromarray(image)  # NumPy를 PIL로 변환
            pil_image = self.autoaugment(pil_image)  # AutoAugment 적용
            aug_image = np.array(pil_image)  # 다시 NumPy 배열로 변환
            aug_transformed = self.common_transforms(image=aug_image)['image']  # 증강된 이미지 변환

            # 원본 이미지와 증강된 이미지를 모두 반환
            return original_transformed, aug_transformed

        # 검증/테스트 모드일 경우, 원본 이미지만 반환
        return original_transformed, None