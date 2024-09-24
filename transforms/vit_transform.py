# from transformers import AutoImageProcessor
from transformers import ViTImageProcessor
from torchvision import transforms
import random
from PIL import Image
import numpy as np

class ViTAutoImageTransform:
    def __init__(self, is_train=True):
        self.processor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-384', use_fast = True)
        

    def __call__(self, image):
        return self.processor(image, return_tensors='pt')['pixel_values'][0]

# class ViTAutoImageTransform:
#     def __init__(self, is_train=True):
#         self.processor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-384', use_fast = True)
#         self.is_train = is_train

#     def __call__(self, image):
#         if self.is_train:
#             # Mixup을 적용하는 로직
#             alpha = 0.2  # Mixup의 alpha 값
#             lam = np.random.beta(alpha, alpha)  # Mixup의 람다 값
#             image2 = random.choice(self.processor.image_processor)  # 다른 이미지 선택
#             image = (image * lam + image2 * (1 - lam)).astype(np.uint8)  # Mixup 적용
#         return self.processor(image, return_tensors='pt')['pixel_values'][0]