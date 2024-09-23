from transformers import ViTImageProcessor
from torchvision import transforms
import random
from PIL import Image
import numpy as np

class ViTAutoImageTransform:
    def __init__(self, is_train=True):
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k', use_fast = True)
        self.is_train = is_train
        
        if self.is_train:
            self.train_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33))
            ])

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if self.is_train:
            image = self.train_transforms(image)
        
        processed_image = self.processor(image, return_tensors='pt')['pixel_values'][0]
        
        if self.is_train and random.random() < 0.5:
            processed_image = transforms.functional.gaussian_blur(processed_image, kernel_size=3)
        
        return processed_image