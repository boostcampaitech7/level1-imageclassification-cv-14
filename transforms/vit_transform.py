# from transformers import AutoImageProcessor
from transformers import ViTImageProcessor
from torchvision import transforms
import random
from PIL import Image
import numpy as np
from utils.image_processing import gaussian_noise

class ViTAutoImageTransform:
    def __init__(self, is_train=True):
        self.processor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-384', use_fast = True)
        

    def __call__(self, image):
        transform_image = gaussian_noise(image, 255, 15, 2)
        return self.processor(transform_image, return_tensors='pt')['pixel_values'][0]