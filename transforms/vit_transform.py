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