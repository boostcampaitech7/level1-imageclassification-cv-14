# from transformers import AutoFeatureExtractor
from transformers import AutoImageProcessor
from torchvision import transforms
import random
from PIL import Image
import numpy as np

class CvTAutoImageTransform:
    def __init__(self, is_train=True):
        self.processor = AutoImageProcessor.from_pretrained("microsoft/cvt-13")
        

    def __call__(self, image):
        return self.processor(image, return_tensors='pt')['pixel_values'][0]