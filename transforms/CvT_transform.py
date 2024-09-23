# from transformers import AutoFeatureExtractor
from transformers import AutoFeatureExtractor
from torchvision import transforms
import random
from PIL import Image
import numpy as np

class CvTAutoFeatureExtractor:
    def __init__(self, is_train=True):
        self.processor = AutoFeatureExtractor.from_pretrained("microsoft/cvt-13", use_fast = True)
        

    def __call__(self, image):
        return self.processor(image, return_tensors='pt')['pixel_values'][0]