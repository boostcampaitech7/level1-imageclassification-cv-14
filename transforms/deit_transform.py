from transformers import AutoFeatureExtractor
from utils.image_processing import gaussian_noise

class DeiTProcessor:
    def __init__(self, transform_name):
        self.processor = AutoFeatureExtractor.from_pretrained(transform_name)

    def __call__(self, input_image):
        img = gaussian_noise(input_image, 255, 15, 2)
        return self.processor(images=img, return_tensors="pt")
    
        