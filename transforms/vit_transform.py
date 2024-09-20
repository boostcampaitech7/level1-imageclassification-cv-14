from transformers import AutoImageProcessor

class ViTAutoImageTransform:
    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')

    def __call__(self, image):
        return self.processor(image, return_tensors='pt')['pixel_values'][0]