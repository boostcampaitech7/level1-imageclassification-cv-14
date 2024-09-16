from transformers import CLIPProcessor

class ClipProcessor:
    def __init__(self, transform_name):
        self.processor = CLIPProcessor.from_pretrained(transform_name, clean_up_tokenization_spaces=True)

    def __call__(self, input_image, input_text):
        res = self.processor(text=input_text, images=input_image, return_tensors="pt", padding=True)
        return res