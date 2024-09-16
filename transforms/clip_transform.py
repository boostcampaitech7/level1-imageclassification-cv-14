from transformers import CLIPProcessor

class ClipProcessor:
    def __init__(self, transform_name, is_train : bool = True):
        self.processor = CLIPProcessor.from_pretrained(transform_name, clean_up_tokenization_spaces=True)
        self.is_train = is_train

    def __call__(self, input_image, input_text):
        return self.processor(text=input_text, images=input_image, return_tensors="pt", padding=True)
        