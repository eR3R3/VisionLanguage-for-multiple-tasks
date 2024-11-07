from data_processing import PaliGemmaProcessor
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer
import numpy as np
from PIL import Image

tokenizer = AutoTokenizer.from_pretrained(
    "google/paligemma-3b-pt-224",
    token="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")
preprocesser = PaliGemmaProcessor(tokenizer=tokenizer, num_image_tokens=3, image_size=224)
images = [Image.open('/Users/apple/Desktop/照片/d52e646f16675d19.jpeg')]
text = ["this is hemin"]
x = preprocesser(text=text, images=images)
print(x.items())