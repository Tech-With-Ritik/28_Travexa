import pytesseract
from PIL import Image

def extract_text_from_image(path):
    try:
        return pytesseract.image_to_string(Image.open(path))
    except:
        return ""
