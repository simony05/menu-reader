from recognition import get_text
from PIL import Image

im = Image.open("348s.jpg")

get_text(im)