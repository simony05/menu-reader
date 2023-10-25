from recognition import configure, image64, image_to_text
from translate import translate

def image_translate(image):
    translations = []
    texts = image_to_text(image)
    for text in texts:
        translation = translate(text)
        translations.append(translation)
    return translations