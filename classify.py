from transformers import pipeline
from imagetranslate import image_translate
import nltk
from nltk.corpus import stopwords
import string

nltk.download('stopwords')
stopwords = set(stopwords.words('english'))

def normalize(text):
    text = text.lower() # All lowercase
    text = text.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation
    kept_words = [word for word in text.split() if not word in stopwords] # Remove stopwords and short words
    kept_words = [word for word in kept_words if not word.isnumeric()] # Remove numbers
    kept_words = [word for word in kept_words if not word == "yuan"] # Remove "yuan"
    return (" ".join(kept_words)).strip()

translations = image_translate("menu.jpg")

def classify(translations):
    classifier = pipeline("zero-shot-classification", device = 0)

    foods = []
    labels = ["food", "not food"]
    for translation in translations:
        results = classifier(translation, labels)
        if results['labels'][0] == "food":
            if results['scores'][0] > 0.99:
                foods.append(normalize(results['sequence']))

    return foods