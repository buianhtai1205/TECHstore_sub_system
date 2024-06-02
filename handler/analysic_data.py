import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import zipfile

# Function to load nltk resources from files
def load_nltk_resources():
    # Check if resources have been downloaded
    if (not os.path.isfile('/home/anhtai/nltk_data/tokenizers/punkt.zip')
            or not os.path.isfile('/home/anhtai/nltk_data/corpora/wordnet.zip')):
        raise FileNotFoundError("Necessary NLTK resources not found in the system.")

    # Load punkt tokenizer
    punkt_zip_path = '/home/anhtai/nltk_data/tokenizers/punkt/PY3/english.pickle'
    nltk.data.load(punkt_zip_path)

    # Load wordnet
    wordnet_zip_path = '/home/anhtai/nltk_data/corpora/wordnet.zip'
    with zipfile.ZipFile(wordnet_zip_path, 'r') as zip_ref:
        zip_ref.extractall('/home/anhtai/nltk_data/corpora/wordnet/')

# Call the function to load nltk resources
load_nltk_resources()

# Load stopwords từ file stopwords_vietnamese.txt
stopwords_file = '/home/anhtai/PycharmProjects/fastApiProject/stopwords_vietnamese.txt'
with open(stopwords_file, 'r', encoding='utf-8') as f:
    custom_stopwords = f.read().splitlines()

# Tải tài nguyên WordNetLemmatizer một lần
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Remove HTML tags
    text = clean_html_tags(text)
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Convert the list of stopwords into a set for faster lookup
    stop_words = set(custom_stopwords)

    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def clean_html_tags(text):
    if isinstance(text, str):
        soup = BeautifulSoup(text, 'html.parser')
        clean_text = soup.get_text(separator=' ')
        return clean_text
    else:
        return ''
