import io
import os
import re
import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import zipfile

# Đảm bảo rằng đầu ra tiêu chuẩn sử dụng mã hóa utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', write_through=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', write_through=True)

# Thiết lập môi trường UTF-8 cho console
os.system('chcp 65001')

# Function to load nltk resources from files
def load_nltk_resources():
    nltk_data_path = 'D:\\KLTN\\TECHstore_sub_system\\resources'

    # Ensure NLTK can find the downloaded resources
    nltk.data.path.append(nltk_data_path)

    # Load punkt tokenizer
    punkt_zip_path = os.path.join(nltk_data_path, 'tokenizers\\punkt.zip')
    if not os.path.exists(os.path.join(nltk_data_path, 'tokenizers\\punkt')):
        with zipfile.ZipFile(punkt_zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(nltk_data_path, 'tokenizers'))
    nltk.data.path.append(os.path.join(nltk_data_path, 'tokenizers'))

    # Load wordnet
    wordnet_zip_path = os.path.join(nltk_data_path, 'corpora\\wordnet.zip')
    if not os.path.exists(os.path.join(nltk_data_path, 'corpora\\wordnet')):
        with zipfile.ZipFile(wordnet_zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(nltk_data_path, 'corpora'))
    nltk.data.path.append(os.path.join(nltk_data_path, 'corpora'))

def preprocess_text(text):
    # Call the function to load nltk resources
    load_nltk_resources()

    # Load stopwords từ file stopwords_vietnamese.txt
    stopwords_file = 'D:\\KLTN\\TECHstore_sub_system\\stopwords_vietnamese.txt'
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        custom_stopwords = f.read().splitlines()

    # Tải tài nguyên WordNetLemmatizer một lần
    lemmatizer = WordNetLemmatizer()

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

# Ví dụ sử dụng
sample_text = "<p>Đây là một ví dụ về text với <b>HTML tags</b>.</p>"
print(preprocess_text(sample_text))
