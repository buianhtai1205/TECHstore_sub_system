import tensorflow_hub as hub
import numpy as np
from transformers import BertModel, BertTokenizer


def load_model_USE():
    # Load Universal Sentence Encoder
    embed = hub.load("https://www.kaggle.com/models/google/nnlm/TensorFlow2/en-dim128/1")

    # Biểu diễn các câu dưới dạng vector
    sentences = ["I love coding", "Python is great for machine learning", "I enjoy hiking in the mountains"]
    sentence_embeddings = embed(sentences)

    # Tính toán độ tương đồng giữa các câu
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            similarity = np.inner(sentence_embeddings[i], sentence_embeddings[j])
            print(f"Similarity between '{sentences[i]}' and '{sentences[j]}': {similarity}")

    return embed

def load_model_BERT():
    # Khởi tạo mô hình BERT và tokenizer
    model_name = 'bert-base-uncased'
    model = BertModel.from_pretrained(model_name)
    return model

def load_tokenizer_BERT():
    # Khởi tạo mô hình BERT và tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return tokenizer