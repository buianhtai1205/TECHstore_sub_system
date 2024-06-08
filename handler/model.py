from transformers import BertModel, BertTokenizer

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