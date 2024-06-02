import torch
import numpy as np


# Hàm biểu diễn văn bản thành vector sử dụng mô hình BERT
def text_to_vector_bert(text, model, tokenizer):
    inputs = tokenizer(str(text), return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    vector = torch.mean(outputs.last_hidden_state, dim=1).squeeze(0)  # Sử dụng vector trung bình của các token
    return vector.numpy()


# Tính toán độ tương đồng giữa văn bản đầu vào và tất cả các mẫu dữ liệu
def calculate_similarity_bert(input_text, data_column, model, tokenizer):
    input_vector = text_to_vector_bert(input_text, model, tokenizer)
    similarities = []
    for text in data_column:
        text_vector = text_to_vector_bert(text, model, tokenizer)
        similarity = np.dot(input_vector, text_vector) / (np.linalg.norm(input_vector) * np.linalg.norm(text_vector))
        similarities.append(similarity)
    return similarities


# Hàm lấy các sản phẩm gợi ý dựa trên mô hình BERT
def get_recommendations_bert(input_text, data, model, tokenizer, column, top_n=10, threshold=0.999):
    similarities = calculate_similarity_bert(input_text, data[column].values.astype('U'), model, tokenizer)
    recommendations = []
    for idx, sim in enumerate(similarities):
        if sim < threshold:
            recommendations.append((idx, sim))
    top_recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:top_n]
    for idx, sim in top_recommendations:
        print(f"Similarity: {sim}, Product: {data.iloc[idx][column].encode('utf-8')}")
    return [idx for idx, _ in top_recommendations]


# Hàm tính toán độ tương tự giữa sản phẩm đích và tất cả các sản phẩm khác
def calculate_product_similarity_all_columns(product_info, data, model, tokenizer):
    similarities = []
    for idx, row in data.iterrows():
        if row['id'] != product_info['id']:  # Loại bỏ sản phẩm đích
            similarity_sum = 0
            for column in data.columns:
                if column != 'id':
                    similarity = calculate_similarity_bert(product_info[column], row[column], model, tokenizer)
                    if isinstance(similarity, list):
                        similarity_sum += sum(similarity)
                    else:
                        similarity_sum += similarity
            average_similarity = similarity_sum / (len(data.columns) - 1)  # Loại bỏ cột 'id'
            similarities.append((idx, average_similarity))
    return similarities


# Tính toán độ tương đồng giữa hai vector
def calculate_similarity(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


# Tính toán độ tương tự giữa sản phẩm đích và mỗi sản phẩm khác
def calculate_similarity_for_product(product_row, other_products, model, tokenizer):
    similarities = []
    for _, other_product_row in other_products.iterrows():
        if other_product_row['id'] != product_row['id']:
            similarity_info = {'id': other_product_row['id']}
            sum = 0
            for column in product_row.index:
                if column != 'id':
                    similarity = calculate_similarity(text_to_vector_bert(product_row[column], model, tokenizer),
                                                      text_to_vector_bert(other_product_row[column], model, tokenizer))
                    similarity_info[column] = similarity
                    sum += similarity
            similarity_info["avg"] = sum / (len(product_row.index) - 1)
            similarities.append(similarity_info)
    return similarities


# Hàm lấy các sản phẩm gợi ý dựa trên mô hình BERT và tất cả các cột (trừ cột 'id')
def get_recommendations_based_on_product_all_columns(product_info, data, model, tokenizer, top_n=10, threshold=0.999):
    similarities = calculate_similarity_for_product(product_info, data, model, tokenizer)
    # Sắp xếp danh sách theo cột 'avg' từ lớn đến nhỏ
    top_recommendations = sorted(similarities, key=lambda x: x['avg'], reverse=True)[:top_n]

    # Tạo một danh sách để lưu trữ các ID duy nhất của top N sản phẩm
    top_n = []
    seen_ids = set()
    for product in top_recommendations:
        if product['id'] not in seen_ids:
            top_n.append(product)
            seen_ids.add(product['id'])
        if len(top_n) == top_n:
            break

    for product in top_n:
        print(f"Similarity: {product['avg']}, Product: {data.loc[data['id'] == product['id']]['name']}")
    return [product['id'] for product in top_n]

