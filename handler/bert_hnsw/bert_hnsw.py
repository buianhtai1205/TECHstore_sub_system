import numpy as np
import torch
import nmslib
import os

# Hàm để chuẩn hóa các vector
def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

# Hàm để chuyển đổi một thuộc tính thành vector
def encode_attribute(attribute, model, tokenizer):
    inputs = tokenizer(attribute, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Hàm để chuyển đổi sản phẩm thành vector đa chiều (kết hợp các vector thuộc tính)
def encode_product(product, model, tokenizer):
    vectors = []
    for prop in ['name', 'brand_name', 'characteristic_name', 'description', 'attribute_values']:
        vectors.append(encode_attribute(str(product[prop]), model, tokenizer))
    return np.concatenate(vectors)

def create_vectors_and_hnsw_index(data, model, tokenizer):
    # Chuyển đổi tất cả các sản phẩm thành vector và chuẩn hóa chúng
    product_vectors = np.array([encode_product(product, model, tokenizer) for index, product in data.iterrows()])
    product_vectors = normalize(product_vectors)
    np.save('/home/anhtai/PycharmProjects/fastApiProject/handler/bert_hnsw/product_vectors.npy', product_vectors)

    # Tạo index HNSW
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(product_vectors)
    index.createIndex({'post': 2}, print_progress=True)

    # Lưu trữ index hnsw
    index.saveIndex('/home/anhtai/PycharmProjects/fastApiProject/handler/bert_hnsw/product_index.hnsw')

def load_vectors_and_hnsw_index():
    # Đọc các vector sản phẩm đã lưu trữ
    product_vectors = np.load('/home/anhtai/PycharmProjects/fastApiProject/handler/bert_hnsw/product_vectors.npy')

    # Tải lại index HNSW
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.loadIndex('/home/anhtai/PycharmProjects/fastApiProject/handler/bert_hnsw/product_index.hnsw')

    return product_vectors, index

def get_recommendations_bert_hnsw(data, product, model, tokenizer):
    # Kiểm tra nếu tệp vector và hnsw index đã tồn tại
    if (not os.path.exists('/home/anhtai/PycharmProjects/fastApiProject/handler/bert_hnsw/product_vectors.npy')
            or not os.path.exists('/home/anhtai/PycharmProjects/fastApiProject/handler/bert_hnsw/product_index.hnsw')):
        # Tạo vector và hnsw index nếu chưa tồn tại
        print("Creating hnsw index...")
        create_vectors_and_hnsw_index(data, model, tokenizer)
    else:
        print("Loading hnsw index...")

    # Tải lại các vector và hnsw index
    product_vectors, index = load_vectors_and_hnsw_index()

    # Tạo vector truy vấn
    print("Product query vectors create...")
    query_product = {
        "name": product["name"],
        "brand_name": product["brand_name"],
        "characteristic_name": product["characteristic_name"],
        "description": product["description"],
        "attribute_values": product["attribute_values"]
    }

    query_vector = encode_product(query_product, model, tokenizer).astype('float32').reshape(1, -1)
    query_vector = normalize(query_vector)  # Chuẩn hóa vector truy vấn

    # Tìm kiếm vector truy vấn trong index HNSW
    k = 10  # Số lượng vector gần nhất cần tìm
    top_n = []
    ids, distances = index.knnQuery(query_vector, k)

    # Lưu trữ danh sách các tên sản phẩm
    product_ids = data['id'].tolist()
    product_names = data['name'].tolist()

    # In ra các sản phẩm tương ứng với các chỉ số được tìm thấy
    print("Kết quả tìm kiếm:")
    for i in range(k):
        top_n.append(product_ids[ids[i]])
        print(f"Vector {i + 1}: ID {ids[i]}, Similarity {distances[i]}")
        print("ID:", product_ids[ids[i]])
        print("Tên sản phẩm:", product_names[ids[i]])
        print("Vector sản phẩm:", product_vectors[ids[i]])
        print()

    return top_n