import numpy as np
import pandas as pd

def get_recommendation_USE(embed, input_text, data, column_train, top_n=10):

    # Biểu diễn mỗi mẫu dữ liệu dưới dạng vector
    data_embeddings = embed(data[column_train].values.astype('U'))

    # Biểu diễn dữ liệu đầu vào dưới dạng vector
    input_text_embedding = embed([input_text])

    # Tính toán độ tương đồng giữa dữ liệu đầu vào và tất cả các mẫu dữ liệu
    similarities = np.inner(input_text_embedding, data_embeddings)[0]
    print(similarities)
    print(data[column_train][0])

    # Loại bỏ độ tương đồng là 1 (tương ứng với sản phẩm đầu vào)
    similarities = np.where(similarities > 0.9999, 0, similarities)

    # Sắp xếp các mẫu dữ liệu theo độ tương đồng giảm dần và lấy ra top_n mẫu có độ tương đồng cao nhất
    top_indices = np.argsort(similarities)[::-1][:top_n]

    # Tạo DataFrame chứa thông tin các mẫu gợi ý và độ tương đồng tương ứng
    recommendation_df = pd.DataFrame(columns=[column_train, 'Similarity'])
    for idx in top_indices:
        recommendation_df.loc[len(recommendation_df)] = [data.iloc[idx][column_train], similarities[idx]]
    print(recommendation_df)

    return [idx for idx, _ in top_indices]