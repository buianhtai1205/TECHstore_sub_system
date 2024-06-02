from typing import List

from fastapi import FastAPI, HTTPException
from handler.read_csv import readData
from handler.model import load_model_BERT, load_tokenizer_BERT, load_model_USE
from handler.recommend_BERT import get_recommendations_bert, get_recommendations_based_on_product_all_columns
from handler.recommend_USE import get_recommendation_USE
from handler.connect import update_data_to_csv
import handler.utils as utils

app = FastAPI()
update_data_to_csv()
data = readData()

# model_USE = load_model_USE()
model_BERT = load_model_BERT()
tokenizer_BERT = load_tokenizer_BERT()


@app.get("/")
async def root():
    return {"message": "Hello World"}


# @app.get("/use")
# async def recommend_based_USE():
#     # Sử dụng hàm get_recommendation_use
#     input_text = "50MP + 2MP + 2MP Triple Rear & 13MP Front Camera"
#     column_train = "camera"
#     recommended_products = get_recommendation_USE(model_USE, input_text, data, column_train)
#     print(f"Recommended products based on '{column_train}':")
#     print(recommended_products)
#     return {"Data": f"{recommended_products}"}


@app.post("/bert")
async def recommend_based_BERT(productIds : List[int]):
    # Kiểm tra xem danh sách ID sản phẩm có rỗng không
    if not productIds:
        raise HTTPException(status_code=400, detail="Empty list of product IDs")

    product_info = utils.get_product_by_id(data, productIds[0])
    if product_info is not None:
        print("Product found:")
        print(product_info)
        top_recommendations = get_recommendations_based_on_product_all_columns(product_info, data, model_BERT,
                                                                               tokenizer_BERT)
        return {"Data": top_recommendations}
    else:
        print("Product not found.")
        return {"Data": []}

    # # Sử dụng mô hình BERT để tìm sản phẩm gợi ý
    # input_text = "50MP + 2MP + 2MP Triple Rear & 13MP Front Camera"
    # column_train = "camera"
    # top_recommendations = get_recommendations_bert(input_text, data, model_BERT, tokenizer_BERT, column_train)
    # return {"Data": f"{top_recommendations}"}