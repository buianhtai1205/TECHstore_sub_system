from typing import List

from fastapi import FastAPI, HTTPException
from handler.read_csv import readData
from handler.model import load_model_BERT, load_tokenizer_BERT
from handler.recommend_BERT import get_recommendations_based_on_product_all_columns
from handler.connect import update_data_to_csv
from handler.bert_faiss.bert_faiss import get_recommendations_bert_faiss
from handler.bert_faiss_ann.bert_faiss_ann import get_recommendations_bert_faiss_ann
from handler.bert_hnsw.bert_hnsw import get_recommendations_bert_hnsw
import handler.utils as utils

app = FastAPI()
# update_data_to_csv()
data = readData()

# Load model BERT
model_BERT = load_model_BERT()
tokenizer_BERT = load_tokenizer_BERT()


@app.get("/")
async def root():
    return {"message": "Hello World"}

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

@app.post("/bert-faiss")
async def recommend_based_BERT_FAISS(productIds : List[int]):
    # Kiểm tra xem danh sách ID sản phẩm có rỗng không
    if not productIds:
        raise HTTPException(status_code=400, detail="Empty list of product IDs")

    product_info = utils.get_product_by_id(data, productIds[0])
    if product_info is not None:
        print("Product found:")
        print(product_info)
        top_recommendations = get_recommendations_bert_faiss(data, product_info, model_BERT, tokenizer_BERT)
        return {"Data": top_recommendations}
    else:
        print("Product not found.")
        return {"Data": []}

@app.post("/bert-faiss-ann")
async def recommend_based_BERT_FAISS_ANN(productIds : List[int]):
    # Kiểm tra xem danh sách ID sản phẩm có rỗng không
    if not productIds:
        raise HTTPException(status_code=400, detail="Empty list of product IDs")

    product_info = utils.get_product_by_id(data, productIds[0])
    if product_info is not None:
        print("Product found:")
        print(product_info)
        top_recommendations = get_recommendations_bert_faiss_ann(data, product_info, model_BERT, tokenizer_BERT)
        return {"Data": top_recommendations}
    else:
        print("Product not found.")
        return {"Data": []}

@app.post("/bert-hnsw")
async def recommend_based_BERT_HNSW(productIds : List[int]):
    # Kiểm tra xem danh sách ID sản phẩm có rỗng không
    if not productIds:
        raise HTTPException(status_code=400, detail="Empty list of product IDs")

    product_info = utils.get_product_by_id(data, productIds[0])
    if product_info is not None:
        print("Product found:")
        print(product_info)
        top_recommendations = get_recommendations_bert_hnsw(data, product_info, model_BERT, tokenizer_BERT)
        return {"Data": top_recommendations}
    else:
        print("Product not found.")
        return {"Data": []}