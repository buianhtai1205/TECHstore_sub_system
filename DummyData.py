from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import requests
from typing import List
import random

app = FastAPI()

# Conversion rate from INR to VND
INR_TO_VND = 304.67

# List of colors to randomly assign
colors = ["Đen", "Xanh", "Trắng"]

# Dictionary to map brand names to their IDs
brand_map = {
    "Apple": 1,
    "Samsung": 9,
    "Oppo": 10,
    "Xiaomi": 11,
    "Vivo": 12,
    "Realme": 13,
    "Nokia": 14,
    "Mobell": 15,
    "itel": 16,
    "Masstel": 17,
    "Other": 33
}

# List of known brands
known_brands = list(brand_map.keys())


def convert_price_to_vnd(price_in_inr):
    price_numeric = float(price_in_inr.replace("₹", "").replace(",", ""))
    price_in_vnd = price_numeric * INR_TO_VND
    return round(price_in_vnd)


def extract_ram_and_storage(ram_storage_str):
    parts = ram_storage_str.split(',')
    ram = parts[0].replace('RAM', '').strip()
    storage_capacity = parts[1].replace('inbuilt', '').strip() if len(parts) > 1 else ""
    return ram, storage_capacity


def get_brand_id(model_name):
    for brand in known_brands:
        if brand.lower() in model_name.lower():
            return brand_map[brand]
    return brand_map["Other"]


def format_data(record):
    ram, storage_capacity = extract_ram_and_storage(record["ram"])
    brand_id = get_brand_id(record["model"])
    camera_value = record["camera"]
    if isinstance(camera_value, float):
        camera_value = str(camera_value)

    # Handle different data types
    for key, value in record.items():
        if pd.isna(value) or value is None or value == "":
            record[key] = ""  # Replace NaN, None, or empty strings with an empty string
        elif isinstance(value, (int, float)):
            record[key] = str(value)  # Convert numbers to strings

    # Replace 'nan' string with an empty string for 'value' field
    if 'value' in record and record['value'] == 'nan':
        record['value'] = ''

    formatted_data = {
        "name": record["model"],
        "description": "DATA DUMMY",
        "design": "Kim loại",
        "dimension": "Kích thước",
        "mass": 1.5,
        "launchTime": 2023,
        "accessories": "Sạc, Tai nghe",
        "productStatus": 100,
        "lstProductImageUrl": ["https://techstore2023.s3.ap-southeast-1.amazonaws.com/images/171223939403533dd75d6-a634-4517-a6fd-e0a4d4714694-samsung-galaxy-s24-ultra-xam-1.jpg"],
        "brandId": brand_id,  # assuming a default brandId
        "characteristicId": 1,  # assuming a default characteristicId
        "categoryId": 1,  # assuming a default categoryId
        "lstProductTypeAndPrice": [
            {
                "ram": ram,
                "storageCapacity": storage_capacity,
                "color": random.choice(colors),
                "price": convert_price_to_vnd(record["price"]),
                "salePrice": convert_price_to_vnd(record["price"]),
                "quantity": 100,
                "depotId": 1  # assuming a default depotId
            }
        ],
        "lstProductAttribute": [
            {"name": "monitor", "value": record["display"]},
            {"name": "operatingSystem", "value": record["os"]},
            {"name": "rearCamera",
             "value": camera_value.split('&')[-1].strip() if '&' in camera_value else camera_value},
            {"name": "frontCamera",
             "value": camera_value.split('&')[-1].strip() if '&' in camera_value else camera_value},
            {"name": "chip", "value": record["processor"]},
            {"name": "sim", "value": record["sim"]},
            {"name": "battery", "value": record["battery"]},
            {"name": "charging",
             "value": record["battery"].split('with')[-1].strip() if 'with' in record["battery"] else ""},
            {"name": "networkSupport", "value": record["sim"]}
        ]
    }
    return formatted_data


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/upload_csv")
async def upload_csv():
    data = pd.read_csv('D:\\KLTN\\TECHstore_sub_system\\handler\\smartphones.csv', dtype=str)
    records = data.to_dict(orient='records')

    api_url = "http://localhost:8080/api/manage/product/create"
    headers = {
        'accept': '*/*',
        'Authorization': 'Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIwMTIzNDU2Nzg5IiwiaWF0IjoxNzE4ODA2MjAyLCJleHAiOjE3MTg4MDgwMDJ9.CXmxLCYQxUNBNLoVd5PAI0r6eTLOhro4ktyG_9nQjw4',
        'Content-Type': 'application/json'
    }

    for record in records:
        formatted_data = format_data(record)
        print(formatted_data)
        response = requests.post(api_url, json=formatted_data, headers=headers)
        if response.status_code != 200:
            # If response is not successful, print an error message and continue to the next record
            print(f"Failed to insert data for record: {formatted_data}")

    return {"message": "Data inserted successfully"}


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='127.0.0.1', port=8000)
