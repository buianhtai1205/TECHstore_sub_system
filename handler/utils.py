# Function to find product info by ID in the data list
def get_product_by_id(data, product_id):
    product = data.loc[data['id'] == product_id]
    if not product.empty:
        return product.iloc[0]
    else:
        return None