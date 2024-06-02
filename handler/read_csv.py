import pandas as pd

def readData():
    # Đọc dữ liệu
    data = pd.read_csv('/home/anhtai/PycharmProjects/fastApiProject/handler/data.csv')
    data.head()
    return data