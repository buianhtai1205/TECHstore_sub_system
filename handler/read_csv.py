import pandas as pd
import sys
import io

def readData():
    # Đọc dữ liệu
    data = pd.read_csv('D://KLTN//TECHstore_sub_system//handler//data.csv')
    data.head()
    return data