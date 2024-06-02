import csv
import mysql.connector
from handler.analysic_data import preprocess_text

def get_database_connection():
    return mysql.connector.connect(
        host='127.0.0.1',
        user='root',
        password='root',
        database='TECHstore'
    )


# update data from database to csv file to read fast
def update_data_to_csv():
    connection = get_database_connection()

    # handle basic data
    cursorBasic = connection.cursor()
    # Query to fetch basic data
    queryBasic = 'select p.id,p.name,c.name as category_name,b.name as brand_name,ch.name as characteristic_name,p.product_status,p.mass,p.design,p.accessories,p.launch_time,p.description,pp.sale_price,pp.price,t.ram,t.storage_capacity as rom,t.color from product p inner join category c on p.category_id = c.id and c.is_deleted = 0 and p.is_deleted = 0 inner join brand b on p.brand_id = b.id and b.is_deleted = 0 inner join characteristic ch on p.characteristic_id = ch.id inner join product_price pp on p.id = pp.product_id and pp.is_deleted = 0 inner join type t on pp.type_id = t.id'
    cursorBasic.execute(queryBasic)
    resultBasic = cursorBasic.fetchall()

    # handle attribute data
    cursorAttribute = connection.cursor()
    # Query to fetch attribute data
    queryAttribute = 'select pa.product_id as id, pa.name, pa.value from product_attribute pa'
    cursorAttribute.execute(queryAttribute)
    resultAttribute = cursorAttribute.fetchall()

    # Close the database connection
    connection.close()

    # Open the CSV file to write data with UTF-8 encoding
    with open('/home/anhtai/PycharmProjects/fastApiProject/handler/data.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Create header with new column for attribute values
        header = [i[0] for i in cursorBasic.description] + ['attribute_values']
        writer.writerow(header)  # Write header
        # Process and write data
        for rowBasic in resultBasic:
            # Get values of the 'value' column from the fetched attribute data
            values = []
            for attribute_row in resultAttribute:
                if attribute_row[0] == rowBasic[0]:  # Compare IDs
                    values.append(attribute_row[2])  # Get attribute value

            # Join values together and add to the data row
            row = list(rowBasic) + [', '.join(values)]

            # Apply preprocessing to the 'description' column
            row[10] = preprocess_text(row[10])
            writer.writerow(row)