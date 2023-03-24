import csv
from data_analysis.models import Order


with open('daf.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)  # Skip the header row
    for row in csvreader:
        obj = Order()
        obj.order_id = row[0]
        obj.date = row[1]
        obj.user_id = row[2]
        obj.total_purchase = row[3]
        obj.latitude = row[4]
        obj.longitude = row[5]
        obj.save()
