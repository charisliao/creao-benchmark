import csv
from collections import defaultdict
from datetime import datetime

def revenue_by_category_for_month(csv_path, year, month):
    totals = defaultdict(float)
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = datetime.strptime(row['order_date'], "%Y-%m-%d")
            if date.year == year and date.month == month:
                cat = row['category']
                price = float(row['unit_price'])
                qty = int(row['quantity'])
                totals[cat] += price * qty
    return dict(totals)

# If you run revenue_by_category_for_month("orders.csv", 2025, 5), you get:
# {
#   "A": (25.00*2) + (30.00*3) = 165.00,
#   "B": (15.00*1) + (15.00*2) = 45.00
# }
