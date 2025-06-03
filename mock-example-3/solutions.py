import csv
from collections import defaultdict

def top_n_customers_by_spend(tsv_path, n=5):
    totals = defaultdict(float)
    with open(tsv_path, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            cid = row['customer_id']
            amt = float(row['amount'])
            totals[cid] += amt
    # Sort customers by spend descending
    sorted_customers = sorted(totals.items(), key=lambda x: x[1], reverse=True)
    return sorted_customers[:n]

# On the snippet above, totals are:
# C004: 3500.0, C001: 2000.0, C003: 1800.0, C002: 1050.0, C005: 750.0, C006: 200.0
# Top 5: [("C004",3500.0), ("C001",2000.0), ("C003",1800.0), ("C002",1050.0), ("C005",750.0)]
