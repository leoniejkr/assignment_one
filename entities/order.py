from collections import Counter

class Order:
    def __init__(self, row, col, items):
        self.row = row
        self.col = col
        # items: dict product_type -> qty
        self.items = Counter(items)  # Counter of product type ids