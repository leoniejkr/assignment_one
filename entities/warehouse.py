

class Warehouse:
    def __init__(self, row, col, stock):
        self.row = row
        self.col = col
        self.stock = list(stock)  # mutable