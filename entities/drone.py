from collections import Counter

class DroneState:
    def __init__(self, drone_id, row, col, max_load):
        self.id = drone_id
        self.row = row
        self.col = col
        self.time = 0
        self.load = Counter()    # product_type -> qty
        self.load_weight = 0
        self.max_load = max_load