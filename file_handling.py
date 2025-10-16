import numpy as np
from entities.warehouse import Warehouse
from entities.order import Order

# -----------------------------------------------------------
# input file parsing
def load_input(filename):
    with open(filename, 'r') as f:
        data = [line.strip() for line in f.readlines() if line.strip() != ""]
    idx = 0
    rows, cols, D, T, max_load = map(int, data[idx].split()); idx += 1
    P = int(data[idx]); idx += 1
    product_weights = list(map(int, data[idx].split())); idx += 1
    W = int(data[idx]); idx += 1
    warehouses = []
    for _ in range(W):
        r, c = map(int, data[idx].split()); idx += 1
        stock = list(map(int, data[idx].split())); idx += 1
        warehouses.append(Warehouse(r, c, stock))
    C = int(data[idx]); idx += 1
    orders = []
    for _ in range(C):
        r, c = map(int, data[idx].split()); idx += 1
        L = int(data[idx]); idx += 1
        items = list(map(int, data[idx].split())); idx += 1
        orders.append(Order(r, c, items))
    return {
        "rows": rows, "cols": cols, "num_drones": D, "deadline": T, "num_products": P,
        "max_load": max_load, "product_weights": product_weights,
        "warehouses": warehouses, "orders": orders
    }

# TESSSST example usage:
if __name__ == "__main__":
    data = load_input("data/input/busy_day.in")

    print("Grid size:", data["rows"], "x", data["cols"])
    print("Drones:", data["num_drones"], "Deadline:", data["deadline"], "Max load:", data["max_load"])
    print("Products:", data["num_products"], "weights:", data["product_weights"])

    print("\nWarehouses:")
    for i, w in enumerate(data["warehouses"]):
        print(f"  Warehouse {i}: Location=({w.row},{w.col}), Stock={w.stock}")

    print("\nOrders:")
    for i, o in enumerate(data["orders"]):
        print(f"  Order {i}: Deliver to ({o.row},{o.col}), Items={o.items}")




# -----------------------------------------------------------
# Output submission file generation

def generate_commands_from_trips(drone_trips):
    """
    Convert simulated drone_trips into valid Hash Code output commands.
    Each trip in drone_trips[d_id] is a dict with:
      - order_id
      - warehouse_id
      - pack: Counter({product_type: qty})
    Returns: list of strings (commands)
    """
    commands = []
    for d_id, trips in enumerate(drone_trips):
        for trip in trips:
            wid = trip["warehouse_id"]
            oid = trip["order_id"]
            pack = trip["pack"]
            # 1) LOAD commands (for each product type)
            for pid, qty in pack.items():
                commands.append(f"{d_id} L {wid} {pid} {qty}")
            # 2) DELIVER commands (for each product type)
            for pid, qty in pack.items():
                commands.append(f"{d_id} D {oid} {pid} {qty}")
    return commands


def write_submission(drone_trips, filename="submission.out"):
    """
    Generate output file following the required Hash Code 2016 format.
    """
    commands = generate_commands_from_trips(drone_trips)
    with open(filename, "w") as f:
        f.write(f"{len(commands)}\n")
        for cmd in commands:
            f.write(cmd + "\n")
    print(f"Submission file '{filename}' written with {len(commands)} commands.")
