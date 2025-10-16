import numpy as np
from entities.warehouse import Warehouse
from entities.order import Order


# to check if the file is read correctly
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
