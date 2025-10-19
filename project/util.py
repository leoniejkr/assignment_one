import math
from entities.warehouse import Warehouse
from collections import Counter


# utility function to calculate the euclidean distance rounded up to the nearest integer
def distance(a, b):
    return math.ceil(math.hypot(a[0]-b[0], a[1]-b[1]))

# utility function to deep copy a list of Warehouse objects
def stock_warehouses(warehouses):
    return [Warehouse(w.row, w.col, list(w.stock)) for w in warehouses]

# utility function to calculate total weight of items given as a Counter {product_type: qty}
def total_weight_of_items(items_counter, product_weights):
    return sum(product_weights[p]*q for p,q in items_counter.items())

# score simulation of drone trips
def simulate_and_score(drone_trips, data):
    """
    Simulate drone activity from trips and compute official Hash Code score.
    Returns total_score, per_order_scores.
    """
    T = data["deadline"]
    product_weights = data["product_weights"]
    warehouses = data["warehouses"]
    orders = data["orders"]
    D = data["num_drones"]

    # Drone states
    drone_positions = [ (warehouses[0].row, warehouses[0].col) for _ in range(D) ]
    drone_time = [0 for _ in range(D)]

    # Track order deliveries
    remaining = [Counter(o.items) for o in orders]
    completion_turn = [None for _ in range(len(orders))]

    for d_id, trips in enumerate(drone_trips):
        for trip in trips:
            wid = trip["warehouse_id"]
            oid = trip["order_id"]
            order = orders[oid]
            warehouse = warehouses[wid]
            pack = trip["pack"]

            # 1 Load step
            dist_load = math.ceil(distance(drone_positions[d_id], (warehouse.row, warehouse.col)))
            drone_time[d_id] += dist_load + 1
            drone_positions[d_id] = (warehouse.row, warehouse.col)

            # 2 Deliver step
            dist_deliver = math.ceil(distance(drone_positions[d_id], (order.row, order.col)))
            drone_time[d_id] += dist_deliver + 1
            drone_positions[d_id] = (order.row, order.col)

            # Update order remaining items
            for pid, qty in pack.items():
                remaining[oid][pid] -= qty
                if remaining[oid][pid] <= 0:
                    del remaining[oid][pid]

            # If order now complete, record completion time
            if not remaining[oid] and completion_turn[oid] is None:
                completion_turn[oid] = drone_time[d_id]

    # Compute total score
    total_score = 0
    order_scores = []
    for oid, t in enumerate(completion_turn):
        if t is None or t > T:
            score = 0
        else:
            score = math.ceil((T - t) / T * 100)
        order_scores.append(score)
        total_score += score

    # print("\n Simulation Summary:")
    # print(f"  Total score: {total_score}")
    # print(f"  Completed orders: {sum(1 for s in order_scores if s>0)} / {len(orders)}")

    return total_score, order_scores
