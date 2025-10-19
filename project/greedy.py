from collections import Counter
from util import distance

from collections import Counter
from util import distance
from file_handling import load_input, write_submission
from util import simulate_and_score

from collections import Counter
from util import distance

# Plan fulfillment from warehouses considering stock only
def find_warehouse_plan_for_order(order, warehouses):
    needed = Counter(order.items)
    # Try a single warehouse
    for wid, w in enumerate(warehouses):
        if all(w.stock[pid] >= qty for pid, qty in needed.items()):
            return [(wid, needed.copy())]
    # Split across closest warehouses
    plan = []
    remaining = needed.copy()
    wdist = sorted(
        [(wid, distance((w.row, w.col), (order.row, order.col))) for wid, w in enumerate(warehouses)],
        key=lambda x: x[1]
    )
    for wid, _ in wdist:
        supply = Counter()
        for pid, qty in list(remaining.items()):
            avail = warehouses[wid].stock[pid]
            if avail > 0:
                take = min(avail, qty)
                supply[pid] = take
                remaining[pid] -= take
                if remaining[pid] == 0:
                    del remaining[pid]
        if supply:
            plan.append((wid, supply))
        if not remaining:
            break
    return plan if not remaining else None

# Assign orders to drones round-robin
def initial_solution(data):
    warehouses = data["warehouses"]
    orders = data["orders"]
    D = data["num_drones"]

    assignments = []
    for oid, order in enumerate(orders):
        plan = find_warehouse_plan_for_order(order, warehouses)
        if plan is None:
            raise RuntimeError(f"Order {oid} cannot be fulfilled.")
        drone_id = oid % D
        assignments.append({"plan": plan, "drone": drone_id})
    return assignments

# Convert order plans to trips respecting drone capacity
def build_greedy_trips(assignments, data):
    D = data["num_drones"]
    max_load = data["max_load"]
    product_weights = data["product_weights"]

    drone_trips = [[] for _ in range(D)]

    for oid, a in enumerate(assignments):
        drone_id = a["drone"]
        plan = a["plan"]
        for wid, items in plan:
            items_left = items.copy()
            while items_left:
                pack = Counter()
                pack_weight = 0
                # Sort by weight descending for first-fit decreasing packing
                for pid, qty in sorted(items_left.items(), key=lambda x: product_weights[x[0]], reverse=True):
                    w = product_weights[pid]
                    max_qty = min(qty, (max_load - pack_weight) // w) 
                    # min
                    #   qty: only deliver what we HAVE to, respect max load
                    #   rest: maximum number of units of product that can fit in the remaining capacity.
                    # max_qty --> how many units of this product we can actually load
                    if max_qty > 0:
                        pack[pid] = max_qty
                        pack_weight += max_qty * w
                if not pack:
                    # No items can fit — should rarely happen unless an item exceeds max_load
                    raise RuntimeError(f"Item too heavy for drone: order {oid}, items {items_left}")
                # Add trip
                drone_trips[drone_id].append({
                    "order_id": oid,
                    "warehouse_id": wid,
                    "pack": pack
                })
                # Reduce remaining
                for pid, q in pack.items():
                    items_left[pid] -= q
                    if items_left[pid] <= 0:
                        del items_left[pid]
    return drone_trips



if __name__ == "__main__":
    from project.file_handling import load_input
    data = load_input("data/input/busy_day.in")

    # Step 1: build initial greedy solution
    sol = initial_solution(data)
    #for oid, a in enumerate(sol):
    #    print(f"Order {oid} --> Drone {a['drone']} | Plan: {a['plan']}")

    # Step 2: convert to drone trips
    drone_trips = build_greedy_trips(sol, data)

    # Step 3: simulate and score the solution
    total_score, order_scores = simulate_and_score(drone_trips, data)

    # Step 4: write submission file
    write_submission(drone_trips, "data/output/greedy_submission.out")

