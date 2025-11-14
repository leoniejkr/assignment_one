"""
Improved greedy solution that considers spatial locality and order priorities.
"""
from collections import Counter
import numpy as np
from util import distance
from file_handling import load_input, write_submission
from util import simulate_and_score

def find_warehouse_plan_for_order(order, warehouses):
    """Find which warehouses can fulfill an order (unchanged from original)."""
    needed = Counter(order.items)
    
    # Try single warehouse first
    for wid, w in enumerate(warehouses):
        if all(w.stock[pid] >= qty for pid, qty in needed.items()):
            return [(wid, needed.copy())]
    
    # Split across closest warehouses
    plan = []
    remaining = needed.copy()
    wdist = sorted(
        [(wid, distance((w.row, w.col), (order.row, order.col))) 
         for wid, w in enumerate(warehouses)],
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


def spatial_clustering_assignment(data, num_clusters=None):
    """
    Assign orders to drones using spatial clustering.
    Orders close to each other go to the same drone.
    """
    orders = data["orders"]
    warehouses = data["warehouses"]
    D = data["num_drones"]
    
    if num_clusters is None:
        num_clusters = D
    
    # Extract order locations
    order_locations = np.array([[o.row, o.col] for o in orders])
    
    # Simple k-means clustering (manual implementation to avoid sklearn dependency)
    # Initialize centroids randomly from order locations
    np.random.seed(42)
    centroid_indices = np.random.choice(len(orders), size=min(num_clusters, len(orders)), replace=False)
    centroids = order_locations[centroid_indices].astype(float)
    
    # K-means iterations
    for _ in range(20):
        # Assign orders to nearest centroid
        distances = np.array([[np.linalg.norm(loc - cent) for cent in centroids] 
                             for loc in order_locations])
        assignments = np.argmin(distances, axis=1)
        
        # Update centroids
        new_centroids = np.array([order_locations[assignments == k].mean(axis=0) 
                                  if (assignments == k).sum() > 0 else centroids[k]
                                  for k in range(num_clusters)])
        
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    # Map clusters to drones (cluster k → drone k % D)
    drone_assignments = [assignments[oid] % D for oid in range(len(orders))]
    
    # Build solution format
    solution_assignments = []
    for oid, drone_id in enumerate(drone_assignments):
        order = orders[oid]
        plan = find_warehouse_plan_for_order(order, warehouses)
        if plan is None:
            raise RuntimeError(f"Order {oid} cannot be fulfilled.")
        solution_assignments.append({"plan": plan, "drone": drone_id})
    
    return solution_assignments


def priority_based_assignment(data):
    """
    Assign orders to drones based on urgency (distance to delivery).
    Closer orders = higher priority = assigned to less loaded drones.
    """
    orders = data["orders"]
    warehouses = data["warehouses"]
    D = data["num_drones"]
    
    # Calculate order "urgency" = min distance to any warehouse
    order_urgency = []
    for oid, order in enumerate(orders):
        min_dist = min(distance((order.row, order.col), (w.row, w.col)) for w in warehouses)
        order_urgency.append((oid, min_dist))
    
    # Sort by urgency (ascending = closer orders first)
    order_urgency.sort(key=lambda x: x[1])
    
    # Track drone workload (number of orders assigned)
    drone_load = [0] * D
    drone_assignments = {}
    
    # Assign orders to least loaded drone
    for oid, _ in order_urgency:
        least_loaded_drone = min(range(D), key=lambda d: drone_load[d])
        drone_assignments[oid] = least_loaded_drone
        drone_load[least_loaded_drone] += 1
    
    # Build solution format
    solution_assignments = []
    for oid in range(len(orders)):
        order = orders[oid]
        drone_id = drone_assignments[oid]
        plan = find_warehouse_plan_for_order(order, warehouses)
        if plan is None:
            raise RuntimeError(f"Order {oid} cannot be fulfilled.")
        solution_assignments.append({"plan": plan, "drone": drone_id})
    
    return solution_assignments


def build_greedy_trips(assignments, data):
    """Convert order assignments to drone trips (unchanged from original)."""
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
                
                # First-fit decreasing bin packing
                for pid, qty in sorted(items_left.items(), 
                                      key=lambda x: product_weights[x[0]], 
                                      reverse=True):
                    w = product_weights[pid]
                    max_qty = min(qty, (max_load - pack_weight) // w)
                    
                    if max_qty > 0:
                        pack[pid] = max_qty
                        pack_weight += max_qty * w
                
                if not pack:
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
    data = load_input("data/input/busy_day.in")
    
    print("\n" + "="*60)
    print("TESTING IMPROVED GREEDY STRATEGIES")
    print("="*60)
    
    # Test 1: Spatial clustering
    print("\n1. Spatial Clustering Assignment:")
    sol1 = spatial_clustering_assignment(data)
    trips1 = build_greedy_trips(sol1, data)
    score1, _ = simulate_and_score(trips1, data)
    print(f"   Score: {score1}")
    write_submission(trips1, "data/output/greedy_spatial.out")
    
    # Test 2: Priority-based
    print("\n2. Priority-Based Assignment:")
    sol2 = priority_based_assignment(data)
    trips2 = build_greedy_trips(sol2, data)
    score2, _ = simulate_and_score(trips2, data)
    print(f"   Score: {score2}")
    write_submission(trips2, "data/output/greedy_priority.out")
    
    print(f"\n✓ Best greedy score: {max(score1, score2)}")