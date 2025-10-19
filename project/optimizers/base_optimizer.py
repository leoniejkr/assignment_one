import random
from typing import List

from greedy import build_greedy_trips, find_warehouse_plan_for_order
from util import simulate_and_score

class BaseOptimizer:
    """Base class for all optimization algorithms"""
    
    def __init__(self, data, time_limit=60):
        self.data = data
        self.time_limit = time_limit
        self.num_orders = len(data["orders"])
        self.num_drones = data["num_drones"]
        
        # Track best solution
        self.best_solution = None
        self.best_score = 0
        self.convergence_history = []  # (iteration, score, time)
        
    def random_solution(self) -> List[int]:
        """Generate random initial solution"""
        return [random.randint(0, self.num_drones - 1) 
                for _ in range(self.num_orders)]
    
    def evaluate(self, solution: List[int]) -> int:
        """Evaluate a solution and return score"""
        try:
            assignments = []
            for oid, drone_id in enumerate(solution):
                order = self.data["orders"][oid]
                plan = find_warehouse_plan_for_order(order, self.data["warehouses"])
                if plan is None:
                    return 0  # Infeasible
                assignments.append({"plan": plan, "drone": drone_id})
            
            trips = build_greedy_trips(assignments, self.data)
            score, _ = simulate_and_score(trips, self.data)
            return score
        except:
            return 0
    
    def get_neighbors(self, solution: List[int]) -> List[List[int]]:
        """Generate neighbor solutions (swap operations)"""
        neighbors = []
        
        # Operator 1: Swap two orders between drones
        for i in range(len(solution)):
            for j in range(i + 1, min(i + 20, len(solution))):  # Limit neighborhood
                neighbor = solution.copy()
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbors.append(neighbor)
        
        # Operator 2: Change one order to different drone
        for i in range(min(50, len(solution))):  # Limit to 50 for speed
            for d in range(self.num_drones):
                if solution[i] != d:
                    neighbor = solution.copy()
                    neighbor[i] = d
                    neighbors.append(neighbor)
        
        return neighbors
    
    def mutate(self, solution: List[int]) -> List[int]:
        """Apply random mutation to solution"""
        mutated = solution.copy()
        mutation_type = random.choice(['swap', 'reassign', 'sequence'])
        
        if mutation_type == 'swap':
            # Swap two random orders
            i, j = random.sample(range(len(mutated)), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]
            
        elif mutation_type == 'reassign':
            # Reassign one order to random drone
            i = random.randint(0, len(mutated) - 1)
            mutated[i] = random.randint(0, self.num_drones - 1)
            
        elif mutation_type == 'sequence':
            # Reassign sequence of orders to same drone
            if len(mutated) > 5:
                start = random.randint(0, len(mutated) - 5)
                end = start + random.randint(2, 5)
                new_drone = random.randint(0, self.num_drones - 1)
                for i in range(start, min(end, len(mutated))):
                    mutated[i] = new_drone
        
        return mutated