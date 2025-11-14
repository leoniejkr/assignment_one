import random
import time
from typing import List
from greedy import build_greedy_trips, find_warehouse_plan_for_order
from util import simulate_and_score

class BaseOptimizer:
    """Base class for all optimization algorithms with evaluation tracking."""
    
    def __init__(self, data, time_limit=60):
        self.data = data
        self.time_limit = time_limit
        self.num_orders = len(data["orders"])
        self.num_drones = data["num_drones"]
        
        # Track best solution
        self.best_solution = None
        self.best_score = 0
        self.convergence_history = []  # (iteration, score, time)
        
        # Evaluation tracking
        self.num_evaluations = 0
        self.start_time = None
    
    def reset_counters(self):
        """Reset evaluation counters (useful for multiple runs)."""
        self.num_evaluations = 0
        self.start_time = time.time()
    
    def random_solution(self) -> List[int]:
        """Generate random initial solution."""
        return [random.randint(0, self.num_drones - 1) 
                for _ in range(self.num_orders)]
    
    def evaluate(self, solution: List[int]) -> int:
        """
        Evaluate a solution and return score.
        Automatically increments evaluation counter.
        """
        self.num_evaluations += 1
        
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
    
    def check_time_limit(self) -> bool:
        """Check if time limit has been exceeded."""
        if self.start_time is None:
            return False
        return (time.time() - self.start_time) > self.time_limit
    
    def update_best(self, solution: List[int], score: int, iteration: int):
        """Update best solution if current is better."""
        if score > self.best_score:
            self.best_solution = solution.copy()
            self.best_score = score
            
            if self.start_time:
                elapsed = time.time() - self.start_time
            else:
                elapsed = 0
            
            self.convergence_history.append((iteration, self.best_score, elapsed))
            return True
        return False
    
    # =============== NEIGHBORHOOD OPERATORS ===============
    
    def get_neighbors(self, solution: List[int], max_neighbors=100) -> List[List[int]]:
        """
        Generate neighbor solutions with multiple operators.
        Limited to max_neighbors to control evaluation budget.
        """
        neighbors = []
        
        # Operator 1: Swap two orders between drones
        for i in range(len(solution)):
            for j in range(i + 1, min(i + 10, len(solution))):
                if len(neighbors) >= max_neighbors:
                    break
                neighbor = solution.copy()
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbors.append(neighbor)
            if len(neighbors) >= max_neighbors:
                break
        
        # Operator 2: Reassign single order to different drone
        for i in range(min(30, len(solution))):
            for d in range(self.num_drones):
                if len(neighbors) >= max_neighbors:
                    break
                if solution[i] != d:
                    neighbor = solution.copy()
                    neighbor[i] = d
                    neighbors.append(neighbor)
            if len(neighbors) >= max_neighbors:
                break
        
        return neighbors[:max_neighbors]
    
    def mutate(self, solution: List[int]) -> List[int]:
        """Apply random mutation to solution."""
        mutated = solution.copy()
        mutation_type = random.choice(['swap', 'reassign', 'sequence', 'multi_swap'])
        
        if mutation_type == 'swap':
            # Swap two random orders
            if len(mutated) >= 2:
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
        
        elif mutation_type == 'multi_swap':
            # Multiple random swaps
            num_swaps = random.randint(2, 5)
            for _ in range(num_swaps):
                if len(mutated) >= 2:
                    i, j = random.sample(range(len(mutated)), 2)
                    mutated[i], mutated[j] = mutated[j], mutated[i]
        
        return mutated
    
    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Single-point crossover for genetic algorithm."""
        if len(parent1) != len(parent2):
            return parent1.copy()
        
        crossover_point = random.randint(1, len(parent1) - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child
    
    # =============== MAIN OPTIMIZATION METHOD ===============
    
    def optimize(self):
        """
        Main optimization method (to be overridden by subclasses).
        Should return: (best_solution, best_score, convergence_history)
        """
        raise NotImplementedError("Subclasses must implement optimize()")
    
    def get_stats(self) -> dict:
        """Get optimization statistics."""
        return {
            "num_evaluations": self.num_evaluations,
            "best_score": self.best_score,
            "convergence_points": len(self.convergence_history),
            "final_iteration": self.convergence_history[-1][0] if self.convergence_history else 0,
            "time_to_best": self.convergence_history[-1][2] if self.convergence_history else 0,
        }
    
    def print_progress(self, iteration: int, current_score: int, frequency: int = 100):
        """Print progress update at specified frequency."""
        if iteration % frequency == 0:
            stats = f"Iter {iteration:5d} | Current: {current_score:6.0f} | Best: {self.best_score:6.0f} | Evals: {self.num_evaluations:6d}"
            if self.start_time:
                elapsed = time.time() - self.start_time
                stats += f" | Time: {elapsed:5.1f}s"
            print(f"  {stats}")