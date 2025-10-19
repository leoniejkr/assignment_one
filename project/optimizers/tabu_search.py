from collections import deque
import time
from typing import List
from optimizers.base_optimizer import BaseOptimizer


class TabuSearch(BaseOptimizer):
    """Tabu Search algorithm"""
    
    def __init__(self, data, tabu_tenure=20, max_iterations=1000, time_limit=60):
        super().__init__(data, time_limit)
        self.tabu_tenure = tabu_tenure
        self.max_iterations = max_iterations
        self.tabu_list = deque(maxlen=tabu_tenure)
    
    def solution_hash(self, solution: List[int]) -> str:
        """Create hash of solution for tabu list"""
        return ''.join(map(str, solution))
    
    def optimize(self):
        """Run tabu search"""
        print(f"\n{'='*60}")
        print(f"TABU SEARCH (tenure={self.tabu_tenure}, iter={self.max_iterations})")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Initial solution
        current = self.random_solution()
        current_score = self.evaluate(current)
        
        self.best_solution = current.copy()
        self.best_score = current_score
        
        iteration = 0
        
        while iteration < self.max_iterations:
            if time.time() - start_time > self.time_limit:
                break
            
            # Generate neighbors
            neighbors = []
            for _ in range(50):  # Generate 50 random neighbors
                neighbor = self.mutate(current)
                neighbors.append(neighbor)
            
            # Find best non-tabu neighbor
            best_neighbor = None
            best_neighbor_score = float('-inf')
            
            for neighbor in neighbors:
                neighbor_hash = self.solution_hash(neighbor)
                neighbor_score = self.evaluate(neighbor)
                
                # Accept if not tabu OR if aspiration criteria met (better than best)
                if (neighbor_hash not in self.tabu_list or 
                    neighbor_score > self.best_score):
                    
                    if neighbor_score > best_neighbor_score:
                        best_neighbor = neighbor
                        best_neighbor_score = neighbor_score
            
            if best_neighbor is not None:
                # Move to best neighbor
                current = best_neighbor
                current_score = best_neighbor_score
                
                # Add to tabu list
                self.tabu_list.append(self.solution_hash(current))
                
                # Update best
                if current_score > self.best_score:
                    self.best_solution = current.copy()
                    self.best_score = current_score
                    self.convergence_history.append((
                        iteration,
                        self.best_score,
                        time.time() - start_time
                    ))
                    print(f"  New best at iteration {iteration}: {self.best_score}")
            
            iteration += 1
            
            if iteration % 100 == 0:
                print(f"  Iteration {iteration}, current={current_score}, best={self.best_score}")
        
        runtime = time.time() - start_time
        print(f"\n✓ Best score: {self.best_score}")
        print(f"✓ Runtime: {runtime:.2f}s")
        
        return self.best_solution, self.best_score, self.convergence_history
