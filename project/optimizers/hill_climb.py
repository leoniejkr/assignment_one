import time
from typing import List
from optimizers.base_optimizer import BaseOptimizer

class HillClimbing(BaseOptimizer):
    """Hill-Climbing with random restart"""

    def __init__(self, data, max_iterations=1000, restarts=5, time_limit=60, use_greedy_init=True):
        super().__init__(data, time_limit, use_greedy_init)
        self.max_iterations = max_iterations
        self.restarts = restarts
    
    def optimize(self):
        """Run hill-climbing with random restarts"""
        print(f"\n{'='*60}")
        print(f"HILL CLIMBING (iter={self.max_iterations}, restarts={self.restarts})")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        for restart in range(self.restarts):
            print(f"\nRestart {restart + 1}/{self.restarts}")
            
            # Or use a greedy initial solution
            if self.use_greedy_init:
                print("  Using greedy initial solution")
                current = self.get_initial_solution()
            else:
                current = self.random_solution()

            current_score = self.evaluate(current)
            
            iteration = 0
            improvements = 0
            
            while iteration < self.max_iterations:
                if time.time() - start_time > self.time_limit:
                    break
                
                # Get neighbors
                neighbors = self.get_neighbors(current)
                
                # Find best neighbor (steepest ascent)
                best_neighbor = None
                best_neighbor_score = current_score
                
                for neighbor in neighbors[:100]:  # Limit evaluations
                    score = self.evaluate(neighbor)
                    if score > best_neighbor_score:
                        best_neighbor = neighbor
                        best_neighbor_score = score
                
                # Move to better neighbor or stop
                if best_neighbor is not None:
                    current = best_neighbor
                    current_score = best_neighbor_score
                    improvements += 1
                    
                    # Track best
                    if current_score > self.best_score:
                        self.best_solution = current.copy()
                        self.best_score = current_score
                        self.convergence_history.append((
                            iteration, 
                            self.best_score, 
                            time.time() - start_time
                        ))
                else:
                    print(f"  Local optimum at iteration {iteration}, score={current_score}")
                    break
                
                iteration += 1
            
            print(f"  Final score: {current_score} ({improvements} improvements)")
        
        runtime = time.time() - start_time
        print(f"\n✓ Best score found: {self.best_score}")
        print(f"✓ Runtime: {runtime:.2f}s")
        
        return self.best_solution, self.best_score, self.convergence_history

