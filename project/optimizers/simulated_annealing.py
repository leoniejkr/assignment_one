import time
from optimizers.base_optimizer import BaseOptimizer
import math
import random

class SimulatedAnnealing(BaseOptimizer):
    """Simulated Annealing algorithm"""
    
    def __init__(self, data, initial_temp=10000, cooling_rate=0.95, 
                 iterations_per_temp=100, min_temp=1, time_limit=60, use_greedy_init=True):
        super().__init__(data, time_limit, use_greedy_init)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.iterations_per_temp = iterations_per_temp
        self.min_temp = min_temp
    
    def optimize(self):
        """Run simulated annealing"""
        print(f"\n{'='*60}")
        print(f"SIMULATED ANNEALING")
        print(f"T0={self.initial_temp}, α={self.cooling_rate}, iter/temp={self.iterations_per_temp}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Or use a greedy initial solution
        if self.use_greedy_init:
            print("  Using greedy initial solution")
            current = self.get_initial_solution()
        else:
            current = self.random_solution()

        current_score = self.evaluate(current)
        
        self.best_solution = current.copy()
        self.best_score = current_score
        
        temperature = self.initial_temp
        iteration = 0
        acceptances = 0
        
        while temperature > self.min_temp:
            if time.time() - start_time > self.time_limit:
                break
            
            for _ in range(self.iterations_per_temp):
                # Generate neighbor
                neighbor = self.mutate(current)
                neighbor_score = self.evaluate(neighbor)
                
                # Calculate acceptance probability
                delta = neighbor_score - current_score
                
                if delta > 0:
                    # Better solution - always accept
                    accept = True
                else:
                    # Worse solution - accept with probability
                    probability = math.exp(delta / temperature)
                    accept = random.random() < probability
                
                if accept:
                    current = neighbor
                    current_score = neighbor_score
                    acceptances += 1
                    
                    # Track best
                    if current_score > self.best_score:
                        self.best_solution = current.copy()
                        self.best_score = current_score
                        self.convergence_history.append((
                            iteration,
                            self.best_score,
                            time.time() - start_time
                        ))
                        print(f"  New best at T={temperature:.1f}: {self.best_score}")
                
                iteration += 1
            
            # Cool down
            temperature *= self.cooling_rate
        
        runtime = time.time() - start_time
        acceptance_rate = acceptances / iteration if iteration > 0 else 0
        
        print(f"\n✓ Best score: {self.best_score}")
        print(f"✓ Runtime: {runtime:.2f}s")
        print(f"✓ Acceptance rate: {acceptance_rate:.2%}")
        
        return self.best_solution, self.best_score, self.convergence_history

