from optimizers.simulated_annealing import SimulatedAnnealing
from optimizers.tabu_search import TabuSearch
from optimizers.genetic_algorithm import GeneticAlgorithm
from optimizers.hill_climb import HillClimbing

def define_experiments():
    """
    UPDATED: Configurations optimized for LARGE problems (1250 orders).
    
    Key changes:
    1. Use greedy initialization (not random) - starts at ~72,000 instead of ~60,000
    2. Increase computational budget significantly
    3. Balance exploration vs exploitation
    4. Longer time limits for proper convergence
    """
    
    return {
        "HillClimbing": [
            {
                "class": HillClimbing, 
                "params": {
                    "max_iterations": 100,      # With 1250 orders, each iteration is expensive
                    "restarts": 3,              # Multiple restarts for diversity
                    "time_limit": 180,          # 3 minutes
                    "use_greedy_init": True     # START FROM GREEDY!
                }
            },
            {
                "class": HillClimbing, 
                "params": {
                    "max_iterations": 200,      # More iterations, fewer restarts
                    "restarts": 1,
                    "time_limit": 180,
                    "use_greedy_init": True
                }
            },
        ],
        
        "SimulatedAnnealing": [
            {
                "class": SimulatedAnnealing, 
                "params": {
                    "initial_temp": 100000,     # MUCH higher for large problems
                    "cooling_rate": 0.99,       # Slower cooling
                    "iterations_per_temp": 30,  # ~230 temps * 30 = 6900 iterations
                    "min_temp": 1,
                    "time_limit": 180,
                    "use_greedy_init": True     # START FROM GREEDY!
                }
            },
            {
                "class": SimulatedAnnealing, 
                "params": {
                    "initial_temp": 50000,      # Alternative: faster cooling
                    "cooling_rate": 0.97,       # ~115 temps
                    "iterations_per_temp": 50,  # 115 * 50 = 5750 iterations
                    "min_temp": 1,
                    "time_limit": 180,
                    "use_greedy_init": True
                }
            },
        ],
        
        "TabuSearch": [
            {
                "class": TabuSearch, 
                "params": {
                    "tabu_tenure": 50,          # Longer memory for large problems
                    "max_iterations": 5000,     # Many iterations
                    "neighborhood_size": 30,    # Smaller neighborhoods (large problem)
                    "time_limit": 180,
                    "use_greedy_init": True     # START FROM GREEDY!
                }
            },
            {
                "class": TabuSearch, 
                "params": {
                    "tabu_tenure": 100,         # Even longer memory
                    "max_iterations": 3000,     # Fewer iterations, more thorough
                    "neighborhood_size": 50,
                    "time_limit": 180,
                    "use_greedy_init": True
                }
            },
        ],
        
        "GeneticAlgorithm": [
            {
                "class": GeneticAlgorithm, 
                "params": {
                    "population_size": 30,      # Smaller pop for large problems
                    "generations": 150,         # More generations
                    "mutation_rate": 0.3,       # Higher mutation for exploration
                    "crossover_rate": 0.8,
                    "time_limit": 180,
                    "use_greedy_init": True     # START FROM GREEDY!
                }
            },
            {
                "class": GeneticAlgorithm, 
                "params": {
                    "population_size": 50,      # Larger population
                    "generations": 100,         # Fewer generations
                    "mutation_rate": 0.2,
                    "crossover_rate": 0.9,
                    "time_limit": 180,
                    "use_greedy_init": True
                }
            },
        ],
    }


def define_experiments_aggressive():
    """
    AGGRESSIVE: For users who want MAXIMUM quality and have time.
    Expected runtime: ~5-10 minutes per algorithm run.
    Use this if you want the absolute best results.
    """
    
    return {
        "HillClimbing": [
            {
                "class": HillClimbing, 
                "params": {
                    "max_iterations": 500,
                    "restarts": 5,
                    "time_limit": 600,          # 10 minutes!
                    "use_greedy_init": True
                }
            },
        ],
        
        "SimulatedAnnealing": [
            {
                "class": SimulatedAnnealing, 
                "params": {
                    "initial_temp": 200000,     # Very hot start
                    "cooling_rate": 0.995,      # Very slow cooling
                    "iterations_per_temp": 50,
                    "min_temp": 1,
                    "time_limit": 600,
                    "use_greedy_init": True
                }
            },
        ],
        
        "TabuSearch": [
            {
                "class": TabuSearch, 
                "params": {
                    "tabu_tenure": 100,
                    "max_iterations": 20000,    # Many iterations
                    "neighborhood_size": 50,
                    "time_limit": 600,
                    "use_greedy_init": True
                }
            },
        ],
        
        "GeneticAlgorithm": [
            {
                "class": GeneticAlgorithm, 
                "params": {
                    "population_size": 50,
                    "generations": 300,
                    "mutation_rate": 0.25,
                    "crossover_rate": 0.85,
                    "time_limit": 600,
                    "use_greedy_init": True
                }
            },
        ],
    }


def define_experiments_quick_test():
    """
    Quick test configuration for debugging.
    Budget: ~100-200 evaluations per run, 30 seconds each.
    """
    return {
        "HillClimbing": [
            {"class": HillClimbing, "params": {
                "max_iterations": 20, "restarts": 1, "time_limit": 30, "use_greedy_init": True
            }},
        ],
        "SimulatedAnnealing": [
            {"class": SimulatedAnnealing, "params": {
                "initial_temp": 10000, "cooling_rate": 0.90, "iterations_per_temp": 10, 
                "min_temp": 1, "time_limit": 30, "use_greedy_init": True
            }},
        ],
        "TabuSearch": [
            {"class": TabuSearch, "params": {
                "tabu_tenure": 20, "max_iterations": 200, "neighborhood_size": 20,
                "time_limit": 30, "use_greedy_init": True
            }},
        ],
        "GeneticAlgorithm": [
            {"class": GeneticAlgorithm, "params": {
                "population_size": 15, "generations": 15, "mutation_rate": 0.2, 
                "crossover_rate": 0.8, "time_limit": 30, "use_greedy_init": True
            }},
        ],
    }


# ============================================================================
# EVALUATION BUDGET ANALYSIS
# ============================================================================

def estimate_evaluations():
    """
    Estimate number of evaluations for each configuration.
    Useful for understanding computational budget.
    """
    
    configs = define_experiments()
    
    print("\n" + "="*80)
    print("ESTIMATED EVALUATION BUDGET (with greedy init)")
    print("="*80)
    
    estimates = {
        "HillClimbing": lambda p: p["max_iterations"] * p["restarts"] * 20,  # ~20 neighbors/iter
        "SimulatedAnnealing": lambda p: int((p["initial_temp"] / (1 - p["cooling_rate"])) * p["iterations_per_temp"]),
        "TabuSearch": lambda p: p["max_iterations"] * p["neighborhood_size"],
        "GeneticAlgorithm": lambda p: p["population_size"] * p["generations"]
    }
    
    for algo_name, setups in configs.items():
        print(f"\n{algo_name}:")
        for i, setup in enumerate(setups, 1):
            params = setup["params"]
            evals = estimates[algo_name](params)
            print(f"  Config {i}: ~{evals:,} evaluations (time_limit={params['time_limit']}s)")


if __name__ == "__main__":
    estimate_evaluations()