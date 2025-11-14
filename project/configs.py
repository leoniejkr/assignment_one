from optimizers.simulated_annealing import SimulatedAnnealing
from optimizers.tabu_search import TabuSearch
from optimizers.genetic_algorithm import GeneticAlgorithm
from optimizers.hill_climb import HillClimbing

def define_experiments():
    """
    Define parameter grids with COMPARABLE computational budgets.
    
    Strategy: All algorithms get approximately 5000 solution evaluations
    within 60 seconds time limit.
    """
    
    # EVALUATION BUDGET: ~5000 evaluations per run
    # This ensures fair comparison across algorithms
    
    return {
        "HillClimbing": [
            {
                "class": HillClimbing, 
                "params": {
                    "max_iterations": 250,      # 250 iters * ~20 neighbors = 5000 evals
                    "restarts": 5,              # 5 restarts for diversity
                    "time_limit": 60
                }
            },
            {
                "class": HillClimbing, 
                "params": {
                    "max_iterations": 500,      # Alternative: fewer restarts, more iterations
                    "restarts": 2,
                    "time_limit": 60
                }
            },
        ],
        
        "SimulatedAnnealing": [
            {
                "class": SimulatedAnnealing, 
                "params": {
                    "initial_temp": 10000,
                    "cooling_rate": 0.95,       # ~90 temperature levels
                    "iterations_per_temp": 50,  # 90 * 50 = 4500 evals
                    "min_temp": 1,
                    "time_limit": 60
                }
            },
            {
                "class": SimulatedAnnealing, 
                "params": {
                    "initial_temp": 5000,       # Different cooling schedule
                    "cooling_rate": 0.98,       # ~115 temperature levels
                    "iterations_per_temp": 40,  # 115 * 40 = 4600 evals
                    "min_temp": 1,
                    "time_limit": 60
                }
            },
        ],
        
        "TabuSearch": [
            {
                "class": TabuSearch, 
                "params": {
                    "tabu_tenure": 20,
                    "max_iterations": 5000,     # Direct: 5000 iterations
                    "neighborhood_size": 50,    # Evaluate 50 neighbors per iteration (added param)
                    "time_limit": 60
                }
            },
            {
                "class": TabuSearch, 
                "params": {
                    "tabu_tenure": 50,          # Longer memory
                    "max_iterations": 5000,
                    "neighborhood_size": 30,    # Smaller neighborhood for speed
                    "time_limit": 60
                }
            },
        ],
        
        "GeneticAlgorithm": [
            {
                "class": GeneticAlgorithm, 
                "params": {
                    "population_size": 50,      # 50 * 100 = 5000 evals
                    "generations": 100,
                    "mutation_rate": 0.2,
                    "crossover_rate": 0.8,
                    "time_limit": 60
                }
            },
            {
                "class": GeneticAlgorithm, 
                "params": {
                    "population_size": 100,     # Alternative: larger pop, fewer gens
                    "generations": 50,          # 100 * 50 = 5000 evals
                    "mutation_rate": 0.15,
                    "crossover_rate": 0.9,
                    "time_limit": 60
                }
            },
        ],
    }


def define_experiments_quick_test():
    """
    Smaller configuration for quick testing.
    Budget: ~500 evaluations per run
    """
    return {
        "HillClimbing": [
            {"class": HillClimbing, "params": {"max_iterations": 25, "restarts": 2, "time_limit": 30}},
        ],
        "SimulatedAnnealing": [
            {"class": SimulatedAnnealing, "params": {"initial_temp": 1000, "cooling_rate": 0.90, "iterations_per_temp": 20, "min_temp": 1, "time_limit": 30}},
        ],
        "TabuSearch": [
            {"class": TabuSearch, "params": {"tabu_tenure": 10, "max_iterations": 500, "time_limit": 30}},
        ],
        "GeneticAlgorithm": [
            {"class": GeneticAlgorithm, "params": {"population_size": 25, "generations": 20, "mutation_rate": 0.2, "crossover_rate": 0.8, "time_limit": 30}},
        ],
    }