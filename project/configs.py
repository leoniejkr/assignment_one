from optimizers.simulated_annealing import SimulatedAnnealing
from optimizers.tabu_search import TabuSearch
from optimizers.genetic_algorithm import GeneticAlgorithm
from optimizers.hill_climb import HillClimbing

def define_experiments():
    """Define parameter grids for all algorithms."""
    return {
        "HillClimbing": [
            {"class": HillClimbing, "params": {"max_iterations": 1000, "restarts": 3, "time_limit": 60}},
            {"class": HillClimbing, "params": {"max_iterations": 2000, "restarts": 5, "time_limit": 60}},
        ],
        "SimulatedAnnealing": [
            {"class": SimulatedAnnealing, "params": {"initial_temp": 10000, "cooling_rate": 0.95, "iterations_per_temp": 100, "min_temp": 1, "time_limit": 60}},
            {"class": SimulatedAnnealing, "params": {"initial_temp": 20000, "cooling_rate": 0.99, "iterations_per_temp": 200, "min_temp": 1, "time_limit": 60}},
        ],
        "TabuSearch": [
            {"class": TabuSearch, "params": {"tabu_tenure": 20, "max_iterations": 1000, "time_limit": 60}},
        ],
        "GeneticAlgorithm": [
            {"class": GeneticAlgorithm, "params": {"population_size": 100, "generations": 100, "mutation_rate": 0.2, "crossover_rate": 0.8, "time_limit": 60}},
        ],
    }
