from file_handling import load_input, write_submission
from optimizers.hill_climing import HillClimbing
from optimizers.simulated_annealing import SimulatedAnnealing
from optimizers.tabu_search import TabuSearch
from optimizers.genetic_algorithm import GeneticAlgorithm


if __name__ == "__main__":

    data = load_input("project/data/input/busy_day.in")

    # Test each algorithm
    algorithms = [
        HillClimbing(data, max_iterations=500, restarts=3, time_limit=30),
        SimulatedAnnealing(data, initial_temp=10000, cooling_rate=0.95, time_limit=30),
        TabuSearch(data, tabu_tenure=20, max_iterations=500, time_limit=30),
        GeneticAlgorithm(data, population_size=100, generations=50, time_limit=30)
    ]
    
    results = []
    for algo in algorithms:
        solution, score, history = algo.optimize()
        if solution:
            results.append({
                "name": algo.__class__.__name__,
                "score": score,
                "history": history
            })
    
    # Show comparison
    print("\n" + "="*60)
    print("ALGORITHM COMPARISON")
    print("="*60)
    for r in sorted(results, key=lambda x: x["score"], reverse=True):
        print(f"{r['name']:<25} Score: {r['score']}")