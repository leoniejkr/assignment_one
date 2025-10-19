import time
import random
from typing import List
from optimizers.base_optimizer import BaseOptimizer


class GeneticAlgorithm(BaseOptimizer):
    """Genetic Algorithm using DEAP library"""
    
    def __init__(self, data, population_size=100, generations=50, 
                 mutation_rate=0.2, crossover_rate=0.8, time_limit=60):
        super().__init__(data, time_limit)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
    
    def optimize(self):
        """Run genetic algorithm"""
        try:
            from deap import base, creator, tools, algorithms
            
            print(f"\n{'='*60}")
            print(f"GENETIC ALGORITHM")
            print(f"Pop={self.population_size}, Gen={self.generations}, Mut={self.mutation_rate}")
            print(f"{'='*60}")
            
            start_time = time.time()
            
            # Setup DEAP
            if hasattr(creator, "FitnessMax"):
                del creator.FitnessMax
            if hasattr(creator, "Individual"):
                del creator.Individual
            
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)
            
            toolbox = base.Toolbox()
            toolbox.register("attr_drone", random.randint, 0, self.num_drones - 1)
            toolbox.register("individual", tools.initRepeat, creator.Individual,
                           toolbox.attr_drone, n=self.num_orders)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            
            def eval_func(individual):
                score = self.evaluate(individual)
                return (score,)
            
            toolbox.register("evaluate", eval_func)
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", tools.mutUniformInt, 
                           low=0, up=self.num_drones - 1, indpb=0.1)
            toolbox.register("select", tools.selTournament, tournsize=3)
            
            # Create population
            pop = toolbox.population(n=self.population_size)
            hof = tools.HallOfFame(1)
            
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", lambda x: sum(v[0] for v in x) / len(x))
            stats.register("max", lambda x: max(v[0] for v in x))
            
            # Run evolution
            for gen in range(self.generations):
                if time.time() - start_time > self.time_limit:
                    break
                
                # Select next generation
                offspring = toolbox.select(pop, len(pop))
                offspring = list(map(toolbox.clone, offspring))
                
                # Crossover
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < self.crossover_rate:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                
                # Mutation
                for mutant in offspring:
                    if random.random() < self.mutation_rate:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values
                
                # Evaluate
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                
                # Replace population
                pop[:] = offspring
                hof.update(pop)
                
                # Track convergence
                record = stats.compile(pop)
                if hof[0].fitness.values[0] > self.best_score:
                    self.best_score = hof[0].fitness.values[0]
                    self.best_solution = list(hof[0])
                    self.convergence_history.append((
                        gen,
                        self.best_score,
                        time.time() - start_time
                    ))
                
                print(f"  Gen {gen}: avg={record['avg']:.0f}, max={record['max']:.0f}")
            
            runtime = time.time() - start_time
            print(f"\n✓ Best score: {self.best_score}")
            print(f"✓ Runtime: {runtime:.2f}s")
            
            return self.best_solution, self.best_score, self.convergence_history
            
        except ImportError:
            print("ERROR: DEAP not installed. Run: pip install deap")
            return None, 0, []

