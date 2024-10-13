import numpy as np
import subprocess
import matplotlib as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

class LanderGA:

    def __init__(self, population_size=15, num_generations=40, crossover_chance=0.7, mutation_rate=0.2, tournament_size=10, elitism_size=2):
        self.population_size = population_size
        self.num_generations = num_generations
        self.gene_size = 4  # PID_heightConst, P_controllerGain, I_controllerGain, D_controllerGain 4 variables
        self.cpp_exe = "./lander_trainer"
        self.crossover_chance = crossover_chance
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_size = elitism_size
        self.weight_range = (1,30)
        self.const_range = (0.001,0.1)

    def initialize_population(self):
        Gain_values = np.random.uniform(low=self.weight_range[0], high=self.weight_range[1], size=(self.population_size, 3))  # uniform for first 3 columns
        height_const_values = np.random.uniform(low=self.const_range[0], high=self.const_range[1], size=(self.population_size, 1)) 
        combined_array = np.concatenate((height_const_values, Gain_values), axis=1)
        return combined_array
    
    def evaluate_individual(self, individual):
        input_string = f"{individual[0]} {individual[1]} {individual[2]} {individual[3]} \n"
        process = subprocess.Popen(self.cpp_exe, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(input=input_string)
        output = stdout.strip().split(',')
        return float(output[-1])

    def evaluate_fitness(self, population):
        fitnesses = np.zeros(self.population_size)
        
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.evaluate_individual, individual): i for i, individual in enumerate(population)}
            
            for future in as_completed(futures):
                i = futures[future]
                fitnesses[i] = future.result()
                print(f"Run: {i+1}/{self.population_size}, Fitness: {fitnesses[i]}")
    
        return fitnesses

    def tournament_selection(self, population, fitnesses):
        selected = np.zeros((self.population_size - self.elitism_size, self.gene_size))
        for i in range(len(selected)):
            tournament_indices = np.random.choice(len(population), self.tournament_size, replace=False)
            tournament_fitnesses = fitnesses[tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
            selected[i] = population[winner_index]
        return selected

    def crossover(self, parents):
        offspring = np.zeros((len(parents), self.gene_size))
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                if np.random.random() < self.crossover_chance:
                    crossover_point = np.random.randint(1, self.gene_size)
                    offspring[i, :crossover_point] = parents[i, :crossover_point]
                    offspring[i, crossover_point:] = parents[i+1, crossover_point:]
                    offspring[i+1, :crossover_point] = parents[i+1, :crossover_point]
                    offspring[i+1, crossover_point:] = parents[i, crossover_point:]
                else:
                    offspring[i] = parents[i]
                    offspring[i+1] = parents[i+1]
            else:
                offspring[i] = parents[i]
        return offspring

    def mutate(self, offspring):
        for i in range(len(offspring)):
            for j in range(self.gene_size):
                if np.random.rand() < self.mutation_rate:
                    if j == 0:  # PID_heightConst
                        offspring[i, j] = np.random.uniform(low=self.const_range[0], high=self.const_range[1])
                    else:  # Gain values
                        # Gaussian mutation with smaller variance for more refined changes
                        offspring[i, j] *= np.random.normal(1, 0.1)  # Reduce standard deviation
        return offspring

    def run(self):
        population = self.initialize_population()
        best_fitnesses = []
        avg_fitnesses = []
        overall_best_individual = None
        overall_best_fitness = float('-inf')

        stagnation_counter = 0  # To track stagnant generations
        last_best_fitness = None  # Track best fitness from the previous generation

        for generation in range(self.num_generations):
            fitnesses = self.evaluate_fitness(population)
            
            # Update overall best
            gen_best_idx = np.argmax(fitnesses)
            gen_best_fitness = fitnesses[gen_best_idx]
            gen_best_individual = population[gen_best_idx]
            
            if gen_best_fitness > overall_best_fitness:
                overall_best_fitness = gen_best_fitness
                overall_best_individual = gen_best_individual.copy()

            print(f"Generation {generation}: Best Fitness = {gen_best_fitness}")
            print(f"Best Individual: {gen_best_individual}")
            print(f"Overall Best Fitness: {overall_best_fitness}")
            print(f"Overall Best Individual: {overall_best_individual}")

            best_fitnesses.append(gen_best_fitness)
            avg_fitnesses.append(np.mean(fitnesses))

            # Check for stagnation
            if last_best_fitness is not None and gen_best_fitness == last_best_fitness:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            if stagnation_counter >= 2:  # Restart if stagnant for 3 generations
                print("Population stagnated, restarting...")
                population = self.initialize_population()
                stagnation_counter = 0

            last_best_fitness = gen_best_fitness

            # Elitism
            elite_indices = np.argsort(fitnesses)[-self.elitism_size:]
            elites = population[elite_indices]

            # Selection
            selected = self.tournament_selection(population, fitnesses)

            # Crossover
            offspring = self.crossover(selected)

            # Mutation
            offspring = self.mutate(offspring)

            # Create new population
            population[:-self.elitism_size] = offspring
            population[-self.elitism_size:] = elites

        return overall_best_individual, overall_best_fitness


if __name__ == "__main__":
    ga = LanderGA()
    best_params, best_fitness = ga.run()
    print(f"Optimization complete. Best parameters: {best_params}, Best fitness: {best_fitness}")
