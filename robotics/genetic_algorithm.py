import pygad


class EndEffectorGA(pygad.GA):
    def __init__(self, num_generations, num_parents_mating, population_size, num_genes) -> None:
        super().__init__(
                num_generations=num_generations,
                num_parents_mating=num_parents_mating,
                sol_per_pop=population_size,
                num_genes=num_genes,
                fitness_func=self.fitness_func
            )

    def fitness_func(end_effector, idx):
        raise NotImplementedError
