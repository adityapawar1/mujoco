from gym.core import gym
import numpy as np
import pygad
import utils
from stable_baselines3.ppo.ppo import PPO

from envs.train_env import TrainEnv

NUM_JOINTS = 6
TRAIN_TIMESTEPS = 100_000
TEST_TIMESTEPS = 5_000


class EndEffectorGA(pygad.GA):
    """
    Genetic algorithm variables:
    - position (x, y, z) (floats)
    - joint type (hinge/slide)
    - size (x, y, z) (floats)
    - attachment location (x, y, z) (one of the 3)
    """

    FLOAT_SPACE = {"low": 0.0001, "high": 0.050, "step": 0.0005}
    JOINT_TYPE_SPACE = [1, 2]
    ATTACHMENT_LOC_SPACE = [0, 1, 2]
    SINGLE_JOINT_VARIABLES = [float, float, float, int, float, float, float, int]
    SINGLE_JOINT_SPACE = [
        FLOAT_SPACE,
        FLOAT_SPACE,
        FLOAT_SPACE,
        JOINT_TYPE_SPACE,
        FLOAT_SPACE,
        FLOAT_SPACE,
        FLOAT_SPACE,
        ATTACHMENT_LOC_SPACE,
    ]

    def __init__(
        self,
        num_generations: int,
        num_parents_mating: int,
        population_count: int,
    ) -> None:

        print(
            f"Creating genetic algorithm: Number of joints: {NUM_JOINTS}, Number of generations: {num_generations}, Parents mating: {num_parents_mating}, Population count: {population_count}"
        )
        gene_type = np.tile(self.SINGLE_JOINT_VARIABLES, NUM_JOINTS).tolist()
        gene_space = np.tile(self.SINGLE_JOINT_SPACE, NUM_JOINTS).tolist()

        super().__init__(
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            num_genes=len(gene_type),
            sol_per_pop=population_count,
            gene_type=gene_type,
            gene_space=gene_space,
            fitness_func=self.fitness_func,
            on_mutation=self.on_mutation,
            on_generation=self.callback_gen,
        )

    @staticmethod
    def callback_gen(ga):
        print("Generation : ", ga.generations_completed)
        print("Fitness of the best solution :", ga.best_solution()[1])

    @staticmethod
    def fitness_func(chromosome, idx):
        # TODO: FIXME
        end_effector = utils.chromosome_to_end_effector(chromosome, NUM_JOINTS)
        end_effector.build()

        env = TrainEnv()
        model = PPO("MultiInputPolicy", env, verbose=1)
        print(f"Starting training for idx: {idx}")
        model.learn(total_timesteps=TRAIN_TIMESTEPS)
        print(f"Finished training for idx: {idx}")

        obs = env.reset()
        done = False

        total_reward = 0
        for _ in range(TEST_TIMESTEPS):
            obs, reward, done, _ = env.step(model.predict(obs)[0])
            total_reward += reward
            if done:
                print("Resetting env")
                obs = env.reset()

        print(end_effector.ga_string())
        print(f"Fitness for {idx}: {total_reward}")
        return total_reward

    @staticmethod
    def on_mutation(chromosome, ga):
        random_gene_index = np.random.choice(range(len(ga.SINGLE_JOINT_SPACE)))
        random_gene_space = ga.SINGLE_JOINT_SPACE[random_gene_index]

        new_gene_value = None
        if isinstance(random_gene_space, list):
            new_gene_value = np.random.choice(random_gene_space)
        elif isinstance(random_gene_space, dict):
            # FLOAT_SPACE = {"low": 0.0001, "high": 0.050, "step": 0.0005}
            min = random_gene_space["low"]
            max = random_gene_space["high"]
            step = random_gene_space["step"]
            new_gene_value = np.random.choice(range(min, max, step))
        else:
            raise ValueError(
                f"Got gene with erroneous space: Index: {random_gene_index}, Chromosome: {chromosome}"
            )

        chromosome[random_gene_index] = new_gene_value
        return chromosome


if __name__ == "__main__":
    num_generations = 10
    num_parents_mating = 2
    population_count = 5

    ga = EndEffectorGA(
        num_generations,
        num_parents_mating,
        population_count,
    )

    ga.run()
    ga.plot_fitness()

    solution, solution_fitness, solution_idx = ga.best_solution()
    print(f"Parameters of the best solution : {solution}")
    print(f"Fitness value of the best solution = {solution_fitness}")
    print(f"Index of the best solution : {solution_idx}")
