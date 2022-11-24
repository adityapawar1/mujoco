from time import time
from gym.core import gym
import numpy as np
import pygad
import utils
from stable_baselines3.ppo.ppo import PPO

from envs.train_env import TrainEnv

NUM_JOINTS = 6
TRAIN_TIMESTEPS = 30_000
TEST_TIMESTEPS = 2_000


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
        print(f"Single joint space shape: {np.array(self.SINGLE_JOINT_SPACE)}")

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
            on_generation=self.callback_gen,
            parallel_processing=["thread", 2],
        )

    @staticmethod
    def callback_gen(ga: pygad.GA):
        print("============= GENERATION FINISHED =============")
        print("Generation: ", ga.generations_completed)
        print("Attributes of the best solution :", ga.best_solution())

    @staticmethod
    def fitness_func(chromosome, idx):
        # TODO: FIXME
        # TODO: Better logging
        start_time = time()
        end_effector = utils.chromosome_to_end_effector(chromosome, NUM_JOINTS)
        # print(f"Chromosome: {chromosome}, Shape: {chromosome.shape}")
        end_effector.build()
        print(f"Building end effector {idx}:")
        print(end_effector)

        env = TrainEnv()
        model = PPO("MultiInputPolicy", env, verbose=0)
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

        model.save(f"models/end_effector{idx}")
        env.close()

        print(f"Fitness for {idx}: {total_reward}")
        print(f"Total training time {(time() - start_time)/60}min")
        return total_reward

    @staticmethod
    def on_mutation(ga, chromosome):
        random_gene_index = np.random.choice(
            range(len(EndEffectorGA.SINGLE_JOINT_SPACE))
        )
        random_gene_space = EndEffectorGA.SINGLE_JOINT_SPACE[random_gene_index]

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
    num_parents_mating = 4
    population_count = 8

    try:
        ga = pygad.load("mujoco_ga_instance")
        print("Loaded GA instance")
    except Exception as e:
        print("Error while loading previous GA instace, creating a new one")
        print(e)

        ga = EndEffectorGA(
            num_generations,
            num_parents_mating,
            population_count,
        )

    try:
        ga.run()
    except Exception as e:
        print("Error while running genetic algorithm, saving ga instance")
        print(e)

    ga.save("mujoco_ga_instance")
    ga.plot_fitness()

    solution, solution_fitness, solution_idx = ga.best_solution()
    print(f"Parameters of the best solution : {solution}")
    print(f"Fitness value of the best solution = {solution_fitness}")
    print(f"Index of the best solution : {solution_idx}")
