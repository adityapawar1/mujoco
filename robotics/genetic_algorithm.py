from time import time
import logging
from datetime import datetime

import shutil
import os

import numpy as np
import pygad
import utils
from stable_baselines3.ppo.ppo import PPO

from envs.train_env import TrainEnv

NUM_JOINTS = 6
TRAIN_TIMESTEPS = 40_000
TEST_TIMESTEPS = 2_000
BATCH_SIZE = 10_000
THREADS = 3


class EndEffectorGA(pygad.GA):
    """
    Genetic algorithm variables:
    - position (x, y, z) (floats)
    - joint type (hinge/slide)
    - size (x, y, z) (floats)
    - attachment location (x, y, z) (one of the 3)
    """

    FLOAT_SPACE = {"low": 0.0001, "high": 0.01, "step": 0.005}
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
    current_gen_fitness = np.zeros((6,))

    def __init__(
        self,
        num_generations: int,
        num_parents_mating: int,
        population_count: int,
    ) -> None:

        logging.info(
            f"Creating genetic algorithm: Number of joints: {NUM_JOINTS}, Number of generations: {num_generations}, Parents mating: {num_parents_mating}, Population count: {population_count}"
        )

        self.current_gen_fitness = np.zeros((population_count,))
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
            save_solutions=True,
            parallel_processing=["thread", THREADS],
        )

    @staticmethod
    def callback_gen(ga):
        logging.info(f"Generation {ga.generations_completed} Finished")
        logging.info(
            f"Attributes of the best solution: {ga.best_solution(pop_fitness=ga.current_gen_fitness)}"
        )
        logging.info(f"Generation {ga.generations_completed + 1} Starting")
        ga.save(f"mujoco_ga_instance_gen_{ga.generations_completed}")
        backup_data(ga=ga)
        logging.debug("Saved and downloaded GA instance checkpoint")

    @staticmethod
    def fitness_func(chromosome, idx):
        # TODO: FIXME
        # TODO: Better logging
        start_time = time()
        robot_asset_path = f"robot_{idx}"
        tb_logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
        logging.debug(f"Tensorboard logging at {tb_logdir}")

        end_effector = utils.chromosome_to_end_effector(chromosome, NUM_JOINTS)
        end_effector.build(robot_asset_path)

        logging.debug(f"Building end effector {idx}:")
        logging.debug(end_effector)

        env = TrainEnv(robot_asset_path)
        logging.info(f"Starting training for idx: {idx}")

        try:
            model = PPO.load(
                f"models/end_effector{idx}", env, verbose=0, tensorboard_log=tb_logdir
            )
            logging.debug(f"Loaded {idx} model")
        except Exception as e:
            model = PPO(
                "MultiInputPolicy",
                env,
                verbose=1,
                tensorboard_log=tb_logdir,
                batch_size=BATCH_SIZE,
            )
            logging.info(f"Could not load {idx} model, creating new model")
            logging.info(e)

        model.learn(
            total_timesteps=TRAIN_TIMESTEPS,
            tb_log_name=robot_asset_path,
            reset_num_timesteps=False,
        )
        logging.info(f"Finished training for idx: {idx}")

        obs = env.reset()
        done = False

        total_reward = 0
        for _ in range(TEST_TIMESTEPS):
            obs, reward, done, _ = env.step(model.predict(obs)[0])
            total_reward += reward
            if done:
                logging.debug("Resetting env")
                obs = env.reset()
        fitness = total_reward / TEST_TIMESTEPS

        model.save(f"models/end_effector{idx}")
        env.close()

        EndEffectorGA.current_gen_fitness[idx] = fitness
        logging.info(f"Fitness for {idx}: {fitness}")
        logging.debug(f"Total training time {(time() - start_time)/60}min")

        return fitness


def backup_data(ga):
    dt_string = datetime.now().strftime("%d.%m.%Y-%H:%M:%S")
    logging.debug(f"Backing up genetic algorithm at {dt_string}")

    gdrive_path = f"/content/gdrive/My Drive/rl_training_v2/{dt_string}_gen_checkpoint"
    os.mkdir(gdrive_path)

    try:
        ga.save("mujoco_ga_instance")
        shutil.copyfile(
            f"/content/mujoco/mujoco_ga_instance_gen_{ga.generations_completed}.pkl",
            f"{gdrive_path}/mujoco_ga_instance_gen_{ga.generations_completed}.pkl",
        )
    except Exception as e:
        logging.error(f"Could not backup checkpoint GA file: {e}")

    for i in range(8):
        shutil.copyfile(
            f"/content/mujoco/models/end_effector{i}.zip",
            f"{gdrive_path}/end_effector{i}.zip",
        )

    shutil.copyfile(
        "/content/mujoco/mujoco_ga_instance.pkl",
        f"{gdrive_path}/mujoco_ga_instance.pkl",
    )

    logging.debug(f"Backed up at {gdrive_path}")


if __name__ == "__main__":
    num_generations = 10
    num_parents_mating = 4
    population_count = 8

    ga: pygad.GA = pygad.load(
        "/Users/adityapawar/Documents/code/Projects/polygence/models/mujoco_ga_instance_gen_10"
    )

    ga.plot_fitness()

    try:
        ga = pygad.load("mujoco_ga_instance")
        logging.info(
            f"Loaded GA instance, generations completed: {ga.generations_completed}"
        )
    except Exception as e:
        logging.info("Error while loading previous GA instace, creating a new one")
        logging.info(e)
        ga = EndEffectorGA(
            num_generations,
            num_parents_mating,
            population_count,
        )

    try:
        ga.run()
    except KeyboardInterrupt:
        logging.debug("User stopped GA, saving ga instance")
    finally:
        ga.save("mujoco_ga_instance")
        backup_data(ga)

    solution, solution_fitness, solution_idx = ga.best_solution()
    logging.info(f"Parameters of the best solution : {solution}")
    logging.info(f"Fitness value of the best solution = {solution_fitness}")
    logging.info(f"Index of the best solution : {solution_idx}")
