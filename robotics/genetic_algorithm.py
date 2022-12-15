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
TRAIN_TIMESTEPS = 40_960
TEST_TIMESTEPS = 2_048
BATCH_SIZE = 2_048
PARALLELS = 0
PARALLEL_TYPE = "process"

BACKUP = False

logging.basicConfig(format="%(asctime)s %(message)s")
ga_logger = logging.getLogger("GA Logger")
ga_logger.setLevel(logging.INFO)


class EndEffectorGA(pygad.GA):
    """
    Genetic algorithm variables:
    - position (x, y, z) (floats)
    - joint type (slide rotating in x or y/hinge rotating in z)
    - size (x, y, z) (floats)
    [POS X, POS Y, POS Z, JOINT TYPE, SIZE X, SIZE Y, SIZE Z]
    """

    POS_SPACE = {"low": 0.015, "high": 0.040, "step": 0.01}
    SIZE_SPACE = {"low": 0.020, "high": 0.040, "step": 0.005}
    JOINT_TYPE_SPACE = [0, 1, 2]

    SINGLE_JOINT_VARIABLES = [float, float, float, int, float, float, float]
    SINGLE_JOINT_SPACE = [
        POS_SPACE,
        POS_SPACE,
        POS_SPACE,
        JOINT_TYPE_SPACE,
        SIZE_SPACE,
        SIZE_SPACE,
        SIZE_SPACE,
    ]
    current_gen_fitness = np.zeros((6,))

    def __init__(
        self,
        num_generations: int,
        num_parents_mating: int,
        population_count: int,
    ) -> None:

        ga_logger.info(
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
            fitness_func=self.dev_fitness_func,
            on_generation=self.callback_gen,
            save_solutions=True,
            # parallel_processing=[PARALLEL_TYPE, PARALLELS],
        )

    @staticmethod
    def callback_gen(ga):
        ga_logger.info(f"Generation {ga.generations_completed} Finished")
        ga_logger.info(
            f"Attributes of the best solution: {ga.best_solution(pop_fitness=ga.current_gen_fitness)}"
        )
        ga_logger.info(f"Generation {ga.generations_completed + 1} Starting")
        ga.save(f"mujoco_ga_instance_gen_{ga.generations_completed}")
        backup_data(ga=ga)
        ga_logger.debug("Saved and downloaded GA instance checkpoint")

    @staticmethod
    def dev_fitness_func(chromosome, idx):
        ga_logger.warning(
            f"You are using the dev fitness function for {idx=}, make sure this is intentional"
        )
        robot_asset_path = f"robot_{idx}"

        end_effector = utils.chromosome_to_end_effector(chromosome, NUM_JOINTS)
        end_effector.build(robot_asset_path)

        ga_logger.debug(f"Building end effector {idx}:")
        ga_logger.debug(end_effector)

        env = TrainEnv(robot_asset_path)
        ga_logger.info(f"Starting training for idx: {idx}")

        env = TrainEnv(robot_asset_path)

        n_steps = 1_000
        for _ in range(n_steps):
            # Random action
            env.render()
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()

    @staticmethod
    def fitness_func(chromosome, idx):
        # TODO: FIXME
        # TODO: Better logging
        start_time = time()
        robot_asset_path = f"robot_{idx}"
        tb_logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
        ga_logger.debug(f"Tensorboard logging at {tb_logdir}")

        end_effector = utils.chromosome_to_end_effector(chromosome, NUM_JOINTS)
        end_effector.build(robot_asset_path)

        ga_logger.debug(f"Building end effector {idx}:")
        ga_logger.debug(end_effector)

        env = TrainEnv(robot_asset_path)
        ga_logger.info(f"Starting training for idx: {idx}")

        try:
            model = PPO.load(
                f"models/end_effector{idx}", env, verbose=0, tensorboard_log=tb_logdir
            )
            ga_logger.debug(f"Loaded {idx} model")

        except Exception as e:
            model = PPO(
                "MultiInputPolicy",
                env,
                verbose=1,
                tensorboard_log=tb_logdir,
                batch_size=BATCH_SIZE,
            )
            ga_logger.info(f"Could not load {idx} model, creating new model")
            ga_logger.info(e)

        model.learn(
            total_timesteps=TRAIN_TIMESTEPS,
            tb_log_name=robot_asset_path,
            reset_num_timesteps=False,
        )
        ga_logger.info(f"Finished training for idx: {idx}")

        obs = env.reset()
        done = False

        total_reward = 0
        for _ in range(TEST_TIMESTEPS):
            obs, reward, done, _ = env.step(model.predict(obs)[0])
            total_reward += reward
            if done:
                ga_logger.debug("Resetting env")
                obs = env.reset()
        fitness = total_reward / TEST_TIMESTEPS

        model.save(f"models/end_effector{idx}")
        env.close()

        EndEffectorGA.current_gen_fitness[idx] = fitness
        ga_logger.info(f"Fitness for {idx}: {fitness}")
        ga_logger.debug(f"Total training time {(time() - start_time)/60}min")

        return fitness


def backup_data(ga):
    if not BACKUP:
        return

    dt_string = datetime.now().strftime("%d.%m.%Y-%H:%M:%S")
    ga_logger.debug(f"Backing up genetic algorithm at {dt_string}")

    gdrive_path = f"/content/gdrive/My Drive/rl_training_v2/{dt_string}_gen_checkpoint"
    os.mkdir(gdrive_path)

    try:
        ga.save("mujoco_ga_instance")
        shutil.copyfile(
            f"/content/mujoco/mujoco_ga_instance_gen_{ga.generations_completed}.pkl",
            f"{gdrive_path}/mujoco_ga_instance_gen_{ga.generations_completed}.pkl",
        )
    except Exception as e:
        ga_logger.error(f"Could not backup checkpoint GA file: {e}")

    for i in range(8):
        shutil.copyfile(
            f"/content/mujoco/models/end_effector{i}.zip",
            f"{gdrive_path}/end_effector{i}.zip",
        )

    shutil.copyfile(
        "/content/mujoco/mujoco_ga_instance.pkl",
        f"{gdrive_path}/mujoco_ga_instance.pkl",
    )

    ga_logger.debug(f"Backed up at {gdrive_path}")


if __name__ == "__main__":
    num_generations = 10
    num_parents_mating = 4
    population_count = 8

    try:
        raise Exception

        ga = pygad.load("mujoco_ga_instance")
        ga_logger.info(
            f"Loaded GA instance, generations completed: {ga.generations_completed}"
        )
    except Exception as e:
        ga_logger.info("Error while loading previous GA instace, creating a new one")
        ga_logger.info(e)
        ga = EndEffectorGA(
            num_generations,
            num_parents_mating,
            population_count,
        )

    try:
        ga.run()
    except KeyboardInterrupt:
        ga_logger.debug("User stopped GA, saving ga instance")
    finally:
        ga.save("mujoco_ga_instance")
        backup_data(ga)

    solution, solution_fitness, solution_idx = ga.best_solution()
    ga_logger.info(f"Parameters of the best solution : {solution}")
    ga_logger.info(f"Fitness value of the best solution = {solution_fitness}")
    ga_logger.info(f"Index of the best solution : {solution_idx}")
