from time import time
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
    current_gen_fitness = np.zeros((6,))

    def __init__(
        self,
        num_generations: int,
        num_parents_mating: int,
        population_count: int,
    ) -> None:

        print(
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
            parallel_processing=["thread", THREADS],
        )

    @staticmethod
    def callback_gen(ga):
        print(
            f"============= GENERATION {ga.generations_completed} FINISHED ============="
        )
        print(
            "Attributes of the best solution :",
            ga.best_solution(pop_fitness=ga.current_gen_fitness),
        )
        print(
            f"============= STARTING GENERATION {ga.generations_completed+1} ============="
        )
        ga.save(f"mujoco_ga_instance_gen_{ga.generations_completed}")
        backup_data(ga=ga)
        print("Saved and downloaded GA instance checkpoint")

    @staticmethod
    def fitness_func(chromosome, idx):
        # TODO: FIXME
        # TODO: Better logging
        start_time = time()
        robot_asset_path = f"robot_{idx}"
        tb_logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
        print(tb_logdir)

        end_effector = utils.chromosome_to_end_effector(chromosome, NUM_JOINTS)
        end_effector.build(robot_asset_path)

        print(f"Building end effector {idx}:")
        print(end_effector)

        env = TrainEnv(robot_asset_path)
        print(f"Starting training for idx: {idx}")

        try:
            model = PPO.load(
                f"models/end_effector{idx}", env, verbose=0, tensorboard_log=tb_logdir
            )
            print(f"Loaded {idx} model")
        except Exception as e:
            model = PPO(
                "MultiInputPolicy",
                env,
                verbose=1,
                tensorboard_log=tb_logdir,
                batch_size=BATCH_SIZE,
            )
            print(f"Could not load {idx} model, creating new model")
            print(e)

        model.learn(
            total_timesteps=TRAIN_TIMESTEPS,
            tb_log_name=robot_asset_path,
            reset_num_timesteps=False,
        )
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
        fitness = total_reward / TEST_TIMESTEPS

        model.save(f"models/end_effector{idx}")
        env.close()

        EndEffectorGA.current_gen_fitness[idx] = fitness
        print(f"Fitness for {idx}: {fitness}")
        print(f"Total training time {(time() - start_time)/60}min")
        if idx % 4 == 0:
            backup_data()
        return fitness


def backup_data(ga=None):
    dt_string = datetime.now().strftime("%d.%m.%Y-%H:%M:%S")
    gdrive_path = ""
    if ga is not None:
        gdrive_path = (
            f"/content/gdrive/My Drive/rl_training_v2/{dt_string}_gen_checkpoint"
        )
    else:
        gdrive_path = f"/content/gdrive/My Drive/rl_training_v2/{dt_string}"
    os.mkdir(gdrive_path)

    try:
        if ga is not None:
            ga.save("mujoco_ga_instance")
            shutil.copyfile(
                f"/content/mujoco/mujoco_ga_instance_gen_{ga.generations_completed}.pkl",
                f"{gdrive_path}/mujoco_ga_instance_gen_{ga.generations_completed}.pkl",
            )
    except Exception as e:
        print(f"Could not backup checkpoint GA file: {e}")

    for i in range(8):
        shutil.copyfile(
            f"/content/mujoco/models/end_effector{i}.zip",
            f"{gdrive_path}/end_effector{i}.zip",
        )
    shutil.copyfile(
        "/content/mujoco/mujoco_ga_instance.pkl",
        f"{gdrive_path}/mujoco_ga_instance.pkl",
    )

    print(f"Backed up at {gdrive_path}")


if __name__ == "__main__":
    num_generations = 10
    num_parents_mating = 4
    population_count = 8

    try:
        ga = pygad.load("mujoco_ga_instance")
        print(f"Loaded GA instance, generations completed: {ga.generations_completed}")
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
    except KeyboardInterrupt as e:
        print("User stopped GA, saving ga instance")
        print(e)
    finally:
        ga.save("mujoco_ga_instance")
        backup_data(ga=ga)

    solution, solution_fitness, solution_idx = ga.best_solution()
    print(f"Parameters of the best solution : {solution}")
    print(f"Fitness value of the best solution = {solution_fitness}")
    print(f"Index of the best solution : {solution_idx}")
