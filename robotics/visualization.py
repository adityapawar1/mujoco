import pygad

ga: pygad.GA = pygad.load(
    "/Users/adityapawar/Documents/code/Projects/polygence/models/mujoco_ga_instance_gen_10"
)

ga.plot_fitness()
