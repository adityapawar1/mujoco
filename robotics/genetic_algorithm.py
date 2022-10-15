import numpy as np
import pygad
from robotics.end_effector import EndEffector

from robotics.xml.joint import Joint, JointType
from robotics.xml.part import Attachment, Position, Size


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
    SINGLE_JOINT_SPACE = [FLOAT_SPACE, FLOAT_SPACE, FLOAT_SPACE, JOINT_TYPE_SPACE, FLOAT_SPACE, FLOAT_SPACE, FLOAT_SPACE, ATTACHMENT_LOC_SPACE]

    def __init__(self, num_joints: int, num_generations: int, num_parents_mating: int, population_count: int) -> None:
        self.num_joints = num_joints
        gene_type = np.tile(self.SINGLE_JOINT_VARIABLES, num_joints).tolist()
        gene_space = np.tile(self.SINGLE_JOINT_SPACE, num_joints).tolist()

        super().__init__(
                num_generations=num_generations,
                num_parents_mating=num_parents_mating,
                num_genes=len(gene_type),
                sol_per_pop=population_count,
                gene_type=gene_type,
                gene_space=gene_space,
                fitness_func=self.fitness_func,
                on_mutation=self.on_mutation
            )

    def chromosome_to_end_effector(self, chromosome):
        joints = []
        # First, create all components into parts
        joints_genes = np.split(chromosome, self.num_joints)
        for joint_genes in joints_genes:
            position = Position(joint_genes[0], joint_genes[1], joint_genes[2])
            joint_type = JointType.SLIDE if joint_genes[3] == 1 else JointType.HINGE
            size = Size(joint_genes[4], joint_genes[5], joint_genes[6])
            attachment = [Attachment.X, Attachment.Y, Attachment.Z][joint_genes[7]]
            joint = Joint(
                range=0.15 if joint_type == JointType.SLIDE else 0.75, # hinge needs more range
                position=position,
                joint_type=joint_type,
                size=size,
                attachment=attachment,
                friction=1.0
            )
            joints.append(joint)

        # Second, assign children (TODO: arbitrarily for now)
        # TODO: generalize for other num_joints (only really works for 9)
        base_links = []
        for i in range(0, len(joints), 3):
            j1 = joints[i]
            j2 = joints[i+1]
            j3 = joints[i+2]

            j1.add_child(j2)
            j2.add_child(j3)
            base_links.append(j1)

        return EndEffector(base_links)

    def fitness_func(end_effector, idx):
        raise NotImplementedError

    def on_mutation(end_effector, idx):
        raise NotImplementedError


if __name__ == "__main__":
    num_joints = 6
    num_generations = 5
    num_parents_mating = 5
    population_count = 10
    ga = EndEffectorGA(
            num_joints,
            num_generations,
            num_parents_mating,
            population_count
        )
    pop1 = ga.population[0]
    ee1 = ga.chromosome_to_end_effector(pop1)
    print(pop1)
    print(ee1)
    ee1.build()
