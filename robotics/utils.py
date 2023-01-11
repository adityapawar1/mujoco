import numpy as np
from numpy.linalg import cholesky

from end_effector import EndEffector
from mujoco_xml.joint import Joint, JointType
from mujoco_xml.part import Position, Size
from mujoco_xml.part import Rotation

LEFT_PARENT_POSITION = Position(0, 0.03, 0)
RIGHT_PARENT_POSITION = Position(0, -0.03, 0)


def normalize_chromosome(chromosome):
    """Adds empty sizes to the parent joints so the chromosome can be split easily"""
    chromosome = np.array(chromosome)
    copy = chromosome.copy()

    sides = np.split(copy, 2)
    right_side = sides[0]
    left_side = sides[1]
    empty_position = np.array([0, 0, 0])

    return np.concatenate([empty_position, right_side, empty_position, left_side])


def chromosome_to_end_effector(chromosome, num_joints):
    """Converts a chromosome to an end effector that can be run in MuJoCo"""
    # Adds position back to parent joints so it can be split easily
    joints = []
    # First, create all components into parts
    print(chromosome)
    print(normalize_chromosome(chromosome))
    joints_genes = np.split(normalize_chromosome(chromosome), num_joints)
    for i, joint_genes in enumerate(joints_genes):
        position = Position(joint_genes[0], joint_genes[1], joint_genes[2])
        if i == 0:  # Right parent
            position = RIGHT_PARENT_POSITION
        elif i == 3:  # Left parent
            position = LEFT_PARENT_POSITION

        joint_type = JointType.HINGE if joint_genes[3] == 2 else JointType.SLIDE
        rotation = [Rotation.X, Rotation.Y, Rotation.Z][int(joint_genes[3])]
        size = Size(joint_genes[4], joint_genes[5], joint_genes[6])
        joint_range = (
            0.04 if joint_type == JointType.SLIDE else 0.75
        )  # hinge needs more range

        # print(f"{i=}")
        # print(f"position: {joint_genes[0]}, {joint_genes[1]}, {joint_genes[2]}")
        # print(f"Joint type: {joint_genes[3]}")
        # print(f"Rotation: {joint_genes[3]}")
        # print(f"Size {joint_genes[4]}, {joint_genes[5]}, {joint_genes[6]}")

        joint = Joint(
            range=joint_range,
            position=position,
            joint_type=joint_type,
            size=size,
            rotation=rotation,
            friction=1.0,
            children=[],
        )
        joints.append(joint)

    # Second, assign children (TODO: arbitrarily for now)
    # TODO: generalize for other num_joints (only really works for multiples of 3)
    base_links = []
    for i in range(0, len(joints), 3):
        j1 = joints[i]
        j2 = joints[i + 1]
        j3 = joints[i + 2]

        j1.add_child(j2)
        j2.add_child(j3)
        base_links.append(j1)

    return EndEffector(base_links)
