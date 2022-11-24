import numpy as np

from robotics.end_effector import EndEffector
from robotics.mujoco_xml.joint import Joint, JointType
from robotics.mujoco_xml.part import Attachment, Position, Size


def chromosome_to_end_effector(chromosome, num_joints):
    """Converts a chromosome to an end effector that can be run in MuJoCo"""
    joints = []
    # First, create all components into parts
    joints_genes = np.split(chromosome, num_joints)
    for joint_genes in joints_genes:
        position = Position(joint_genes[0], joint_genes[1], joint_genes[2])
        joint_type = JointType.SLIDE if joint_genes[3] == 1 else JointType.HINGE
        size = Size(joint_genes[4], joint_genes[5], joint_genes[6])
        attachment = [Attachment.X, Attachment.Y, Attachment.Z][joint_genes[7]]
        joint_range = (
            0.15 if joint_type == JointType.SLIDE else 0.75
        )  # hinge needs more range
        joint = Joint(
            range=joint_range,
            position=position,
            joint_type=joint_type,
            size=size,
            attachment=attachment,
            friction=1.0,
            children=[],
        )
        joints.append(joint)

    # Second, assign children (TODO: arbitrarily for now)
    # TODO: generalize for other num_joints (only really works for 9)
    base_links = []
    for i in range(0, len(joints), 3):
        j1 = joints[i]
        j2 = joints[i + 1]
        j3 = joints[i + 2]

        j1.add_child(j2)
        j2.add_child(j3)
        base_links.append(j1)

    return EndEffector(base_links)
