from dataclasses import dataclass
from enum import Enum

from mujoco_xml.part import MuJoCoPart, Rotation


class JointType(Enum):
    """Defines the way the joint rotates (slide/hinge)"""

    SLIDE = "slidePart"
    HINGE = "hingePart"


@dataclass()
class Joint(MuJoCoPart):
    """Defines a MuJoCo Joint"""

    range: float
    rotation: Rotation
    joint_type: JointType
    parent: MuJoCoPart = None
    idx: int = 0

    def name(self):
        """Returns the name of this joint as a string"""
        if self.parent is None:
            return f"Base/{self.idx}"
        else:
            return f"{self.parent.name()}/{self.idx}j"

    def set_parent(self, parent: MuJoCoPart, idx: int):
        self.parent = parent
        self.idx = idx

    def build_geometry(self) -> str:
        """Creates the geometry XML for this joint"""
        child_geometry = "\n".join([child.build_geometry() for child in self.children])

        return f"""
            <body childclass="robot0:{self.joint_type.value}" name="{self.name()}" pos="{self.position.to_attribute()}">
                <inertial diaginertia="0.1 0.1 0.1" mass="4" pos="-0.01 0 0"></inertial>
                <joint axis="{Rotation.to_attribute(self.rotation)}" name="robot0:{self.name()}" range="{-self.range} {self.range}"></joint>
                <geom pos="{self.position.to_attribute()}" size="{self.size.to_attribute()}" type="box" name="robot0:{self.name()}" material="robot0:gripper_finger_mat" condim="4" friction="{self.friction} {self.friction} {self.friction}"></geom>
                {child_geometry}
            </body>
        """

    def build_actuator(self) -> str:
        """Creates the actuator XML for this joint"""
        child_actuators = "\n".join(
            filter(
                lambda s: s != "", [child.build_actuator() for child in self.children]
            )
        )

        return (
            f"""
            <position ctrllimited="true" ctrlrange="{-self.range} {self.range}" joint="robot0:{self.name()}" kp="30000" name="robot0:{self.name()}" user="1"></position>
        """
            + child_actuators
        )

    def __str__(self):
        return f"Joint<name: {self.name()}, children: {len(self.children)}>"

    def __repr__(self):
        return f"Joint<name: {self.name()}, children: {len(self.children)}>"
