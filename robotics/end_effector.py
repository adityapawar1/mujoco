import os

from robotics.xml.joint import Joint, JointType
from robotics.xml.link import Link
from robotics.xml.part import Attachment, MuJoCoPart, Position, Size

HEADER = """
    <?xml version="1.0" encoding="utf-8"?>
    <mujoco>
"""

FOOTER = """
    </mujoco>
"""


class EndEffector():
    base_link: MuJoCoPart = Link(
        parent=None,
        position=Position.zero(),
        idx=0,
        size=Size.zero(),
        attachment=Attachment.X,
        friction=1
    )

    def __init__(self, parts: list[MuJoCoPart]) -> None:
        [self.base_link.add_child(part) for part in parts]

    def add_to_base(self, part: MuJoCoPart):
        self.base_link.add_child(part)

    def build(self):
        geometry = self.base_link.build_geometry()
        actuator = self.base_link.build_actuator()

        with open(os.path.join("robotics", "assets", "fetch", "end_effector_geometry.xml"), 'w') as file:
            file.write(f"{HEADER} \n {geometry} \n {FOOTER}")

        with open(os.path.join("robotics", "assets", "fetch", "end_effector_actuator.xml"), 'w') as file:
            file.write(f"{HEADER} \n {actuator} \n {FOOTER}")


if __name__ == "__main__":
    joint1 = Joint(
        range=0.15,
        position=Position(0, 0.01, 0),
        joint_type=JointType.SLIDE,
        size=Size(0.05, 0.015, 0.05),
        attachment=Attachment.Z,
        friction=1.0
    )
    joint2 = Joint(
        range=0.15,
        position=Position(0, -0.01, 0),
        joint_type=JointType.SLIDE,
        size=Size(0.05, 0.015, 0.05),
        attachment=Attachment.Z,
        friction=1.0
    )

    end_effector = EndEffector([joint1, joint2])
    end_effector.build()
