import os
import shutil

from mujoco_xml.joint import Joint, JointType
from mujoco_xml.link import Link
from mujoco_xml.part import MuJoCoPart, Position, Size
from mujoco_xml.part import Rotation

HEADER = """
    <?xml version="1.0" encoding="utf-8"?>
    <mujoco>
"""

FOOTER = """
    </mujoco>
"""


class EndEffector:
    """A class to store the tree of MuJoCo parts making up an entire end effector"""

    def __init__(self, parts) -> None:
        self.base_link: MuJoCoPart = Link(
            parent=None,
            position=Position.zero(),
            idx=0,
            size=Size.zero(),
            friction=1,
            children=[],
        )

        [self.base_link.add_child(part) for part in parts]

    def add_to_base(self, part: MuJoCoPart):
        self.base_link.add_child(part)

    def build(self, path: str):
        geometry = self.base_link.build_geometry()
        actuator = self.base_link.build_actuator()

        root = os.path.dirname(os.path.abspath(__file__))
        end_effector_asset_dir = os.path.join(root, "assets", path)
        if not os.path.exists(end_effector_asset_dir):
            os.mkdir(end_effector_asset_dir)

            template_xml_path = os.path.join(root, "assets", "complete.xml")
            shutil.copy(
                template_xml_path, os.path.join(end_effector_asset_dir, "complete.xml")
            )

        with open(
            os.path.join(end_effector_asset_dir, "end_effector_geometry.xml"),
            "w",
        ) as file:
            file.write(f"{HEADER} \n {geometry} \n {FOOTER}")

        with open(
            os.path.join(end_effector_asset_dir, "end_effector_actuator.xml"),
            "w",
        ) as file:
            file.write(f"{HEADER} \n {actuator} \n {FOOTER}")

    def __str__(self) -> str:
        return f"""End effector:\n{self.base_link.tree_representation()}"""


if __name__ == "__main__":
    joint1 = Joint(
        range=0.15,
        position=Position(0, 0.01, 0),
        joint_type=JointType.SLIDE,
        size=Size(0.05, 0.015, 0.05),
        rotation=Rotation.Z,
        friction=1.0,
        children=[],
    )
    joint2 = Joint(
        range=0.15,
        position=Position(0, -0.01, 0),
        joint_type=JointType.SLIDE,
        size=Size(0.05, 0.015, 0.05),
        rotation=Rotation.Z,
        friction=1.0,
        children=[],
    )
    joint3 = Joint(
        range=0.15,
        position=Position(0, -0.01, 0),
        joint_type=JointType.SLIDE,
        size=Size(0.05, 0.015, 0.05),
        rotation=Rotation.Z,
        friction=1.0,
        children=[],
    )

    joint1.add_child(joint3)
    end_effector = EndEffector([joint1, joint2])
    print(end_effector)
    end_effector.build("dev")
