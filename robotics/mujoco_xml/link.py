from dataclasses import dataclass

from mujoco_xml.part import MuJoCoPart


@dataclass()
class Link(MuJoCoPart):
    """Defines a MuJoCo link"""

    parent: MuJoCoPart = None
    idx: int = 0

    def name(self):
        """Returns the name of this link as a string"""
        if self.parent is None:
            return f"Base/{self.idx}"
        else:
            return f"{self.parent.name()}/{self.idx}j"

    def set_parent(self, parent: MuJoCoPart, idx: int):
        if parent == self:
            return
        self.parent = parent
        self.idx = idx

    def build_geometry(self) -> str:
        """Creates the geometry XML for this link"""
        child_geometry = "\n".join([child.build_geometry() for child in self.children])

        return f"""
            <body childclass="robot0:link" name="{self.name()}" pos="{self.position.to_attribute()}">
                <inertial diaginertia="0.1 0.1 0.1" mass="4" pos="-0.01 0 0"></inertial>
                <geom pos="{self.position.x} {self.position.y} {self.position.z}" size="{self.size.x} {self.size.y} {self.size.z}" type="box" name="robot0:{self.name()}" material="robot0:gripper_finger_mat" condim="4" friction="{self.friction} {self.friction} {self.friction}"></geom>
                {child_geometry}
            </body>
        """

    def build_actuator(self):
        """Creates the actuator XML for this link"""
        child_actuators = "\n".join(
            filter(
                lambda s: s != "", [child.build_actuator() for child in self.children]
            )
        )

        return child_actuators

    def __str__(self):
        return f"Link<name: {self.name()}, children: {len(self.children)}>"

    def __repr__(self):
        return f"Link<name: {self.name()}, children: {len(self.children)}>"
