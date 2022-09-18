from dataclasses import dataclass

from robotics.xml.part import MuJoCoPart, Attachment


@dataclass()
class Link(MuJoCoPart):
    parent: MuJoCoPart | None = None
    idx: int = 0

    def name(self):
        if self.parent is None:
            return f"Base_link_{self.idx}"
        else:
            return f"{self.parent.name()}_joint_{self.idx}"

    def set_parent(self, parent: MuJoCoPart, idx: int):
        if parent == self:
            return
        self.parent = parent
        self.idx = idx

    def find_attachment_position(self):
        if self.parent is not None:
            if self.attachment == Attachment.X:
                return f"{self.parent.size.x} 0 0"
            elif self.attachment == Attachment.Y:
                return f"0 {self.parent.size.y} 0"
            elif self.attachment == Attachment.Z:
                return f"0 0 {self.parent.size.z}"
        else:
            return "0 0 0"

    def build_geometry(self) -> str:
        child_geometry = "\n".join(
            [child.build_geometry() for child in self.children]
        )

        return f"""
            <body childclass="robot0:link" name="{self.name()}" pos="{self.find_attachment_position()}">
                <inertial diaginertia="0.1 0.1 0.1" mass="4" pos="-0.01 0 0"></inertial>
                <geom pos="0 0.008 0" size="{self.size.x} {self.size.y} {self.size.z}" type="box" name="robot0:{self.name()}" material="robot0:gripper_finger_mat" condim="4" friction="{self.friction} {self.friction} {self.friction}"></geom>
                {child_geometry}
            </body>
        """

    def build_actuator(self):
        child_actuators = "\n".join(
            filter(
                lambda s: s != "",
                [child.build_actuator() for child in self.children]
            )
        )

        return child_actuators

    def __str__(self):
        return f"Link<name: {self.name()}, children: {len(self.children)}>"

    def __repr__(self):
        return f"Link<name: {self.name()}, children: {len(self.children)}>"
