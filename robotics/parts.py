from dataclasses import dataclass


@dataclass()
class Part:
    length: float
    size: tuple[float, float, float]
    friction: tuple[float, float, float] = (1, 0.05, 0.01)
    children = []

    def build_geometry(self, parent: str, idx: int) -> str:
        return f"""
            <body childclass="robot0:{parent}" name="{self.name(parent, idx)}" pos="0 -0.0159 0">
                <inertial diaginertia="0.1 0.1 0.1" mass="4" pos="-0.01 0 0"></inertial>
                <joint axis="0 -1 0" name="robot0:l_gripper_finger_joint" range="0 0.05"></joint>
                <geom pos="0 0.008 0" size="{self.size[0]} {self.size[1]} {self.size[2]}" type="box" name="robot0:{self.name(parent, idx)}" material="robot0:gripper_finger_mat" condim="4" friction="{self.friction[0]} {self.friction[1]} {self.friction[2]}"></geom>
            </body>
        """

    def build_actuator(self, parent: str, idx: int) -> str:
        return f""" <position ctrllimited="true" ctrlrange="0 0.2" joint="robot0:{self.name(parent, idx)}" kp="30000" name="robot0:{self.name(parent, idx)}" user="1"></position>
        """

    def name(self, parent: str, idx: int):
        return f"{parent}_child_{idx}"

    def __repr__(self):
        return f"{self.__class__.__name__}<length: {self.length} - children: {self.children}"


@dataclass()
class Joint(Part):
    length: float
    range: int
    children = []

    def build_geometry(self):
        pass

    def __repr__(self):
        return f"Joint<length: {self.length}, range: {self.range}> \
                - children: {self.children}"


@dataclass()
class Link(Part):
    length: float
    children = []

    def __repr__(self):
        return f"Link<length: {self.length}> - children: {self.children}"

    def add_child(self, child: Part):
        self.children.append(child)


