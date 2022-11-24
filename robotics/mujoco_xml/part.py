from dataclasses import dataclass, field
from enum import Enum


# Which axis the object will be attached too
class Attachment(Enum):
    X = 0
    Y = 1
    Z = 2


@dataclass()
class Vector3:
    x: float
    y: float
    z: float

    def to_attribute(self) -> str:
        return f"{round(self.x, 4)} {round(self.y, 4)} {round(self.z, 4)}"

    def __str__(self) -> str:
        return f"({round(self.x, 4)}, {round(self.y, 4)}, {round(self.z, 4)})"


@dataclass()
class Size(Vector3):
    @staticmethod
    def zero():
        return Size(0.0001, 0.0001, 0.0001)


@dataclass()
class Position(Vector3):
    @staticmethod
    def zero():
        return Position(0, 0, 0)


@dataclass(kw_only=True)
class MuJoCoPart:
    size: Size
    attachment: Attachment
    friction: float
    position: Position
    children: list = field(default_factory=list)

    def add_child(self, child):
        child.set_parent(self, len(self.children))
        self.children.append(child)

    def name(self):
        pass

    def build_geometry(self):
        pass

    def build_actuator(self):
        pass

    def set_parent(self, parent, idx):
        pass

    def tree_representation(self, depth=0):
        tab_depth = r"   " * depth
        children_tree = "".join(
            [
                f"{tab_depth}├─ {child.tree_representation(depth=depth+1)}"
                for child in self.children
            ]
        )
        tree = f"{self.name()}: Size={self.size}, Pos={self.position}\n{children_tree}"
        return tree
