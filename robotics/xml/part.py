from dataclasses import dataclass, field
from enum import Enum


# Which axis the object will be attached too
class Attachment(Enum):
    X = 0
    Y = 1
    Z = 2


@dataclass()
class Size:
    x: float
    y: float
    z: float

    @staticmethod
    def zero():
        return Size(0.0001, 0.0001, 0.0001)


@dataclass(kw_only=True)
class MuJoCoPart:
    size: Size
    attachment: Attachment
    friction: float
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
