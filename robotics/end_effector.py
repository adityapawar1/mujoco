from dataclasses import dataclass


@dataclass()
class Part:
    length: float
    children = []

    def __repr__(self):
        return f"{self.__class__.__name__}<length: {self.length} - children: {self.children}"

@dataclass()
class Joint(Part):
    length: float
    range: int
    children = []

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


class EndEffector():
    pass
