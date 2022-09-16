from robotics.parts import Link, Part, Joint


class EndEffector():
    base_link = Link(0)

    def __init__(self, parts: list[Part]) -> None:
        [self.base_link.add_child(part) for part in parts]

    def add_to_base(self, part: Part):
        self.base_link.add_child(part)

    def build(self):
        pass
