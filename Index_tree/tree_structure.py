class Node(object):
    def __init__(self, child_id, name, path:list) -> None:
        self.child_id = child_id
        self.name = name
        self.path = path
        self.children = {}

    def __str__(self, level=0):
        ret = "\t" * level + repr(self.child_id) + ':' + repr(self.name) + "\n"
        for child in self.children.values():
            ret += child.__str__(level + 1)
        return ret

    def __repr__(self):
        return '<tree node representation>'
