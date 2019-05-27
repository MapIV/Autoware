import os


class AwFlatNode(object):

    def __init__(self, tree, path):
        self.__tree = tree
        self.__path = path

    @property
    def tree(self):
        return self.__tree

    @property
    def path(self):
        return self.__path

    @property
    def name(self):
        return os.path.basename(self.__path)


class AwFlatTree(object):

    def __init__(self):
        self.nodes = {}

    def find(self, path):
        return self.nodes.get(path)

    def scan(self, path):
        return filter(lambda node: node.startswith(path), self.nodes.keys())

    def dump(self):
        for name in sorted(self.nodes.keys()):
            self.nodes[name].dump()


class AwBaseNode(object):

    def __init__(self, name, base=None):
        self.__basepath = base
        self.__nodename = name
        self.__parent   = None
        self.__children = []
        self.__childmap = {}

    def dump(self, indent = 0):
        print((indent * " ") + str(self))
        for child in self.children(): child.dump(indent + 2)

    @property
    def tree(self):
        return self.__parent.tree

    @property
    def name(self):
        return self.__nodename

    @property
    def path(self):
        return os.path.join(self.__parent.path, self.name)

    def basepath(self):
        return self.__basepath if self.__basepath else self.__parent.basepath()

    def fullpath(self):
        return os.path.join(self.basepath(), self.path)

    def children(self): # ToDo: remove
        return self.__children

    def childnodes(self):
        return self.__children

    def childnames(self):
        return [node.name for node in self.__children]

    def getchild(self, name):
        return self.__childmap.get(name)

    def haschild(self, name):
        return name in self.__childmap

    def addchild(self, node):
        self.__children.append(node)
        self.__childmap[node.name] = node
        node.__parent = self

    def delchild(self, node):
        self.__children.remove(node)
        self.__childmap.pop(node.name)
        node.__parent = None

    def listnode(self, this = False):
        result = [self] if this else []
        for child in self.children(): result.extend(child.listnode(True))
        return result


class AwBaseTree(AwBaseNode):

    def __init__(self):
        super(AwBaseTree, self).__init__(None)

    @property
    def tree(self):
        return self

    @property
    def path(self):
        return ""

    def find(self, path):
        node = self
        for name in path.split("/"):
            node = node.getchild(name)
        return node
