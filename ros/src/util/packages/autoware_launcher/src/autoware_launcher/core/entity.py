from autoware_launcher.core import basetree
from autoware_launcher.core import myutils

class AwEntityTree(basetree.AwBaseTree):

    def load(self, profile):

        paths = myutils.listfiles(myutils.profile(profile))
        for path in paths:
            print path



class AwEntityNode(basetree.AwBaseNode):

    pass



if __name__ == "__main__":

    tree = AwEntityTree()
    tree.load("quickstart3")
