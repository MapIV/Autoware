from logging import getLogger
logger = getLogger(__name__)

import os
import yaml

from autoware_launcher.core import basetree
from autoware_launcher.core import myutils


class AwLaunchTree(basetree.AwBaseTree):

    def __init__(self, server, plugins):
        super(AwLaunchTree, self).__init__()
        self.server  = server
        self.plugins = plugins

    def __str__(self):
        return "Node:{} Base:{} Children:{}".format(self.name, self.basepath(), len(self.children()))

    def save(self, basepath):
        self._AwBaseNode__basepath = basepath
        with open(basepath + ".launch", mode = "w") as fp:
            fp.write("dummy")
        for node in self.listnode():
            fullpath = node.fullpath() + ".yaml"
            myutils.makedirs(os.path.dirname(fullpath), exist_ok = True)
            with open(fullpath, mode = "w") as fp:
                fp.write(yaml.safe_dump(node.export_data(), default_flow_style = False))

    def load(self, basepath):
        self._AwBaseNode__basepath = myutils.profile(basepath)
        def load_node(node, plugins):
            fullpath = node.fullpath()
            with open(fullpath + ".yaml") as fp:
                node.import_data(yaml.safe_load(fp), plugins)
            for child in node.children():
                load_node(child, plugins)
        root = AwLaunchNode("root")
        self.addchild(root)
        load_node(root, self.plugins)

    def load_subtree(self, basepath, nodepath):
        def load_node(node, plugins):
            fullpath = node.fullpath()
            with open(fullpath + ".yaml") as fp:
                node.import_data(yaml.safe_load(fp), plugins)
            for child in node.children():
                load_node(child, plugins)
        parent = self.find(os.path.dirname(nodepath))
        if parent is None:
            return "parent is not found"
        target = self.find(nodepath)
        if target:
            parent.delchild(target)
        target = AwLaunchNode(os.path.basename(nodepath), myutils.profile(basepath))
        parent.addchild(target)
        load_node(target, self.plugins)

    def make(self, ppath, plugins):
        plugin = plugins.find(ppath)
        launch = AwLaunchNode("root")
        launch.plugin = plugin
        launch.config = plugin.default_config()
        self.addchild(launch)

    def export(self, rootpath):
        for node in self.listnode():
            xtext = node.generate_launch()
            xpath = node.path.replace("/", "-") + ".xml"
            xpath = os.path.join(rootpath, xpath)
            with open(xpath, mode="w") as fp: fp.write(xtext)

    def create(self, lpath, ppath):
        logger.debug("Tree Create: " + lpath + ", " + ppath)
        parent = self.find(os.path.dirname(lpath))
        if not parent:
            return "parent is not found"
        if self.find(lpath):
            return "name exists"

        plugin = self.plugins.find(ppath)
        launch = AwLaunchNode(os.path.basename(lpath))
        launch.plugin = plugin
        launch.config = plugin.default_config()
        parent.addchild(launch)
        return None


class AwLaunchNode(basetree.AwBaseNode):

    STOP, EXEC, TERM = 0x00, 0x01, 0x02

    def __init__(self, name, base=None):
        super(AwLaunchNode, self).__init__(name, base)
        self.plugin = None
        self.config = None
        self.status = self.STOP

    def __str__(self):
        return "Node:{} Base:{} Children:{}".format(self.name, self.basepath(), len(self.children()))

    def tostring(self):
        return yaml.safe_dump(self.todict())

    def todict(self):
        return \
        {
            "plugin"  : self.plugin.todict(),
            "config"  : self.config,
            "children": [child.name for child in self.children()]
        }

    # experimental, move to tree
    def remove_child(self, name):
        if not name:
            return "name is empty"
        if not self.haschild(name):
            return "name does not exist"
        self.delchild(name)
        self.send_config_removed(name)
        return None

    def update(self, ldata):
        self.config.update(ldata["config"])
        return None

    def launch(self, xmode):
        if xmode:
            return self.__exec()
        else:
            return self.__term()

    def update_exec_status(self):
        status = self.STOP
        for child in self.children():
            status |= child.status
        if self.status != status:
            self.status = status
            return True
        return False

    def __exec(self):
        if self.plugin.isleaf():
            if self.status == self.STOP:
                self.status = self.EXEC
                return (True, True)
        else:
            status = self.STOP
            for child in self.children():
                status |= child.status
            if self.status != status:
                self.status = status
                return (True, False)
        return (False, False)

    def __term(self):
        if self.plugin.isleaf():
            if self.status == self.EXEC:
                self.status = self.TERM
                return (True, True)
        else:
            status = self.STOP
            for child in self.children():
                status |= child.status
            if self.status != status:
                self.status = status
                return (True, False)
        return (False, False)

    def get_config(self, key, value):
        return self.config.get(key, value)

    def generate_launch(self):
        lines = []
        if self.plugin.isleaf():
            lines.append('<launch>')
            lines.append('  <include file="{}">'.format(self.plugin.rosxml()))
            for data in self.plugin.args():
                argvalue = self.config.get("args." + data.name)
                if argvalue is not None:
                    lines.append('    <arg name="{}" value="{}"/>'.format(data.name, data.xmlstr(argvalue)))
            lines.append('  </include>')
            lines.append('</launch>')
        else:
            lines.append('<launch>')
            for childname in self.childnames():
                childpath = os.path.join(self.path, childname)
                childpath = childpath.replace("/", "-") + ".xml"
                lines.append('  <include file="{}"/>'.format(childpath))
            lines.append('</launch>')
        return "\n".join(lines)

    def import_data(self, data, plugins):
        self.plugin = plugins.find(data["plugin"])
        self.config = data["config"]
        if data["children"] is None:
            self.setleaf()
        else:
            for childname in data["children"]:
                self.addchild(AwLaunchNode(childname))

    def export_data(self):
        children = map(lambda node: node.name, self.children())
        plugin = self.plugin.path
        config = self.config
        return { "children": children, "plugin": plugin, "config": config }


if __name__ == "__main__":
    from .plugin import AwPluginTree
    import_path = myutils.profile("quickstart")
    update_path = myutils.profile("quickstart")
    export_path = myutils.profile("sample.tmp")
    plugin = AwPluginTree()
    launch = AwLaunchTree(None, plugin)
    launch.load(import_path)
    print "============================================================"
    launch.dump()
    print "============================================================"
    launch.load_subtree(update_path, "root/sensing/camera")
    print "============================================================"
    launch.dump()
    print "============================================================"
