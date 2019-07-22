from autoware_launcher.core import myutils
from autoware_launcher.core.plugin import AwPluginTree
from autoware_launcher.core.launch import AwLaunchTree
from autoware_launcher.core.launch import AwLaunchNode
from autoware_launcher.core.process import AwProcessManager

import logging
import yaml
logger = logging.getLogger(__name__)


class AwLaunchServerIF(object):
    def make_profile(self, ppath): raise NotImplementedError("make_profile")
    def load_profile(self, fpath): raise NotImplementedError("load_profile")
    def save_profile(self, fpath): raise NotImplementedError("save_profile")
    def list_node   (self, lpath): raise NotImplementedError("list_node")
    def find_node   (self, lpath): raise NotImplementedError("find_node")
    def update_node (self, lpath, ldata): raise NotImplementedError("update_node")
    def create_node (self, lpath, ppath): raise NotImplementedError("create_node")
    def remove_node (self, lpath):        raise NotImplementedError("remove_node")
    def launch_node (self, lpath):        raise NotImplementedError("launch_node")
    #def list_plugin
    #def find_plugin
    #def process_finished
    #def process_stdouted
    #def process_stderred


class AwLaunchClientIF(object):
    def profile_updated(self):               logger.debug("Not implemented: profile_updated in " + self.__class__.__name__)
    def node_updated   (self, lpath):        logger.debug("Not implemented: node_updated in " + self.__class__.__name__)
    def node_created   (self, lpath):        logger.debug("Not implemented: node_created in " + self.__class__.__name__)
    def node_removed   (self, lpath):        logger.debug("Not implemented: node_removed in " + self.__class__.__name__)
    def status_updated (self, lpath, state): logger.debug("Not implemented: status_updated in " + self.__class__.__name__)


class AwLaunchServer(AwLaunchServerIF):

    def __init__(self, sysarg=None):
        self.__plugins = AwPluginTree()
        self.__profile = AwLaunchTree(self, self.__plugins)
        self.__clients = []
        self.__process  = AwProcessManager()
        self.__process.register_server(self)

    def register_runner(self, runner):
        self.__process = runner

    def register_client(self, client):
        self.__clients.append(client)

    def make_profile(self, ppath):
        logger.debug("make_profile: " + ppath)
        self.__profile = AwLaunchTree(self, self.__plugins)
        self.__profile.make(ppath, self.__plugins)
        for client in self.__clients: client.profile_updated()

    def load_profile(self, fpath):
        logger.debug("load_profile: " + fpath)
        self.__profile = AwLaunchTree(self, self.__plugins)
        self.__profile.load(myutils.profile(fpath))
        for client in self.__clients: client.profile_updated()

    def load_profile_subtree(self, basepath, nodepath):
        logger.debug("load_profile_subtree: " + nodepath)
        self.__profile.load_subtree(basepath, nodepath)

    def save_profile(self, fpath):
        logger.debug("save_profile: " + fpath)
        self.__profile.save(myutils.profile(fpath))

    def export_profile(self, fpath):
        logger.debug("export_profile: " + fpath)
        self.__profile.export(fpath)

    def list_node(self):
        logger.debug("list_node: ")
        return map(lambda node: node.path, self.__profile.listnode(False))

    def find_node(self, lpath):
        logger.debug("find_node: " + lpath)
        return self.__profile.find(lpath)

    def update_node(self, lpath, ldata):
        error = self.__profile.find(lpath).update(ldata)
        if not error:
            for client in self.__clients: client.node_updated(lpath)
        return error

    def create_node(self, lpath, ppath):
        error = self.__profile.create(lpath, ppath)
        if not error:
            for client in self.__clients: client.node_created(lpath)
        return error

    def remove_node(self, lpath):
        pass

    def launch_node(self, lpath, xmode): # ToDo: update ancestors status
        logger.debug("launch_node: " + lpath + " " + str(xmode))
        difflist = []
        execlist = []
        nodelist = self.__profile.find(lpath).listnode(True)
        nodelist = sorted(nodelist, reverse = True, key = lambda x: len(x.path))
        for node in nodelist:
            isdiff, isexec = node.launch(xmode)
            if isdiff: difflist.append(node.path)
            if isexec: execlist.append(node.path)
        for node in self.__profile.find(lpath).listparent():
            if type(node) is AwLaunchNode:
                if node.update_exec_status(): difflist.append(node.path)
        logger.debug("Update:" + str(difflist))
        logger.debug("Launch:" + str(execlist))
        for lpath in difflist:
            state = self.__profile.find(lpath).status
            for client in self.__clients: client.status_updated(lpath, state)
        for lpath in execlist:
            if xmode:
                xtext = self.__profile.find(lpath).generate_launch()
                self.__process.roslaunch(lpath, xtext)
            else:
                self.__process.terminate(lpath)

    def runner_finished(self, lpath):
        target = self.__profile.find(lpath)
        target.status = AwLaunchNode.STOP
        difflist = [target]
        for node in target.listparent():
            if type(node) is AwLaunchNode:
                if node.update_exec_status(): difflist.append(node)
        for node in difflist:
            for client in self.__clients: client.status_updated(node.path, node.status)

    def request_json(self, request):
        try:
            request = yaml.safe_load(request)
        except:
            return yaml.safe_dump({"error": "failed to load json"})

        logger.debug(request)
        if request["command"] == "launch":
            self.launch_node(request["path"], True)
            return yaml.safe_dump({"error": None})
        if request["command"] == "terminate":
            self.launch_node(request["path"], False)
            return yaml.safe_dump({"error": None})

        return yaml.safe_dump({"error": "command ignored"})
