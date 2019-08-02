from autoware_launcher.core import AwLaunchServer, AwLaunchClientIF
from autoware_launcher.core.launch import AwLaunchNode
from autoware_launcher.core import myutils
from autoware_launcher.core.rostopic import RosTopicAdapter
import os

class Context(object):

    def __init__(self):
        self.server = AwLaunchServer()
        self.server.load_profile("operator/common")

        self.dirpath = 'operator'
        self.nodepath = 'root'

        self.profile_list = {}
        self.selected_profile = {}

        self.userhome_path = myutils.userhome()
        self.rosbag_play_xml = myutils.package("resources/rosbagplay.xml")

        self.node_status_watcher = NodeStatusWatcher()
        self.server.register_client(self.node_status_watcher)

        self.server.register_client(RosTopicAdapter())
    
    def set_profile_list(self, path, val):
        self.profile_list[path] = val
    
    def set_selected_profile(self, path, val):
        self.selected_profile[path] = val
    
    def load_profile_list(self, path):
        dirpath = myutils.profile(path)
        profile_list = []
        if os.path.exists(dirpath):
            profile_list = [name for name in os.listdir(dirpath) if os.path.isdir(os.path.join(dirpath, name))]
        self.set_profile_list(path, profile_list)
    
    def register_node_status_watcher_client(self, client):
        self.node_status_watcher.register_client(client)

class NodeStatusWatcher(AwLaunchClientIF):
    def __init__(self):
        self.nodes = {}
        self.clients = []    # client implements node_status_updated
    
    def profile_updated(self):
        pass

    def node_updated(self, lpath):
        pass

    def node_created(self, lpath):
        pass

    def node_removed(self, lpath):
        pass

    def status_updated(self, lpath, state):
        self.nodes[lpath] = state
        # print("status updated")
        # print(self.nodes)
        self.notify_status_update()

    def register_client(self, client):
        if client.node_status_updated is None:
            raise NotImplementedError("Not implemented: node_status_updated in {}".format(client.__class__.__name__))
        self.clients.append(client)
    
    def notify_status_update(self):
        for c in self.clients:
            c.node_status_updated(self.nodes)
