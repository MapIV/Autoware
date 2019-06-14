from autoware_launcher.core import AwLaunchServer, AwLaunchClientIF
from autoware_launcher.core.launch import AwLaunchNode
from autoware_launcher.core import myutils
import os

class Context(object):

    def __init__(self):
        self.server = AwLaunchServer()
        self.server.load_profile("operator/common")

        map_path = myutils.profile("operator/maps")
        self.map_profile_list = [name for name in os.listdir(map_path) if os.path.isdir(os.path.join(map_path, name))]
        self.map_profile = self.map_profile_list[0]

        computing_path = myutils.profile("operator/computing")
        self.computing_profile_list = [name for name in os.listdir(computing_path) if os.path.isdir(os.path.join(computing_path, name))]
        self.computing_profile = self.computing_profile_list[0]

        sensing_path = myutils.profile("operator/sensing")
        self.sensing_profile_list = [name for name in os.listdir(sensing_path) if os.path.isdir(os.path.join(sensing_path, name))]
        self.sensing_profile = self.sensing_profile_list[0]

        self.userhome_path = myutils.userhome()
        self.rosbag_play_xml = myutils.package("resources/rosbagplay.xml")

        self.node_status_watcher = NodeStatusWatcher()
        self.server.register_client(self.node_status_watcher)

    def set_map_profile(self, mp):
        self.map_profile = mp

    def set_map_profile_list(self, mp_list):
        self.map_profile_list = mp_list

    def set_computing_profile(self, cp):
        self.computing_profile = cp

    def set_computing_profile_list(self, cp_list):
        self.computing_profile_list = cp_list

    def set_sensing_profile(self, sp):
        self.sensing_profile = sp

    def set_sensing_profile_list(self, sp_list):
        self.sensing_profile_list = sp_list
    
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
