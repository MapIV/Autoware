from autoware_launcher.core import AwLaunchServer
from autoware_launcher.core import myutils
import os

class Context(object):

    def __init__(self):
        self.server = AwLaunchServer()
        self.server.load_profile("operator/common")

        path = myutils.profile("operator/maps")
        self.map_profile_list = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
        self.map_profile = self.map_profile_list[0]

        self.computing_profile_list = ['dummy_cp1', 'dummy_cp2']
        self.computing_profile = self.computing_profile_list[0]

	self.userhome_path = myutils.userhome();
	self.rosbag_play_xml = myutils.package("resources/rosbagplay.xml")

    def set_map_profile(self, mp):
        self.map_profile = mp

    def set_map_profile_list(self, mp_list):
        self.map_profile_list = mp_list

    def set_computing_profile(self, cp):
        self.computing_profile = cp

    def set_computing_profile_list(self, cp_list):
        self.computing_profile_list = cp_list
