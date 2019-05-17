
class Context(object):
    def __init__(self):
        self.map_profile_list = ['dummy_mp1', 'dummy_mp2']
        self.map_profile = self.map_profile_list[0]

        self.computing_profile_list = ['dummy_cp1', 'dummy_cp2']
        self.computing_profile = self.computing_profile_list[0]

    def set_map_profile(self, mp):
        self.map_profile = mp

    def set_map_profile_list(self, mp_list):
        self.map_profile_list = mp_list

    def set_computing_profile(self, cp):
        self.computing_profile = cp

    def set_computing_profile_list(self, cp_list):
        self.computing_profile_list = cp_list
