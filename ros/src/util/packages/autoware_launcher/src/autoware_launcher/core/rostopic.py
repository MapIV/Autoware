import logging
import rospy
import std_msgs

logger = logging.getLogger(__name__)


class RosTopicAdapter(object):

    def __init__(self):
        rospy.init_node("autoware_launcher")
        self.__pub = rospy.Publisher("/autoware_launcher/callback", std_msgs.msg.String, queue_size=10)
        self.__pub_status = rospy.Publisher("/autoware_launcher/status", std_msgs.msg.String, queue_size=1, latch=True)
        self.__status_cache = {}

    def profile_updated(self):
        self.__pub.publish(std_msgs.msg.String(data="profile_updated"))

    def node_updated(self, lpath):
        self.__pub.publish(std_msgs.msg.String(data="node_updated {}".format(lpath)))

    def node_created(self, lpath):
        self.__pub.publish(std_msgs.msg.String(data="node_created {}".format(lpath)))

    def node_removed(self, lpath):
        self.__pub.publish(std_msgs.msg.String(data="node_removed {}".format(lpath)))

    def status_updated(self, lpath, state):
        self.__pub.publish(std_msgs.msg.String(data="status_updated {} {}".format(lpath, state)))
        self.__status_cache[lpath] = state
        status_list = ["{} {}".format(k, v) for k, v in self.__status_cache.items()]
        self.__pub_status.publish(std_msgs.msg.String(data="\n".join(status_list)))
