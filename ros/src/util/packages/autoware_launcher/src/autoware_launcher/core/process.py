from autoware_launcher.core import myutils

import logging
import subprocess
import threading
logger = logging.getLogger(__name__)


class AwProcessManager(object):

    def __init__(self):
        super(AwProcessManager, self).__init__()
        self.__items  = {}
        self.__server = None
        self.__tmpdir = "/tmp/autoware_launcher"
        myutils.makedirs(self.__tmpdir, mode=0o700, exist_ok=True)

    def register_server(self, server):
        self.__server = server

    def roslaunch(self, lpath, xtext):
        xpath = self.__tmpdir + "/" + lpath.replace("/", "-") + ".xml"
        with open(xpath, mode="w") as fp:
            fp.write(xtext)
        args = [lpath, "roslaunch {}".format(xpath)]
        proc = threading.Thread(target=self.__process, args=args)
        proc.start()

    def terminate(self, lpath):
        logger.debug("terminate: {}".format(lpath))
        self.__items[lpath].terminate()

    def __process(self, lpath, command):
        self.__items[lpath] = subprocess.Popen(command, shell=True)
        logger.debug("start: {}".format(command))
        self.__items[lpath].wait()
        logger.debug("finish: {}".format(command))
        self.__server.runner_finished(lpath)
