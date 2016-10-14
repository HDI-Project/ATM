from delphi.config import Config
from delphi.utilities import *
from novaclient.v1_1 import client
import os, sys
import random

class Launcher:

    LAUNCH_TYPE_WORKERS = "workers"
    LAUNCH_NAME_WORKERS = "grid-worker"

    HOME = "/root/hyperdelphi"

    def __init__(self, config):
        """
            Create our connection to the cloud.
        """
        self.cloud = client.Client(
            config.get(Config.CLOUD, Config.CLOUD_USER), config.get(Config.CLOUD, Config.CLOUD_PASS),
            config.get(Config.CLOUD, Config.CLOUD_TENANT), config.get(Config.CLOUD, Config.CLOUD_AUTH_URL),
            service_type=config.get(Config.CLOUD, Config.CLOUD_SERVICE))
        self.config = config
        self.files = {}
        self.userdata = ""
        self.launch_name = ""
        
    def killall(self, name, key_name):
        """
            Kill all nodes with given name and keypair name.
        """
        print "Run name: %s" % name
        print "My key: %s" % key_name
        
        # get all instances running in evo
        nodes = self.cloud.servers.list()
        count = 0
        for node in nodes:
            if node.name.startswith(name) and node.key_name == key_name:
                print "Now terminating %s..." % node
                node.delete()
                count += 1
        return count

    def get_ip_addrs(self, name_startswith, key_name):
        """
            Retrieves list of IP addresses for nodes in this tenant
            matching the key_name and name starts with creteria.
        """
        ips = []
        for node in self.cloud.servers.list():
            if node.name.startswith(name_startswith) and node.key_name == key_name:
                try:
                    ips.append(node.addresses["inet"][0]["addr"])
                except Exception as e:
                    print "Exception getting IP addresses from OpenStack: %s" % e
        random.shuffle(ips)
        return ips

    def sync_models(self, name_startswith, key_name):
        """
            Grab all the IPs matching a certain criteria.
        """
        ips = self.get_ip_addrs(name_startswith, key_name)
        for ip in ips:
            try:
                print "[*] Syncing models from %s..." % ip
                os.system("sh sync_models.sh %s" % ip)
            except Exception as e:
                print "Exception in syncing %s" % ip
    
    def purge_errors(self):
        """
            Removes servers in an error state.
        """
        killed = 0
        for s in self.cloud.servers.list():
            if s.status == "ERROR":
                s.delete()
                killed += 1
        return killed

    def kill_hostnames(self, klist):
        """
            Given a list of hostnames, kills those nodes.
        """    
        # get all instances running in evo
        nodes = self.cloud.servers.list()
        for node in nodes:
            if node.addresses["inet"][0]["addr"] in klist:
                print "Now terminating %s..." % node
                node.delete()
          