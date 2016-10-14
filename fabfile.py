from fabric.api import *
from delphi.config import Config
from delphi.cloud import Openstack
from delphi.utilities import *
from novaclient.v1_1 import client

from multiprocessing.pool import ThreadPool
import os, random

"""
$ fab deploy
"""

# configuration and launcher
config = Config("config/delphi.cnf")

cloud = Openstack(config)
keyname = config.get("cloud", "key")
hostnames = cloud.get_ip_addrs("delphi-worker", keyname)
random.shuffle(hostnames)
print "There are %d unique hostnames" % len(hostnames)
env.hosts = hostnames
env.timeout = 10

# fabric env setup
env.user = 'ubuntu'
env.key_filename = "~/.ssh/"+keyname+".pem"
env.skip_bad_hosts = True
env.colorize_errors = True
env.pool_size = 4
env.disable_known_hosts = True

WORKERS_PER_MACHINE = 3

def sync():
    config = Config("config/delphi.cnf")
    cloud = Openstack(config)
    hostnames = cloud.get_ip_addrs("delphi-worker", keyname)
    random.shuffle(hostnames)
    print "There are %d unique hostnames" % len(hostnames)
    
    inputs = ["sh sync_models.sh %s %s" % (h, key) for h, key in zip(hostnames, [env.key_filename] * len(hostnames))]
    pool = ThreadPool(processes=4)
    pool.map(os.system, inputs)
    
@parallel
def deploy():
    code_dir = '/home/ubuntu/delphi'
    with settings(warn_only=True):
        if run("test -d %s" % code_dir).failed:
            run("git clone https://%s:%s@%s %s" % (
                config.get(Config.GIT, Config.GIT_USER), config.get(Config.GIT, Config.GIT_PASS), 
                config.get(Config.GIT, Config.GIT_REPO), code_dir,))
            with cd(code_dir):
                run("git pull")
                for i in range(1, WORKERS_PER_MACHINE + 1, 1):
                    run("screen -dm -S worker%d python worker.py; sleep 2" % (i,))
        else:
            with cd(code_dir):
                #run("git pull")
                for i in range(1, WORKERS_PER_MACHINE + 1, 1):
                    if not run("screen -ls | grep \"worker%d\";" % i):
                        run("screen -dm -S worker%d python worker.py; sleep 2" % (i,))

@parallel       
def killworkers():
    run("pkill -15 screen")

def purge():
    print "Killed %d nodes in error state." % launcher.purge_errors()
