from fabric.api import *
from fabric.colors import green as _green, yellow as _yellow
import boto
import boto.ec2
import time
from atm.config import Config

def check_instances_pending(instances):
    isPending = False
    for instance in instances:
        instance.update()
        if(instance.state == u'pending'):
            isPending = True

    return isPending

def query_active_instances():
    ec2_region = config.get(Config.AWS, Config.AWS_EC2_REGION)
    ec2_key = config.get(Config.AWS, Config.AWS_ACCESS_KEY)
    ec2_secret = config.get(Config.AWS, Config.AWS_SECRET_KEY)

    conn = boto.ec2.connect_to_region(ec2_region, aws_access_key_id=ec2_key, aws_secret_access_key=ec2_secret)
    reservations = conn.get_all_reservations()

    public_dns_names = []

    for reservation in reservations:
        instances = reservation.instances

        for instance in instances:
            if instance.state == 'running':
                public_dns_names.append(instance.public_dns_name)

    return public_dns_names

def create_instances():
    """
    Creates EC2 Instance
    """
    print(_green("Started..."))
    print(_yellow("...Creating EC2 instance(s)..."))

    ec2_region = config.get(Config.AWS, Config.AWS_EC2_REGION)
    ec2_key = config.get(Config.AWS, Config.AWS_ACCESS_KEY)
    ec2_secret = config.get(Config.AWS, Config.AWS_SECRET_KEY)
    conn = boto.ec2.connect_to_region(ec2_region, aws_access_key_id=ec2_key, aws_secret_access_key=ec2_secret)

    ec2_amis = config.get(Config.AWS, Config.AWS_EC2_AMIS)
    image = conn.get_all_images(ec2_amis)

    ec2_key_pair = config.get(Config.AWS, Config.AWS_EC2_KEY_PAIR)
    ec2_instance_type = config.get(Config.AWS, Config.AWS_EC2_INSTANCE_TYPE)
    num_instances = config.get(Config.AWS, Config.AWS_NUM_INSTANCES)
    # must give num_instances twice because 1 min num and 1 max num
    reservation = image[0].run(num_instances, num_instances, key_name=ec2_key_pair, instance_type=ec2_instance_type)

    while check_instances_pending(reservation.instances):
        print(_yellow("Instances still pending"))
        time.sleep(10)

    for instance in reservation.instances:
        print(_green("Instance state: %s" % instance.state))
        print(_green("Public dns: %s" % instance.public_dns_name))

#@parallel
def deploy():
    code_dir = '/home/ubuntu/atm'
    WORKERS_PER_MACHINE = int(config.get(Config.AWS, Config.AWS_NUM_WORKERS_PER_INSTACNCES))
    with settings(warn_only=True):
        if run("test -d %s" % code_dir).failed:
            run("git clone https://%s:%s@%s %s" % (
                config.get(Config.GIT, Config.GIT_USER), config.get(Config.GIT, Config.GIT_PASS),
                config.get(Config.GIT, Config.GIT_REPO), code_dir))
            with cd(code_dir):
                run("git pull")
                run("mkdir config")
                put("config/atm.cnf", "config");
                for i in range(1, WORKERS_PER_MACHINE + 1, 1):
                    run("screen -dm -S worker%d python worker.py; sleep 2" % (i,))
        else:
            with cd(code_dir):
                run("git pull")
                for i in range(1, WORKERS_PER_MACHINE + 1, 1):
                    if not run("screen -ls | grep \"worker%d\";" % i):
                        run("screen -dm -S worker%d python worker.py; sleep 2" % (i,))

#@parallel
def killworkers():
    with settings(warn_only=True):
        run("pkill -15 screen")


config = Config('config/atm.cnf')

# fabric env setup
env.user = config.get(Config.AWS, Config.AWS_EC2_USERNAME)
env.key_filename = config.get(Config.AWS, Config.AWS_EC2_KEYFILE)
env.skip_bad_hosts = True
env.colorize_errors = True
env.user = 'ubuntu'
env.pool_size = 4
env.timeout = 10
env.disable_known_hosts = True
env.hosts = query_active_instances()
