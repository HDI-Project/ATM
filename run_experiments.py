from delphi.database import *
from delphi.utilities import *
from delphi.cloud import Openstack
from delphi.run import Run
import paramiko
import traceback, sys
import glob

algorithm_codes = ['classify_svm', 'classify_rf', 'classify_sgd', 'classify_gnb', 'classify_dt',
					'classify_mnb', 'classify_knn', 'classify_logreg', 'classify_et']

algorithm_codes = ['classify_svm', 'classify_logreg', 'classify_rf', 
	'classify_sgd', 'classify_gnb', 'classify_mnb', 
	'classify_knn', 'classify_et']

datafiles = [
	#"mooc_nlp5.csv",
	#("letters_train.csv", "letters_test.csv"),
	#"derya_cell.csv",
	#("cifar_10_train.csv", "cifar_10_test.csv"),
	#("data/multiclass/images_train.csv", "data/multiclass/images_test.csv"),
	#("data/multiclass/letters_train.csv", "data/multiclass/letters_test.csv"),
	#("data/multiclass/cifar_10_train.csv", "data/multiclass/cifar_10_test.csv"),
	#("data/processed/contraception_use_train.csv", "data/processed/contraception_use_test.csv"),
	#"data/binary/japan_credit.csv",
	#"data/binary/ionosphere.csv",
	"data/binary/mooc_forum_binary.csv",
]

#prefix = "data/mooc"
#datafiles = glob.glob("%s/*.csv" % prefix)
#datafiles = sorted(datafiles)
#testing = datafiles[::2]
#training = datafiles[1::2]
#datafiles = zip(training, testing)

#datafiles = [datafiles[0]]

selection_pairs = [
	# frozen, sample, k, r
	#("uniform", "uniform", 5, 5),
	#("uniform", "uniform", 5, 10),
	#("uniform", "gp_ei", 5, 2),
	("bestkvel", "gp_ei", 5, 5),
	#("hieralg", "gp_ei", 5, 2),
	#("purebestkvel", "gp_ei", 5, 2),
]

n_runs = 10 # each setting will do this many runs
nlearners = 500
budget = "learner"
metric = "cv"

rdicts = []
for datafile in datafiles:
	for pair in selection_pairs:
		frozen, sample, k_window, r_min = pair
		for i in range(n_runs):
			rdict = {
				"metric" : metric,
				"r_min" : r_min,
				"algorithm_codes" : algorithm_codes,
				"k_window" : k_window,
				"sample_selection" : sample,
				"frozen_selection" : frozen,
				"description" : "__".join([frozen, sample]) + "__Ensembles",
				"budget_type" : budget,
				"priority" : 25002,
				"learner_budget" : nlearners,
				"frozens_separately" : False}

			if isinstance(datafile, tuple):
				rdict["trainpath"] = datafile[0] #prefix + datafile[0]
				rdict["testpath"] = datafile[1] #prefix + datafile[1]
				runname = os.path.basename(datafile[0])
				runname = runname.replace("_train", "")
				runname = runname.replace(".csv", "")
				rdict["runname"] = runname
			else:
				rdict["alldatapath"] = datafile #prefix + datafile
				rdict["runname"] = os.path.basename(datafile).replace(".csv", "")

			print rdict
			print 

			rdicts.append(rdict)

runs = []
for rdict in rdicts:
	datarun_ids = Run(**rdict)
	runs.extend(datarun_ids)

print "RUNS", runs


# configuration and launcher
sys.exit()
'''
runs = []
for i in range(268, 291, 1):
	runs.append(i)
'''

config = Config("config/experiments.cnf")
cloud = Openstack(config)
hostnames = cloud.get_ip_addrs("delphi-worker", "alfa-will-desktop")
random.shuffle(hostnames)

print "There are %d unique hostnames" % len(hostnames)
k = paramiko.RSAKey.from_private_key_file("config/alfa-will-desktop.pem")

hcounter = {}
overcounts = len(hostnames)

print "runs", runs

i = 0
while runs:

	datarun_id = runs.pop(0)
	print "Assigning", datarun_id

	c = None
	try:

		hostname = hostnames[i % len(hostnames)]

		if hostname in hcounter and hcounter[hostname] == 3:
			overcounts -= 1
			if not overcounts:
				print "Ran out of overcounts..."
				break
			continue

		c = paramiko.SSHClient()
		c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
		c.connect(hostname=hostname, username="root", pkey=k)
		print "connected to %s" % hostname

		command = '''
		export A=%d;
		cd /root/delphi/; 
		git pull;
		screen -dm -S worker$A_1 python worker.py -d $A; sleep 2; 
		screen -dm -S worker$A_2 python worker.py -d $A; sleep 2;
		screen -dm -S worker$A_3 python worker.py -d $A; sleep 2;
		''' % datarun_id

		print "Executing {}".format( command )
		stdin , stdout, stderr = c.exec_command(command)
		print stdout.read()

		if not hostname in hcounter:
			hcounter[hostname] = 1
		else:
			hcounter[hostname] += 1
		i += 1

	except Exception:
		print traceback.format_exc()
		runs.append(datarun_id)

	finally:
		if c:
			c.close()
		
with open('hostnames-not-used.txt', 'w') as hf:
	for hostname in hostnames:
		if not hostname in hcounter:
			print "%s never used" % hostname
			hf.write("%s\n" % hostname)

print "Runs never started: %s" % runs
