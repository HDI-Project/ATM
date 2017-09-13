from atm.config import Config
import pickle
import urllib2
import platform, socket
import hashlib
import numpy as np
import os, random
import base64
from boto.s3.connection import S3Connection, Key as S3Key

PUBLIC_IP_URL = "http://ifconfig.me/ip"  # "http://ipecho.net/plain"


def HashDict(dictionary, ignored_keys=[]):
    """
    http://stackoverflow.com/questions/5884066/hashing-a-python-dictionary
    """
    dictionary = dict(dictionary)  # copy dictionary
    for key in ignored_keys:
        del dictionary[key]
    return hashlib.md5(repr(sorted(dictionary.items()))).hexdigest()


def HashNestedTuple(tup):
    return hashlib.md5(repr(sorted(tup))).hexdigest()


def HashString(s):
    return hashlib.md5(str(s)).hexdigest()


def IsDictSubset(sup, sub):
    """
    Tests if one dictionary is a subset of another.
    """
    return all(item in sup.items() for item in sub.items())


def IsNumeric(obj):
    """
    http://stackoverflow.com/questions/500328/identifying-numeric-and-array-types-in-numpy
    """
    if obj is True or obj is False or obj is None or type(obj) == np.bool_:
        return False
    try:
        obj + obj, obj * obj, obj - obj
    except Exception:
        return False
    else:
        return True


def StringToSpecial(string):
    if string == 'True':
        return True
    elif string == 'False':
        return False
    elif string == 'None':
        return None
    elif type(string) == np.string_:
        return str(string)
    else:
        return string


def EnsureDirectory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def MakeModelPath(model_dir, params_hash, run_hash, desc):
    return os.path.join(model_dir, "%s-%s-%s.model" % (run_hash, params_hash, desc))


def MakeMetricPath(model_dir, params_hash, run_hash, desc):
    return os.path.join(model_dir, "%s-%s-%s.metric" % (run_hash, params_hash, desc))


def SaveMetric(metric_path, object):
    with open(metric_path, 'wb') as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)


def GetMetric(metric_path):
    with open(metric_path, 'rb') as handle:
        obj = pickle.load(handle, protocol=pickle.HIGHEST_PROTOCOL)

    return obj


def CreateCSVHeader(n_features, name, class_label_name):
    """
        Creates a CSV header like:
            "<class_label_name>, <name>1, <name>2, ..., <name><n_features>"

        Example:
            print CreateCSVHeader(64, "pixel", "label")
    """
    separator = ","
    header_row_string = separator.join(
        [x + str(y) for (x, y) in
         zip([name for i in range(n_features)], range(1, n_features + 1, 1))])
    return separator.join([class_label_name, header_row_string])


def GetMemory():
    """
    Get node total memory and memory usage
    """
    with open('/proc/meminfo', 'r') as mem:
        ret = {}
        tmp = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) == 'MemTotal:':
                ret['total'] = int(sline[1])
            elif str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                tmp += int(sline[1])
        ret['free'] = tmp
        ret['used'] = int(ret['total']) - int(ret['free'])
    return ret


def GetPublicIP():
    try:
        return urllib2.urlopen(PUBLIC_IP_URL).read().strip()
    except Exception:  # any exception, doesn't matter
        return None


def GetInfo():
    """
        Returns dictionary about information of this
        computer for statistics.
    """
    # how much free memory do we have?
    memory = GetMemory()

    return {
        "trainer_ip_addr": GetPublicIP(),
        "trainer_hostname": socket.gethostname(),
        "trainer_info": " ".join(platform.uname()),
        "trainer_free_memory": memory["free"],
        "trainer_total_memory": memory["total"],
    }


def ObjectToBase64(obj):
    """
        Takes object in memory, then pickles and
        base64 encodes it.
    """
    pickled = pickle.dumps(obj)
    return base64.b64encode(pickled)


def Base64ToObject(b64str):
    """
        Inverse of ObjectToBase64.

        Decodes base64 encoded string and
        then de-pickles it.
    """
    decoded = base64.b64decode(b64str)
    return pickle.loads(decoded)


def DownloadFileS3(config, keyname):
    aws_key = config.get(Config.AWS, Config.AWS_ACCESS_KEY)
    aws_secret = config.get(Config.AWS, Config.AWS_SECRET_KEY)

    conn = S3Connection(aws_key, aws_secret)
    s3_bucket = config.get(Config.AWS, Config.AWS_S3_BUCKET)
    bucket = conn.get_bucket(s3_bucket)

    if config.get(Config.AWS, Config.AWS_S3_FOLDER) and not config.get(Config.AWS, Config.AWS_S3_FOLDER).isspace():
        aws_keyname = os.path.join(config.get(Config.AWS, Config.AWS_S3_FOLDER), keyname)
    else:
        aws_keyname = keyname


    s3key = S3Key(bucket)
    s3key.key = aws_keyname

    s3key.get_contents_to_filename(keyname)

    return keyname


def DownloadFileHTTP(url, verbose=False):
    """
    http://stackoverflow.com/questions/22676/how-do-i-download-a-file-over-http-using-python
    """
    original_filename = url.split('/')[-1]
    filename = original_filename
    u = urllib2.urlopen(url)
    f = open(filename, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print "Downloading: %s Bytes: %s" % (filename, file_size)

    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        status = status + chr(8) * (len(status) + 1)
        if verbose:
            print status

    f.close()
    return original_filename


def ATMToScikit(learner_params):
    """
        TODO: Make this logic into subclasses

        ORRRR, should make each enumerator handle it in a static function
        something like:

        @staticmethod
        def ParamsTransformation(params_dict):
            # does learner-specific changes
            return params_dict

    """
    # do special converstions

    ### RF ###
    if "n_jobs" in learner_params:
        learner_params["n_jobs"] = int(learner_params["n_jobs"])
    if "n_estimators" in learner_params:
        learner_params["n_estimators"] = int(learner_params["n_estimators"])

    ### DBN ###
    if learner_params["function"] == "classify_dbn":

        learner_params["layer_sizes"] = [learner_params["inlayer_size"]]

        # set layer topology
        if learner_params["num_hidden_layers"] == 1:
            learner_params["layer_sizes"].append(learner_params["hidden_size_layer1"])
            del learner_params["hidden_size_layer1"]

        elif learner_params["num_hidden_layers"] == 2:
            learner_params["layer_sizes"].append(learner_params["hidden_size_layer2"])
            del learner_params["hidden_size_layer1"]
            del learner_params["hidden_size_layer2"]

        elif learner_params["num_hidden_layers"] == 3:
            learner_params["layer_sizes"].append(learner_params["hidden_size_layer3"])
            del learner_params["hidden_size_layer1"]
            del learner_params["hidden_size_layer2"]
            del learner_params["hidden_size_layer3"]

        learner_params["layer_sizes"].append(learner_params["outlayer_size"])
        learner_params["layer_sizes"] = [int(x) for x in learner_params["layer_sizes"]]  # convert to ints
        learner_params["epochs"] = int(learner_params["epochs"])

        # delete our fabricated keys
        del learner_params["num_hidden_layers"]
        del learner_params["inlayer_size"]
        del learner_params["outlayer_size"]

    # remove function key and return
    del learner_params["function"]
    return learner_params


def ObjHasMethod(obj, method):
    """http://stackoverflow.com/questions/34439/finding-what-methods-an-object-has"""
    return hasattr(obj, method) and callable(getattr(obj, method))


def GroupBy(objects, attribute):
    """Groups objects by their attribute values"""
    attr2objs = {}  # string attr => [obj, obj, ...]
    for obj in objects:

        if type(obj) is dict:
            attr_val = obj[attribute]
        else:
            attr_val = getattr(obj, attribute)

        if not attr_val in attr2objs:
            attr2objs[attr_val] = []
        attr2objs[attr_val].append(obj)

    return attr2objs


def GetBests(sequence, initBest=0.0):
    best = initBest
    bests = []
    for elt in sequence:
        if elt > best:
            best = elt
        bests.append(best)
    return bests


def GetRandomBests(sequence, n=1000, initBest=0.0):
    matrix = np.zeros((n, len(sequence)))
    for i in range(n):
        random.shuffle(sequence)
        bests = GetBests(sequence, initBest)
        matrix[i, :] = np.array(bests)
    return np.mean(matrix, axis=0), np.std(matrix, axis=0)
