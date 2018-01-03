import json
import pprint
from os.path import join
import btb

CONFIG_PATH = 'methods'

pp = pprint.PrettyPrinter(indent=4)


class HyperParameter(object):
    @property
    def is_categorical(self):
        return False

    @property
    def is_constant(self):
        return False


class Numeric(HyperParameter):
    def __init__(self, name, type, range):
        self.name = name
        self.type = type
        self.range = range

    @property
    def is_constant(self):
        return len(self.range) == 1

    def as_tunable(self):
        return btb.HyperParameter(typ=self.type, rang=self.range)


class Categorical(HyperParameter):
    def __init__(self, name, type, values):
        self.name = name
        self.type = type
        self.values = values

    @property
    def is_categorical(self):
        return True

    @property
    def is_constant(self):
        return len(self.values) == 1


class List(HyperParameter):
    def __init__(self, name, type, sizes, element):
        self.name = name
        self.size = Categorical(self.name + '_size', 'int_cat', sizes)
        element_type = HYPERPARAMETER_TYPES[element['type']]
        self.element = element_type('element', **element)

    @property
    def is_categorical(self):
        return True

    def get_elements(self):
        elements = []
        for i in range(max(self.size.values)):
            # generate names for the pseudo-hyperparameters in the list
            elt_name = '%s[%d]' % (self.name, i)
            elements.append(elt_name)

        conditions = {str(i): elements[:i] for i in self.size.values}
        return elements, conditions


class HyperPartition(object):
    """
    Class which holds the hyperparameter settings that define a hyperpartition.
    """
    def __init__(self, categoricals, constants, tunables):
        """
        categoricals: the hyperparameter values for this hyperpartition which
            have been fixed, defining the hyperpartition
        constants: the hyperparameters with only one choice
        tunables: the numeric hyperparameters which must be tuned (of type
            btb.HyperParameter)
        """
        self.categoricals = categoricals
        self.constants = constants
        self.tunables = tunables


HYPERPARAMETER_TYPES = {
    'int': Numeric,
    'int_exp': Numeric,
    'float': Numeric,
    'float_exp': Numeric,
    'int_cat': Categorical,
    'float_cat': Categorical,
    'string': Categorical,
    'bool': Categorical,
    'list': List,
}


class Method(object):
    """
    This class is initialized with the name of a json configuration file.
    The config contains information about a classification method and the
    hyperparameter arguments it needs to run. Its main purpose is to generate
    hyperpartitions (possible combinations of categorical hyperparameters).
    """
    def __init__(self, config):
        """
        config: JSON dictionary containing all the information needed to specify
            this enumerator
        """
        with open(join(CONFIG_PATH, config)) as f:
            config = json.load(f)

        self.name = config['name']
        self.conditions = config['conditions']
        self.root_params = config['root_parameters']
        self.class_path = config['class']

        # create hyperparameters from the parameter config
        self.parameters = {}
        lists = []
        for k, v in config['parameters'].items():
            param_type = HYPERPARAMETER_TYPES[v['type']]
            self.parameters[k] = param_type(name=k, **v)

        for name, param in self.parameters.items():
            if type(param) == List:
                elements, conditions = param.get_elements()
                for e in elements:
                    self.parameters[e] = param.element

                # add the size parameter, remove the list parameter
                self.parameters[param.size.name] = param.size
                del self.parameters[param.name]
                self.root_params.append(param.size.name)
                self.root_params.remove(param.name)

                self.conditions[param.size.name] = conditions

    def sort_parameters(self, params):
        constants = []
        categoricals = []
        tunables = []
        for p in params:
            param = self.parameters[p]
            if param.is_constant:
                if param.is_categorical:
                    constants.append((p, param.values[0]))
                else:
                    constants.append((p, param.range[0]))
            elif param.is_categorical:
                categoricals.append(p)
            else:
                tunables.append((p, param.as_tunable()))

        return constants, categoricals, tunables

    def get_hyperpartitions(self):
        """
        Traverse the CPT and enumerate all possible hyperpartitions of parameters
        for this method
        """
        return self._enumerate([], *self.sort_parameters(self.root_params))

    def _enumerate(self, fixed_cats, constants, free_cats, tunables):
        """
        Some things are fixed. Make a choice from the things that aren't fixed
        and see where that leaves us. Recurse.

        fixed_cats: a list of (name, value) tuples of qualified categorical
            variables
        constants: a list of (name, value) tuples of fixed constants
        free_cats: a list of names of free categorical variables
        tunables: a list of names of free tunable parameters

        Returns: a list of HyperPartition objects
        """
        # if there are no more free variables, we have a new HyperPartition. We've
        # reached the bottom of the recursion, so return.
        if not free_cats:
            return [HyperPartition(fixed_cats, constants, tunables)]

        parts = []

        # fix a single categorical parameter, removing it from the list of free
        # variables, and see where that takes us
        cat = free_cats.pop(0)

        for val in self.parameters[cat].values:
            # add this value to the list of qualified categoricals
            new_fixed_cats = fixed_cats + [(cat, val)]

            # these lists are copied for now
            new_constants = constants[:]
            new_free_cats = free_cats[:]
            new_tunables = tunables[:]

            # check if choosing this value opens up new parts of the conditional
            # parameter tree.
            # we need to check conditions for str(val) because all keys in json
            # must be strings.
            if cat in self.conditions and str(val) in self.conditions[cat]:
                # categorize the conditional variables which are now in play
                new_params = self.conditions[cat][str(val)]
                cons, cats, tuns = self.sort_parameters(new_params)
                new_constants = constants + cons
                new_free_cats = free_cats + cats
                new_tunables = tunables + tuns

            # recurse with the newly qualified categorical as a constant
            parts.extend(self._enumerate(fixed_cats=new_fixed_cats,
                                         constants=new_constants,
                                         free_cats=new_free_cats,
                                         tunables=new_tunables))

        return parts
