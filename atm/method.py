import json
from builtins import str as newstr
from builtins import object
from os.path import join

from atm.constants import METHOD_PATH, METHODS_MAP

import btb


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
        for i, val in enumerate(values):
            if val is None:
                # the value None is allowed for every parameter type
                continue
            if self.type == 'int_cat':
                values[i] = int(val)
            elif self.type == 'float_cat':
                values[i] = float(val)
            elif self.type == 'string':
                # this is necessary to avoid a bug in sklearn, which won't be
                # fixed until 0.20
                values[i] = str(newstr(val))
            elif self.type == 'bool':
                values[i] = bool(val)
        self.values = values

    @property
    def is_categorical(self):
        return True

    @property
    def is_constant(self):
        return len(self.values) == 1

    def as_tunable(self):
        return btb.HyperParameter(typ=self.type, rang=self.values)


class List(HyperParameter):
    def __init__(self, name, type, list_length, element):
        self.name = name
        self.length = Categorical('len(%s)' % self.name, 'int_cat', list_length)
        element_type = HYPERPARAMETER_TYPES[element['type']]
        self.element = element_type('element', **element)

    @property
    def is_categorical(self):
        return True

    def get_elements(self):
        elements = []
        for i in range(max(self.length.values)):
            # generate names for the pseudo-hyperparameters in the list
            elt_name = '%s[%d]' % (self.name, i)
            elements.append(elt_name)

        conditions = {str(i): elements[:i] for i in self.length.values}
        return elements, conditions


class HyperPartition(object):
    """
    Class which holds the hyperparameter settings that define a hyperpartition.
    """
    def __init__(self, categoricals, constants, tunables):
        """
        categoricals: the values for this hyperpartition which have been fixed
            and define the hyperpartition. List of tuples of the form ('param', val).
        constants: the values for this hyperpartition for which there was no
            choice. List of tuples of the form ('param', val).
        tunables: the free variables which must be tuned. List of tuples of the
            form ('param', HyperParameter).
        """
        self.categoricals = categoricals
        self.constants = constants
        self.tunables = tunables

    def __repr__(self):
        cats, cons, tuns = [None] * 3
        if self.categoricals:
            cats = '[%s]' % ', '.join(['%s=%s' % c for c in self.categoricals])
        if self.constants:
            cons = '[%s]' % ', '.join(['%s=%s' % c for c in self.constants])
        if self.tunables:
            tuns = '[%s]' % ', '.join(['%s' % t for t, _ in self.tunables])
        return '<HyperPartition: categoricals: %s; constants: %s; tunables: %s>' % (cats, cons, tuns)


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
    def __init__(self, method):
        """
        method: method code or path to JSON file containing all the information
            needed to specify this enumerator.
        """
        if method in METHODS_MAP:
            # if the configured method is a code, look up the path to its json
            config_path = join(METHOD_PATH, METHODS_MAP[method])
        else:
            # otherwise, it must be a path to a file
            config_path = method

        with open(config_path) as f:
            config = json.load(f)

        self.name = config['name']
        self.root_params = config['root_hyperparameters']
        self.conditions = config['conditional_hyperparameters']
        self.class_path = config['class']

        # create hyperparameters from the parameter config
        self.parameters = {}
        for k, v in config['hyperparameters'].items():
            param_type = HYPERPARAMETER_TYPES[v['type']]
            self.parameters[k] = param_type(name=k, **v)

        # List hyperparameters are special. These are replaced in the
        # CPT with a size hyperparameter and sets of element hyperparameters
        # conditioned on the size.
        for name, param in self.parameters.items():
            if type(param) == List:
                elements, conditions = param.get_elements()
                for e in elements:
                    self.parameters[e] = param.element

                # add the size parameter, remove the list parameter
                self.parameters[param.length.name] = param.length
                del self.parameters[param.name]

                # if this is a root param, replace its name with the new size
                # name in the root params list
                if param.name in self.root_params:
                    self.root_params.append(param.length.name)
                    self.root_params.remove(param.name)

                # if this is a conditional param, replace it there instead
                for var, cond in self.conditions.items():
                    for val, deps in cond.items():
                        if param.name in deps:
                            deps.append(param.length.name)
                            deps.remove(param.name)
                            self.conditions[var][val] = deps

                # finally, add all the potential sets of list elements as
                # conditions of the list's size
                self.conditions[param.length.name] = conditions

    def _sort_parameters(self, params):
        """
        Sort a list of HyperParameter objects into lists of constants,
        categoricals, and tunables.
        """
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
                cons, cats, tuns = self._sort_parameters(new_params)
                new_constants = constants + cons
                new_free_cats = free_cats + cats
                new_tunables = tunables + tuns

            # recurse with the newly qualified categorical as a constant
            parts.extend(self._enumerate(fixed_cats=new_fixed_cats,
                                         constants=new_constants,
                                         free_cats=new_free_cats,
                                         tunables=new_tunables))

        return parts

    def get_hyperpartitions(self):
        """
        Traverse the CPT and enumerate all possible hyperpartitions of
        categorical parameters for this method.
        """
        return self._enumerate([], *self._sort_parameters(self.root_params))
