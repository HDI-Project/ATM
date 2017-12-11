import json
from importlib import import_module
from os.path import join

from btb import HyperParameter

CONFIG_PATH = 'methods'


class Hyperpartition(object):
    """
    Class which holds the hyperparameter settings that define a hyperpartition.
    """
    def __init__(self, categoricals, constants, tunables):
        """
        categoricals: the values for this hyperpartition which have been fixed, thus
            defining the hyperpartition
        constants: the values for this hyperpartition for which there was no choice
        tunables: the free variables which must be tuned
        """
        self.categoricals = categoricals
        self.constants = constants
        self.tunables = tunables


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
        print "loading file", config
        with open(join(CONFIG_PATH, config)) as f:
            config = json.load(f)

        self.name = config['name']
        self.conditions = config['conditions']
        self.root_params = config['root_parameters']

        # import the method's python class
        path = config['class'].split('.')
        mod_str, cls_str = '.'.join(path[:-1]), path[-1]
        mod = import_module(mod_str)
        self.class_ = getattr(mod, cls_str)

        # create hyperparameters from the parameter config
        self.parameters = {k: HyperParameter(**v) for k, v in
                           config['parameters'].items()}


    def get_hyperpartitions(self):
        """
        Traverse the CPT and enumerate all possible hyperpartitions of parameters
        for this method
        """
        constants = [(p, self.parameters[p].range[0]) for p in self.root_params
                     if len(self.parameters[p].range) == 1]
        categoricals = [p for p in self.root_params
                        if self.parameters[p].is_categorical]
        tunables = [(p, self.parameters[p]) for p in self.root_params
                    if not self.parameters[p].is_categorical]

        return self._enumerate([], constants, categoricals, tunables)

    def _enumerate(self, fixed_cats, constants, free_cats, tunables):
        """
        Some things are fixed. Make a choice from the things that aren't fixed
        and see where that leaves us. Recurse.

        fixed_cats: a list of (name, value) tuples of qualified categorical
            variables
        constants: a list of (name, value) tuples of fixed constants
        free_cats: a list of names of free categorical variables
        tunables: a list of names of free tunable parameters

        Returns: a list of Hyperpartition objects
        """
        # if there are no more free variables, we have a new Hyperpartition. We've
        # reached the bottom of the recursion, so return.
        if not free_cats:
            return [Hyperpartition(fixed_cats, constants, tunables)]

        parts = []

        # fix a single categorical parameter, removing it from the list of free
        # variables, and see where that takes us
        cat = free_cats.pop(0)

        for val in self.parameters[cat].range:
            # add this value to the list of qualified categoricals
            new_fixed_cats = fixed_cats + [(cat, val)]

            # these lists are copied for now
            new_constants = constants[:]
            new_free_cats = free_cats[:]
            new_tunables = tunables[:]

            # check if choosing this value opens up new parts of the conditional
            # parameter tree.
            if cat in self.conditions and val in self.conditions[cat]:
                conditionals = self.conditions[cat][val]

                # categorize the conditional variables which are now in play
                new_constants += [
                    (p, self.parameters[p].range[0]) for p in conditionals
                    if len(self.parameters[p].range) == 1]
                new_free_cats += [p for p in conditionals
                                  if self.parameters[p].is_categorical]
                new_tunables += [(p, self.parameters[p]) for p in conditionals
                                 if not self.parameters[p].is_categorical]

            # recurse with the newly qualified categorical as a constant
            parts.extend(self._enumerate(fixed_cats=new_fixed_cats,
                                         constants=new_constants,
                                         free_cats=new_free_cats,
                                         tunables=new_tunables))

        return parts
