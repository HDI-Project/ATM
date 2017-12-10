import json
from importlib import import_module
from os.path import join

config_path = 'classifiers'


class FrozenSet(object):
    def __init__(self, frozens, constants, tunables):
        """
        constants: the values for this frozen set which have been fixed, thus
            defining the frozen set
        constants: the values for this frozen set for which there was no choice
        tunables: the free variables which must be tuned
        """
        self.frozens = frozens
        self.constants = constants
        self.tunables = tunables


class Classifier(object):
    """
    This class is initialized with the name of a json configuration file.
    The config contains information about a classifier and the hyperparameter
    arguments it needs to run. Its main purpose is to generate frozen sets
    (possible combinations of categorical hyperparameters).
    """
    def __init__(self, config):
        """
        config: JSON dictionary containing all the information needed to specify
            this enumerator
        """
        with open(join(CONFIG_PATH, config)) as f:
            config = json.load(f)

        self.name = config['name']
        self.learner_class = import_module(config['class'])
        self.parameters = {k: HyperParameter(**v) for k, v in
                           config['parameters'].items()}
        self.conditions = config['conditions']
        self.root_params = config['root_parameters']

    def get_frozen_sets(self):
        """
        Traverse the CPT and enumerate all possible frozen sets of parameters
        for this algorithm
        """
        constants = [(p, self.parameters[p].range[0]) for p in self.root_params
                     if len(self.parameters[p].range) == 1]
        categoricals = [p for p in self.root_params if
                        self.parameters[p].is_categorical]
        tunables = [p for p in self.root_params if not
                    self.parameters[p].is_categorical]

        frozen_sets = self._enumerate([], constants, categoricals, tunables)

    def _enumerate(frozens, constants, categoricals, tunables):
        """
        Some things are fixed. Make a choice from the things that aren't fixed
        and see where that leaves us. Recurse.

        frozens: a list of (name, value) tuples of qualified categorical
            variables
        constants: a list of (name, value) tuples of fixed constants
        categoricals: a list of names of free categorical variables
        tunables: a list of names of free tunable parameters

        Returns: a list of FrozenSet objects
        """
        # if there are no more free variables, we have a new FrozenSet. We've
        # reached the bottom of the recursion, so return.
        if not categoricals:
            return [FrozenSet(frozens, constants, tunables)]

        fsets = []

        # fix a single categorical parameter, removing it from the list of free
        # variables, and see where that takes us
        cat = categoricals.pop(0)

        for val in self.parameters[cat].range:
            # add this value to the list of qualified categoricals
            new_frozens = frozens + [(cat, val)]

            # these lists are copied for now
            new_constants = constants[:]
            new_categoricals = categoricals[:]
            new_tunables = tunables[:]

            # check if choosing this value opens up new parts of the conditional
            # parameter tree.
            if cat in self.conditions:
                conditionals = self.conditions[cat][val]

                # categorize the conditional variables which are now in play
                new_constants += [
                    (p, self.parameters[p].range[0]) for p in conditionals
                    if len(self.parameters[p].range) == 1]
                new_categoricals += [p for p in conditionals if
                                     self.parameters[p].is_categorical]
                new_tunables += [p for p in conditionals if not
                                 self.parameters[p].is_categorical]

            # recurse with the newly qualified categorical as a constant
            fsets.extend(self._enumerate(frozens=new_frozens,
                                         constants=new_constants,
                                         categoricals=new_categoricals,
                                         tunables=new_tunables))

        return fsets
