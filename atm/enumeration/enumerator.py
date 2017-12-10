from importlib import import_module


class FrozenSet(object):
    def __init__(self, constants, tunables):
        """
        constants: the values for this frozen set which are fixed
        tunables: the free variables which must be tuned
        """
        self.constants = constants
        self.tunables = tunables


class Enumerator(object):
    """
    This class is initialized with a list of optimizable Hyperparameters, and
    is used to generate frozen sets (possible combinations of categorical
    hyperparameters).
    """
    def __init__(self, config):
        """
        config: JSON dictionary containing all the information needed to specify
            this enumerator
        """
        self.name = config.name
        self.sklearn_class = import_module(config.class)
        self.parameters = {k: HyperParameter(config.parameters)
        self.conditions = config.conditions
        self.root_params = config.root_parameters

    def get_frozen_sets(self):
        """
        Traverse the CPT and enumerate all possible frozen sets of parameters
        for this algorithm
        """
        constants = [k for k in self.root_params if
                     len(self.parameters[k].range) == 1]
        categoricals = [k for k in self.root_params if
                        self.parameters[k].is_categorical]
        tunables = [k for k in self.root_params if not
                    self.parameters[k].is_categorical]

        frozen_sets = self._enumerate(constants, categoricals, tunables)

    def _enumerate(constants, categoricals, tunables):
        """
        Some things are fixed. Make a choice from the things that aren't fixed and
        see where that leaves us. Recurse.

        constants: a list of (name, value) tuples of fixed constants
        categoricals: a list of names of free categorical variables
        tunables: a list of names of free tunable parameters
        """
        # if there are no more free variables, we have a new FrozenSet. We've
        # reached the bottom of the recursion, so return.
        if not categoricals:
            return [FrozenSet(constants, tunables)]

        frozens = []

        # fix a single categorical parameter and see where that takes us
        cat = categoricals.pop(0)

        for val in self.parameters[cat].range:
            # add this fixed value to the list of constants
            new_constants = constants + [(cat, val)]

            # check if choosing this value opens up new parts of the conditional
            # parameter tree.
            new_categoricals = categoricals[:]
            new_tunables = tunables[:]
            if cat in self.conditions:
                conditionals = self.conditions[cat][val]
                new_constants += [p for p in conditionals if
                                  len(self.parameters[p].range) == 1]
                new_categoricals += [p for p in conditionals if
                                     self.parameters[p].is_categorical]
                new_tunables += [p for p in conditionals if not
                                 self.parameters[p].is_categorical]

            # recurse with the newly qualified categorical as a constant
            frozens.extend(self._enumerate(constants=new_constants,
                                           categoricals=new_categoricals,
                                           tunables=new_tunables))

        return frozens
