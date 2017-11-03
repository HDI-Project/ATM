import numpy as np
import itertools


class Node(object):
    def __init__(self):
        pass

    def combinations(self, parent_choice={}):
        raise NotImplementedError


class Combination(Node):
    def __init__(self, nodes):
        Node.__init__(self)
        self.nodes = nodes

    def combinations(self):
        # get all branches contributions
        choices = []
        for node in self.nodes:
            combinations = node.combinations()
            #print "Combinations %s\n" % combinations
            choices.append(combinations)

        # now do a product between all branches
        allchoice_tuples = list(itertools.product(*choices))
        allchoices = []
        for tuple in allchoice_tuples:
            d = {}
            for item in tuple: # reduce tuples to dictionary
                d.update(item)
            allchoices.append(d)

        return allchoices


class Choice(Node):
    def __init__(self, key, values):
        Node.__init__(self)
        self.key = key
        self.values = values
        self.conditionals = {}

    def add_condition(self, value, nodes):
        self.conditionals[value] = nodes

    def combinations(self):
        # get this node's contributions
        choices = []
        for value in self.values:
            choices.append({self.key : value})

        # recursively incorporate childrens' contributions
        remove = []
        add = []
        for i in range(len(choices)):
            choice = choices[i]
            value = choice[self.key]
            if not type(value) == np.ndarray and value in self.conditionals:
                for condition in self.conditionals[value]:
                    child_choices = condition.combinations()
                    if child_choices:
                        remove.append(i)
                        for child_choice in child_choices:
                            choice_updated = dict(choice)
                            choice_updated.update(child_choice)
                            add.append(choice_updated)

        # filter and return
        choices = [j for i,j in enumerate(choices) if i not in remove]
        choices.extend(add)
        return choices

