from btb.selection import Selector
import random


class MySelector(Selector):
    def select(self, choice_scores):
        """ Select a choice uniformly at random.  """
        return self.choices[random.randint(0, len(self.choices) - 1)]
