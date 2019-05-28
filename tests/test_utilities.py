from btb.selection.selector import Selector

from atm import utilities
from atm.constants import SELECTORS


def test_make_selector():
    kwargs = {
        'choices': [1, 2, 3],
        'k': 3,
        'by_algorithm': {'svm': [1, 2], 'rf': [3, 4]}
    }

    for selector_class in SELECTORS.values():
        selector = utilities.get_instance(selector_class, **kwargs)
        assert isinstance(selector, Selector)
