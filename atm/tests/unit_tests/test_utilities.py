import socket

import btb.selection.selector
import pytest
from unittest.mock import patch

from atm import utilities
from atm.constants import SELECTORS_MAP


def test_make_selector():
    kwargs = {
        'choices': [1, 2, 3],
        'k': 3,
        'by_algorithm': {'svm': [1, 2], 'rf': [3, 4]}
    }

    for Selector in SELECTORS_MAP.values():
        selector = utilities.get_instance(Selector, **kwargs)
        assert isinstance(selector, btb.selection.selector.Selector)



@patch('atm.utilities.requests')
def test_public_ip_existing(requests_mock):
    # Set-up
    utilities.public_ip = '1.2.3.4'

    # run
    ip = utilities.get_public_ip()

    # asserts
    assert ip == utilities.public_ip
    requests_mock.get.assert_not_called()


def test_public_ip_success():
    # Set-up
    utilities.public_ip = None

    # run
    ip = utilities.get_public_ip()

    # asserts
    assert ip == utilities.public_ip
    try:
        socket.inet_aton(ip)
    except socket.error:
        pytest.fail("Invalid IP address")


@patch('atm.utilities.requests.get', side_effect=Exception)
def test_public_ip_fail(mock_get):
    # Set-up
    utilities.public_ip = None

    # run
    ip = utilities.get_public_ip()

    # asserts
    assert ip == utilities.public_ip
    assert ip == 'localhost'
    mock_get.assert_called_once_with(utilities.PUBLIC_IP_URL)
