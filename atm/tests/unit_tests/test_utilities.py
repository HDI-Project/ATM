import socket

import pytest
from mock import patch

from atm import utilities


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


@patch('atm.utilities.requests')
def test_public_ip_fail(requests_mock):
    # Set-up
    utilities.public_ip = None
    requests_mock.get.side_effect = Exception    # Force fail

    # run
    ip = utilities.get_public_ip()

    # asserts
    assert ip == utilities.public_ip
    assert ip == 'localhost'
    requests_mock.get.assert_called_once_with(utilities.PUBLIC_IP_URL)
