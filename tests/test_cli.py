from mock import Mock, call, patch

from atm import cli


@patch('atm.cli.get_demos')
def test__get_demos(mock_get_demos):
    """Test that the method get_demos is being called properly."""

    # run
    cli._get_demos(None)  # Args are not being used.

    # assert
    mock_get_demos.assert_called_once_with()


@patch('atm.cli.print')
@patch('atm.cli.download_demo')
def test__download_demo(mock_download_demo, mock_print):
    """Test that the method _download_demo is being called properly with a single dataset."""

    # setup
    args_mock = Mock(dataset='test.csv', path=None)
    mock_download_demo.return_value = 'test.csv'

    # run
    cli._download_demo(args_mock)

    # assert
    mock_download_demo.assert_called_once_with('test.csv', None)
    mock_print.assert_called_once_with('Dataset has been saved to test.csv')


@patch('atm.cli.print')
@patch('atm.cli.download_demo')
def test__download_demo_array(mock_download_demo, mock_print):
    """Test that the method _download_demo is being called properly with a two datasets."""

    # setup
    args_mock = Mock(dataset=['test.csv', 'test2.csv'], path=None)
    mock_download_demo.return_value = ['test.csv', 'test2.csv']

    # run
    cli._download_demo(args_mock)

    # assert
    expected_print_calls = [
        call('Dataset has been saved to test.csv'),
        call('Dataset has been saved to test2.csv')
    ]

    mock_download_demo.assert_called_once_with(['test.csv', 'test2.csv'], None)
    assert mock_print.call_args_list == expected_print_calls


@patch('atm.cli.print')
@patch('atm.cli.download_demo')
def test__download_demo_path(mock_download_demo, mock_print):
    """Test that the method _download_demo is being called properly with a given path."""

    # setup
    args_mock = Mock(dataset=['test.csv', 'test2.csv'], path='my_test_path')
    mock_download_demo.return_value = ['test.csv', 'test2.csv']

    # run
    cli._download_demo(args_mock)

    # assert
    expected_print_calls = [
        call('Dataset has been saved to test.csv'),
        call('Dataset has been saved to test2.csv')
    ]

    mock_download_demo.assert_called_once_with(['test.csv', 'test2.csv'], 'my_test_path')
    assert mock_print.call_args_list == expected_print_calls
