

from mock import call, patch

from atm import data


@patch('atm.data.os.getcwd')
@patch('atm.data.os.path.join')
@patch('atm.data.os.path.exists')
@patch('atm.data.os.makedirs')
@patch('atm.data.Config')
@patch('atm.data.boto3')
def test_download_demo_datasets_islist(mock_boto, mock_config, mock_mkdirs, mock_exists, mock_join,
                                       mock_getcwd):
    """Test downloading demo dataset a list of datasets."""

    # setup

    mock_exists.return_value = False
    mock_getcwd.return_value = 'test_dir'
    datasets = ['test_dataset', 'second_test_dataset']

    # run

    result = data.download_demo(datasets)

    # assert

    expected_os_join_calls = [
        call('test_dir', 'demos'),
        call(mock_join.return_value, 'test_dataset'),
        call(mock_join.return_value, 'second_test_dataset'),
    ]

    assert mock_join.call_args_list == expected_os_join_calls
    mock_mkdirs.assert_called_once_with(mock_join.return_value)
    assert result == [mock_join.return_value, mock_join.return_value]


@patch('atm.data.os.getcwd')
@patch('atm.data.os.path.join')
@patch('atm.data.os.path.exists')
@patch('atm.data.os.makedirs')
@patch('atm.data.Config')
@patch('atm.data.boto3')
def test_download_demo_datasets_not_list(mock_boto, mock_config, mock_mkdirs, mock_exists,
                                         mock_join, mock_getcwd):
    """Test downloading only one demo dataset."""

    # setup

    mock_exists.return_value = False
    mock_getcwd.return_value = 'test_dir'
    datasets = 'test_dataset'

    # run

    result = data.download_demo(datasets)

    # assert

    expected_os_join_calls = [
        call('test_dir', 'demos'),
        call(mock_join.return_value, 'test_dataset'),
    ]

    mock_mkdirs.assert_called_once_with(mock_join.return_value)
    assert mock_join.call_args_list == expected_os_join_calls
    assert result == mock_join.return_value


@patch('atm.data.os.path.join')
@patch('atm.data.os.path.exists')
@patch('atm.data.os.makedirs')
@patch('atm.data.Config')
@patch('atm.data.boto3')
def test_download_demo_datasets_with_path(mock_boto, mock_config, mock_mkdirs, mock_exists,
                                          mock_join):
    """Test downloading a demo dataset by giving a path."""

    # setup

    mock_exists.return_value = False
    datasets = 'test_dataset'

    # run

    result = data.download_demo(datasets, path='test_dir')

    # assert

    mock_join.assert_called_once_with('test_dir', 'test_dataset')
    mock_mkdirs.assert_called_once_with('test_dir')  # The actual dir that we pass
    assert result == mock_join.return_value


@patch('atm.data.os.path.join')
@patch('atm.data.os.path.exists')
@patch('atm.data.os.makedirs')
@patch('atm.data.Config')
@patch('atm.data.boto3')
def test_download_demo_dir_exists(mock_boto, mock_config, mock_mkdirs, mock_exists, mock_join):
    """Test downloading a demo dataset and the given directory exists."""

    # setup

    mock_exists.return_value = True

    # run

    result = data.download_demo('test_dataset', path='test_dir')

    # assert

    assert not mock_mkdirs.called
    assert result == mock_join.return_value
