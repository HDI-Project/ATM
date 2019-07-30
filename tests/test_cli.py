from mock import Mock, patch

from atm import cli


@patch('atm.cli.get_demos')
def test__get_demos(mock_get_demos):
    """Test that the method get_demos is being called properly."""

    # run
    cli._get_demos(None)  # Args are not being used.

    # assert
    mock_get_demos.assert_called_once_with()


@patch('atm.cli.download_demo')
def test__download_demo(mock_download_demo):
    """Test that the method _download_demo is being called properly with a single dataset."""

    # setup
    args_mock = Mock(dataset='test.csv', path=None)

    # run
    cli._download_demo(args_mock)

    # assert
    mock_download_demo.assert_called_once_with('test.csv', None)


@patch('atm.cli.download_demo')
def test__download_demo_array(mock_download_demo):
    """Test that the method _download_demo is being called properly with a two datasets."""

    # setup
    args_mock = Mock(dataset=['test.csv', 'test2.csv'], path=None)
    mock_download_demo.return_value = ['test.csv', 'test2.csv']

    # run
    cli._download_demo(args_mock)

    # assert
    mock_download_demo.assert_called_once_with(['test.csv', 'test2.csv'], None)


@patch('atm.cli.download_demo')
def test__download_demo_path(mock_download_demo):
    """Test that the method _download_demo is being called properly with a given path."""

    # setup
    args_mock = Mock(dataset=['test.csv', 'test2.csv'], path='my_test_path')
    mock_download_demo.return_value = ['test.csv', 'test2.csv']

    # run
    cli._download_demo(args_mock)

    # assert
    mock_download_demo.assert_called_once_with(['test.csv', 'test2.csv'], 'my_test_path')


@patch('atm.cli._get_atm')
def test__work(mock__get_atm):
    # setup
    args_mock = Mock(dataruns=[1], total_time=[1], save_files=False, cloud_mode=False)

    # run
    cli._work(args_mock)

    # assert
    mock__get_atm.assert_called_once_with(args_mock)

    mock__get_atm.return_value.work.assert_called_once_with(
        datarun_ids=[1],
        choose_randomly=False,
        save_files=False,
        cloud_mode=False,
        total_time=[1],
        wait=False
    )


@patch('atm.cli.create_app')
@patch('atm.cli._get_atm')
def test__serve(mock__get_atm, mock_create_app):
    # setup
    args_mock = Mock(debug=False, host='1.2.3', port='456')

    # run
    cli._serve(args_mock)

    # assert
    mock__get_atm.assert_called_once_with(args_mock)
    mock_create_app.assert_called_once_with(mock__get_atm.return_value, False)
    mock_create_app.return_value.run.assert_called_once_with(host='1.2.3', port='456')
