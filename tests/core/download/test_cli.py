from unittest.mock import patch
from cellfinder.core.download.cli import get_parser, main


def test_parser_defaults():
    parser = get_parser()
    args = parser.parse_args([])

    assert args.model == "resnet50_tv"
    assert args.no_amend_config is False
    assert args.install_path is not None


def test_main_calls_download_and_config():
    with patch("sys.argv", ["prog"]):
        with patch("cellfinder.core.download.cli.download_models") as mock_download:
            with patch("cellfinder.core.download.cli.amend_user_configuration") as mock_amend:

                mock_download.return_value = "fake_model_path"

                main()

                mock_download.assert_called_once()
                mock_amend.assert_called_once()


def test_main_no_config_update():
    with patch("sys.argv", ["prog", "--no-amend-config"]):
        with patch("cellfinder.core.download.cli.download_models") as mock_download:
            with patch("cellfinder.core.download.cli.amend_user_configuration") as mock_amend:

                mock_download.return_value = "fake_model_path"

                main()

                mock_download.assert_called_once()
                mock_amend.assert_not_called()
