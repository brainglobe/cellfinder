from pathlib import Path
from typing import Optional

import pytest


def test_main():
    """
    Test main function for setting up and running cellfinder workflow
    with no inputs
    """
    # import inside the test function so that required functions are
    # monkeypatched first
    from workflows.cellfinder import main

    # run main
    cfg = main()

    # check output files exist
    assert Path(cfg._detected_cells_path).is_file()


@pytest.mark.parametrize(
    "input_config",
    [
        "config_local_json",
        "config_GIN_json",
    ],
)
def test_main_w_inputs(
    input_config: Optional[str],
    request: pytest.FixtureRequest,
):
    """
    Test main function for setting up and running cellfinder workflow
    with inputs

    Parameters
    ----------
    input_config : Optional[str]
        Path to input config JSON file
    request : pytest.FixtureRequest
        Pytest fixture to enable requesting fixtures by name
    """
    from workflows.cellfinder import main

    # run main
    cfg = main(str(request.getfixturevalue(input_config)))

    # check output files exist
    assert Path(cfg._detected_cells_path).is_file()
