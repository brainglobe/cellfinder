from pathlib import Path

from imlib.general.exceptions import CommandLineInputError


def get_subdirectories(directory, names_only=False):
    """
    Return all subdirectories in a given directory
    :param directory:
    :param names_only: If true, dont return paths, but the names
    :return: Subdirectories
    """

    p = Path(directory)
    if names_only:
        return [x.name for x in p.iterdir() if x.is_dir()]
    else:
        return [x for x in p.iterdir() if x.is_dir()]


def get_number_of_files_in_dir(directory):
    """
    Sums the number of files in a directory
    :param directory: Any directory with files
    :return: Number of files in directory
    """
    directory = Path(directory)
    files = directory.iterdir()
    total_files = sum(1 for x in files)
    return total_files


def check_path_exists(file):
    """
    Returns True is a file exists, otherwise throws a FileNotFoundError
    :param file: Input file
    :return: True, if the file exists
    """
    file = Path(file)
    if file.exists():
        return True
    else:
        raise FileNotFoundError


def catch_input_file_error(path):
    """
    Catches if an input path doesn't exist, and returns an informative error
    :param path: Input file path
    default)
    """
    try:
        check_path_exists(path)
    except FileNotFoundError:
        message = (
            "File path: '{}' cannot be found. Please check your input "
            "arguments.".format(path)
        )
        raise CommandLineInputError(message)


def memory_in_bytes(memory_amount, unit):
    """
    Converts an amount of memory (in given units) to bytes

    :param float memory_amount: An amount of memory
    :param str unit: The units ('KB', 'MB', 'GB', 'TB', 'PB')
    :return: Amount of memory in bytes
    """
    if memory_amount is None:
        return memory_amount

    supported_units = {"KB": 3, "MB": 6, "GB": 9, "TB": 12, "PB": 15}

    unit = unit.upper()
    if unit not in supported_units:
        raise NotImplementedError(
            f"Unit: {unit} is not supported. Please "
            f"use one of {list(supported_units.keys())}"
        )
    else:
        return memory_amount * 10 ** supported_units[unit]
