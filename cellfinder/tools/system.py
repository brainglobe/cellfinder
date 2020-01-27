import os
import subprocess

from tempfile import gettempdir
from tqdm import tqdm
from pathlib import Path

from cellfinder.tools.exceptions import CommandLineInputError


def delete_directory_contents(directory, progress=False):
    """
    Removes all contents of a directory
    :param directory: Directory with files to be removed
    """
    files = os.listdir(directory)
    if progress:
        for f in tqdm(files):
            os.remove(os.path.join(directory, f))
    else:
        for f in files:
            os.remove(os.path.join(directory, f))


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


def replace_extension(file, new_extension, check_leading_period=True):
    """
    Replaces the file extension of a given file
    :param str file: Input file with file extension to replace
    :param str new_extension: New file extension
    :param bool check_leading_period: If True, any leading period of the
    new extension is removed, preventing "file..txt"
    :return str: File with new file extension
    """
    if check_leading_period:
        new_extension = remove_leading_character(new_extension, ".")
    return os.path.splitext(file)[0] + "." + new_extension


def remove_leading_character(string, character):
    """
    If "string" starts with "character", strip that leading character away.
    Only removes the first instance
    :param string:
    :param character:
    :return: String without the specified, leading character
    """
    if string.startswith(character):
        return string[1:]
    else:
        return string


# ------ UNTESTED --------------------------------------------------


def safe_execute_command(cmd, log_file_path=None, error_file_path=None):
    """
    Executes a command in the terminal, making sure that the output can
    be logged even if execution fails during the call.

    :param cmd:
    :param log_file_path:
    :param error_file_path:
    :return:
    """
    if log_file_path is None:
        log_file_path = os.path.abspath(
            os.path.join(gettempdir(), "safe_execute_command.log")
        )
    if error_file_path is None:
        error_file_path = os.path.abspath(
            os.path.join(gettempdir(), "safe_execute_command.err")
        )

    with open(log_file_path, "w") as log_file, open(
        error_file_path, "w"
    ) as error_file:
        try:
            subprocess.check_call(
                cmd, stdout=log_file, stderr=error_file, shell=True
            )
        except subprocess.CalledProcessError:
            hline = "-" * 25
            try:
                with open(error_file_path, "r") as err_file:
                    errors = err_file.readlines()
                    errors = "".join(errors)
                with open(log_file_path, "r") as _log_file:
                    logs = _log_file.readlines()
                    logs = "".join(logs)
                raise SafeExecuteCommandError(
                    "\n{0}\nProcess failed:\n {1}"
                    "{0}\n"
                    "{2}\n"
                    "{0}\n"
                    "please read the logs at {3} and {4}\n"
                    "{0}\n"
                    "command: {5}\n"
                    "{0}".format(
                        hline,
                        errors,
                        logs,
                        log_file_path,
                        error_file_path,
                        cmd,
                    )
                )
            except IOError as err:
                raise SafeExecuteCommandError(
                    f"Process failed: please read the logs at {log_file_path} "
                    f"and {error_file_path}; command: {cmd}; err: {err}"
                )


class SafeExecuteCommandError(Exception):
    pass
