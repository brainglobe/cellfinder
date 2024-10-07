import pooch


def fetch_pooch_directory(
    registry: pooch.Pooch,
    directory_name: str,
    processor=None,
    downloader=None,
    progressbar=False,
):
    """
    Fetches files from the Pooch registry that belong to a specific directory.
    Parameters:
        registry (pooch.Pooch): The Pooch registry object.
        directory_name (str):
            The remote relative path of the directory to fetch files from.
        processor (callable, optional):
            A function to process the fetched files. Defaults to None.
        downloader (callable, optional):
            A function to download the files. Defaults to None.
        progressbar (bool, optional):
            Whether to display a progress bar during the fetch.
            Defaults to False.
    Returns:
        str: The local absolute path to the fetched directory.
    """
    names = []
    for name in registry.registry_files:
        if name.startswith(f"{directory_name}/"):
            names.append(name)

    if not names:
        raise FileExistsError(
            f"Unable to find files in directory {directory_name}"
        )

    for name in names:
        registry.fetch(
            name,
            processor=processor,
            downloader=downloader,
            progressbar=progressbar,
        )

    return str(registry.abspath / directory_name)
