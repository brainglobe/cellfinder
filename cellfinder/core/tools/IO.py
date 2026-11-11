import pooch


def fetch_from_registry(registry: pooch.Pooch, name: str, **kwargs):
    """
    Fetch ``name`` from ``registry``, trying the secure endpoint first.

    If the https download fails with an ``OSError``, retry once against the
    http mirror.
    """
    try:
        return registry.fetch(name, **kwargs)
    except OSError:
        original_url = registry.base_url
        registry.base_url = original_url.replace("https://", "http://")
        try:
            return registry.fetch(name, **kwargs)
        finally:
            registry.base_url = original_url


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
        fetch_from_registry(
            registry,
            name,
            processor=processor,
            downloader=downloader,
            progressbar=progressbar,
        )

    return str(registry.abspath / directory_name)
