import argparse

BRAINGLOBE_WORKFLOWS = "https://github.com/brainglobe/brainglobe-workflows"
NEW_NAME = "brainmapper"
BLOG_POST = "https://brainglobe.info/blog/version1/core_and_napari_merge.html"


def cli_catch() -> None:
    """
    Will be executed if the user attempts to run 'cellfinder' from the
    command-line.
    Reports that this CLI tool is now only available as a workflow from
    brainglobe-workflows, and has a new name to boot.
    Relevant links are provided.

    Argparse is used so that the CLI arguments are caught, but they are
    not processed.
    The --help feature is overwritten to force argparse to display the
    migration message regardless of user input.
    """
    parser = argparse.ArgumentParser(
        "cellfinder",
        description=(
            "Migration warning if users are trying to use the cellfinder"
            " name for the workflow/ CLI tool now provided by"
            " brainglobe-workflows."
        ),
        add_help=False,
    )
    parser.add_argument(
        "catch-all",
        nargs="*",
        help=(
            "Catch any input arguments to prevent error throws."
            " All arguments are ignored."
        ),
    )

    print(
        "Hey, it looks like you're trying to run the old command-line tool.",
        "This workflow has been renamed and moved -",
        " you can now find it in the brainglobe-workflows package:\n",
        f"\t{BRAINGLOBE_WORKFLOWS}\n",
        f"\tusing the name {NEW_NAME}\n",
        f"For more information, see our blog post: {BLOG_POST}",
    )
    return
