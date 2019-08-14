"""Utilities used across different modules."""
import os
import shutil


def refresh(path):
    """Create brand spanking new directory at `path`.

    Parameters
    ----------
    path : str
        Path of directory to refresh

    Returns
    -------
    None

    """
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)
