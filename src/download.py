"""Download UniMorph datasets.

We use the UniMorph data to supplement the feature values that we derive from
the Universal Dependencies corpora. The UniMorph data is stored on GitHub
(https://github.com/unimorph), so we just download them by cloning the
relevant repos.

This module is intended to be run as a script:
    $ python src/download-um.py

"""
import os

from constants import LANGUAGES
from filenames import UNIMORPH_DIR
from utils import refresh


def download(language):
    """Download the UniMorph dataset for `language`.

    Parameters
    ----------
    language : str
        The ISO 639-3 code for the language

    Returns
    -------
    int
        Exit status of the git command (zero if succesful, non-zero if not)

    """
    url = f'https://github.com/unimorph/{language}.git'
    destination = os.path.join(UNIMORPH_DIR, language)
    command = f'git clone --quiet {url} {destination}'
    return os.system(command)


if __name__ == '__main__':
    refresh(UNIMORPH_DIR)
    for name, code in LANGUAGES.items():
        status = download(code)
        if status == 0:
            print(f'Downloaded UniMorph data for {name}')
        else:
            print(f'No UniMorph data for {name}')
