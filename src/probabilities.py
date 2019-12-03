"""Get predicted words cloze examples.

This module is intended to be run as a script:
    $ python src/probabilities.py

"""
import os

import pandas as pd
from tqdm import tqdm

from bert import BERT
from constants import LANGUAGES, MASK, MISSING
from filenames import CLOZE_DIR, FEATURES_DIR, PROBABILITIES_DIR

ENGLISH_MODEL = 'bert-base-cased'
MULTILINGUAL_MODEL = 'bert-base-multilingual-cased'


def run(language, force_multilingual=False, fold_case=True, gpu=True):
    """Get predicted words cloze examples for `language`.

    Parameters
    ----------
    language : str
    force_multilingual : bool
        Whether to use the multilingual model even on English
    fold_case : bool
        Whether to ignore caseing differences after making predictions
    gpu : bool
        Whether to run on GPU or not (useful for debugging)

    Returns
    -------
    pd.DataFrame

    """
    if (language == 'English') and (not force_multilingual):
        bert = BERT(ENGLISH_MODEL, gpu=gpu)
    else:
        bert = BERT(MULTILINGUAL_MODEL, gpu=gpu)
    vocab = bert.vocab
    if fold_case:
        vocab = [word.lower() for word in vocab]
    code = LANGUAGES[language]
    cloze = pd.read_csv(os.path.join(CLOZE_DIR, f'{code}.csv'))
    num_examples = len(cloze)
    print(f'\n\nNumber of examples for {language}: {num_examples}')
    print_every = num_examples // 100
    features = pd.read_csv(os.path.join(FEATURES_DIR, f'{code}.csv'),
                           dtype={'person': str})
    features_vocab = set(features["word"])
    cols = ['number', 'gender', 'case', 'person']
    result = []
    os.makedirs(os.path.join(PROBABILITIES_DIR, code), exist_ok=True)
    for _, example in tqdm(cloze.iterrows()):
        # guard against inputs too long for this implementation
        length = len(bert.tokenize(example["masked"]))
        if length > 512:
            continue
        for mask in ["masked", "other_masked"]:
            try:
                predictions = bert.predict(example[mask], fold_case)
            except ValueError:  # MASK not in sentence
                continue
            # drop words we don't have features for
            predictions = predictions[predictions.index.isin(features_vocab)]
            file_name = f'{example["uid"]}.csv'
            if mask == "other_masked":
                file_name = "reverse-" + file_name
            file_name = os.path.join(PROBABILITIES_DIR, code, file_name)
            predictions.to_csv(file_name)


if __name__ == '__main__':
    # get probabilities for languages with fewer cloze examples first
    already_done = ["bre", "hun", "hye", "tam", "tel", "tur"]
    ORDER = {
        language: len(pd.read_csv(os.path.join(CLOZE_DIR, f'{code}.csv')))
        for language, code in LANGUAGES.items() if code not in already_done
    }
    ORDER = {"Czech":0, "German":1}
    for language in sorted(ORDER, key=ORDER.get):
        try:
            result = run(language)
            print(f'Finished with {language}')
        except:  # noqa
            print(f'Error with {language}')
