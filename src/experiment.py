"""Run experiment.

This module is intended to be run as a script:
    $ python src/experiment.py

"""
import os

import pandas as pd

from bert import BERT
from constants import LANGUAGES, MASK, MISSING
from filenames import CLOZE_DIR, EXPERIMENTS_DIR, FEATURES_DIR
from utils import refresh

ENGLISH_MODEL = 'bert-base-cased'
MULTILINGUAL_MODEL = 'bert-base-multilingual-cased'


def index_of_masked_word(sentence, bert):
    """Return index of the masked word in `sentence` using `bert`'s' tokenizer.

    We use this function to calculate the linear distance between the target
    and controller as BERT sees it.

    Parameters
    ----------
    sentence : str

    Returns
    -------
    int

    """
    tokens = bert.tokenize(sentence)
    try:
        return tokens.index(MASK)
    except ValueError:  # MASK not in sentence
        return -1


def run(language, force_multilingual=False, fold_case=True, gpu=False):
    """Run the experiment for `language`.

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
    # remove any words that aren't in the vocab
    features = features[features['word'].isin(vocab)]
    # if we are masking out the controller, we know that the masked word is
    # also a noun or a pronoun, so we can remove everything else from features
    features = features[features['pos'].isin(['NOUN', 'PRON'])]
    cols = ['number', 'gender', 'case', 'person']
    result = []
    count, total = 0, 0
    for _, example in cloze.iterrows():
        try:
            predictions = bert.predict(example['masked'], fold_case)
        except ValueError:  # MASK not in sentence
            continue
        predictions = features.merge(predictions,
                                     how='left',
                                     left_on='word',
                                     right_index=True)
        # only keep words of the same POS category as the masked word
        predictions = predictions[predictions['pos'] == example['pos']]
        # A word is correct if all its features are identical with the features
        # of the masked word.
        predictions['correct'] = (predictions[cols] == example[cols]).all(
            axis=1)
        # If a word form has multiple feature bundles and at least one of them
        # is correct, then we count that word form as correct. The values of
        # 'p' for the differently valued but identical word forms will be
        # identical (because BERT predicts word forms). I want to include the
        # 'p' in the resulting dataframe so I just take the first value.
        predictions = predictions.groupby('word').agg({
            'correct': any,
            'p': 'first'
        })
        # we compute the average (unnormalized) probability of all the word
        # forms BERT got correct and all it got incorrect.
        mean = predictions.groupby('correct')['p'].mean()
        try:
            example['correct'] = mean[True]
        except KeyError:
            example['correct'] = 0.0
        try:
            example['incorrect'] = mean[False]
        except KeyError:
            example['incorrect'] = 0.0
        # add in the linear distance between masked and other word
        masked_index = index_of_masked_word(example['masked'], bert)
        other_index = index_of_masked_word(example['other_masked'], bert)
        example['distance'] = abs(masked_index - other_index)
        result.append(example)
        if example['correct'] > example['incorrect']:
            count += 1
        total += 1
        if total % print_every == 0:
            percent_correct = round(100 * (count / total), 3)
            percent_done = round(100 * (total / num_examples), 3)
            print(f'{percent_correct}% correct with {percent_done}% done')
    result = pd.DataFrame(result)
    result['right'] = result['correct'] > result['incorrect']
    file_name = os.path.join(EXPERIMENTS_DIR, f'{code}.csv')
    result.to_csv(file_name, index=False)
    return result


if __name__ == '__main__':
    refresh(EXPERIMENTS_DIR)
    # run experiments for languages with fewer cloze examples first
    ORDER = {
        language: len(pd.read_csv(os.path.join(CLOZE_DIR, f'{code}.csv')))
        for language, code in LANGUAGES.items()
    }
    for language in sorted(ORDER, key=ORDER.get):
        try:
            result = run(language)
            proportion_correct = result['right'].value_counts(
                normalize=True)[True]
            print(language, round(proportion_correct, 2))
        except:  # noqa
            print(f'Error with {language}')
