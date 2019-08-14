"""Prepare feature data from Universal Dependencies and UniMorph datasets.

We need to know the feature values of each word in BERT's vocab. For the
multilingual model, we want to know the feature values for all the languages it
models.

This module is intended to be run as a script:
    $ python src/features.py

"""
import os
from glob import glob

import pandas as pd
import pyconll

from constants import LANGUAGES, NA
from filenames import FEATURES_DIR, UNIMORPH_DIR, UNIVERSAL_DEPENDENCIES_DIR
from utils import refresh

COLS = ['word', 'pos', 'number', 'gender', 'case', 'person']

# Preparing features from UniMorph


def make_mapper(mapping):  # noqa: D202
    """Return function that maps between values in `mapping` if present.

    The Universal Dependencies and UniMorph data use different annotation
    schemas. We need to convert one schema to another. There's a great project
    by Arya McCarthy on converting from the Universal Dependencies schema to
    the UniMorph one, but another library I rely on (pyconll) needs them in
    the original schema. I use the Universal Dependencies data in more places
    than the UniMorph data, so in my case it makes more sense to convert from
    UniMorph to Universal Dependencies.

    Given a set of UniMorph feature values, the returned function looks for
    values that it can map, but returns NA if it doesn't find any.

    Parameters
    ----------
    mapping : dict(str : str)
        Mapping from UniMorph to UD feature values

    Returns
    -------
    function

    """
    def func(features):
        for feature in features:
            if feature in mapping:
                return mapping[feature]
        return NA

    return func


# mappings from UniMorph feature values to Universal Dependencies ones

# these are the only POS that we are interested in for this project
POS_MAPPING = {
    'V': 'VERB',
    'V.PTCP': 'VERB',
    'N': 'NOUN',
    'PRO': 'PRON',
    'ADJ': 'ADJ',
    'ART': 'DET',
    'DET': 'DET',
    'AUX': 'AUX'
}
NUMBER_MAPPING = {'SG': 'Sing', 'PL': 'Plur'}
GENDER_MAPPING = {'MASC': 'Masc', 'FEM': 'Fem', 'NEUT': 'Neut'}
# we restrict our attention to the core case values
CASE_MAPPING = {'NOM': 'Nom', 'ACC': 'Acc', 'ERG': 'Erg', 'ABS': 'Abs'}
PERSON_MAPPING = {'1': '1', '2': '2', '3': '3'}

map_pos = make_mapper(POS_MAPPING)
map_number = make_mapper(NUMBER_MAPPING)
map_gender = make_mapper(GENDER_MAPPING)
map_case = make_mapper(CASE_MAPPING)
map_person = make_mapper(PERSON_MAPPING)


def prepare_um(language):
    """Prepare word feature values from `language` from UniMorph data.

    Some of this function was borrowed from Arya McCarthy's marry.py
    https://github.com/unimorph/ud-compatibility

    Parameters
    ----------
    language : str
        Name of language

    Returns
    -------
    pd.DataFrame
        Contains columns for word form, pos, number, gender, case and person

    """
    code = LANGUAGES[language]
    file_name = os.path.join(UNIMORPH_DIR, code, code)
    result = []
    with open(file_name) as file:
        for line in file:
            if line.split():
                _, inflected, features = line.strip().split('\t')
                features = set(features.split(';'))
                data = {'word': inflected}
                data['pos'] = map_pos(features)
                data['number'] = map_number(features)
                data['gender'] = map_gender(features)
                data['case'] = map_case(features)
                data['person'] = map_person(features)
                result.append(data)
    return pd.DataFrame(result)


# Preparing features from UD

POSSIBLE_FEATURE_VALUES = set(
    list(NUMBER_MAPPING.values()) + list(GENDER_MAPPING.values()) +
    list(CASE_MAPPING.values()) + list(PERSON_MAPPING.values()))


def feature_value(token, feature):
    """Return the value of `feature` in `token`.

    The token may not have a value for the feature, either because the
    language doesn't mark that feature on this kind of token, or because
    the annotation is missing. In this case we return NA.

    Parameters
    ----------
    token : pyconll Token
    feature : str

    Returns
    -------
    str

    """
    feature = feature.title()
    try:
        value = str(next(iter(token.feats[feature])))
        if value in POSSIBLE_FEATURE_VALUES:
            return value
        return NA
    except KeyError:
        return NA


def prepare_one_ud_file(fname):
    """Prepare feature values from `fname` of Universal Dependencies data.

    We look at every token in this file. If the token's POS is one that we care
    about for this project, we extract its feature values.

    Parameters
    ----------
    fname : str

    Returns
    -------
    pd.DataFrame
        Contains columns for word form, pos, number, gender, case and person

    """
    conll = pyconll.iter_from_file(fname)
    result = []
    pos_of_interest = set(POS_MAPPING.values())
    for sentence in conll:
        for token in sentence:
            pos = token.upos
            if pos in pos_of_interest:
                data = {'word': token.form, 'pos': pos}
                for feature in ['number', 'gender', 'case', 'person']:
                    data[feature] = feature_value(token, feature)
                result.append(data)
    return pd.DataFrame(result)


def prepare_ud(language):
    """Prepare feature values from `language` from Universal Dependencies data.

    Parameters
    ----------
    language : str
        Name of language

    Returns
    -------
    pd.DataFrame
        Contains columns for word form, pos, number, gender, case and person

    """
    pattern = os.path.join(UNIVERSAL_DEPENDENCIES_DIR, '**/*.conllu')
    file_names = [f for f in glob(pattern, recursive=True) if language in f]
    result = []
    for file_name in file_names:
        features = prepare_one_ud_file(file_name)
        result.append(features)
    if result:
        return pd.concat(result, ignore_index=True, sort=False)
    return pd.DataFrame([], columns=COLS)


def prepare(language):
    """Prepare word feature values for `language`.

    We source feature values from the Universal Dependencies corpora and the
    UniMorph data. In this work, a word form can take on a particular bundle
    of feature values if either data source tells us it can. The UniMorph data
    directly specifies what feature value bundles a form can take on. For the
    Universal Dependencies data, we say a word can take on a particular bundle
    if we ever see it with that bundle of feature values in a Universal
    Dependencies corpus for that language. Some word forms will have multiple
    entries because syncretism.

    Note that here we lowercase all words. In general, we don't expect feature
    values to differ based on casing. The increased vocab we'll get by
    case-folding will outweigh any errors where our expectation was wrong.
    We case-fold even when using the cased BERT models, so we'll have to check
    against a lowercased version of BERT's vocab.

    Parameters
    ----------
    language : str
        Name of language

    Returns
    -------
    pd.DataFrame
        Contains columns for word form, pos, number, gender, case and person

    """
    ud = prepare_ud(language)
    try:
        um = prepare_um(language)
        result = pd.concat([ud, um], ignore_index=True, sort=False)
    except FileNotFoundError:  # No UniMorph data for this language
        result = ud
    result['word'] = result['word'].str.lower()
    result['person'] = result['person'].astype(str)
    result.drop_duplicates(inplace=True)
    # drop rows with missing words
    result.dropna(subset=['word', 'pos'], how='any', inplace=True)
    # drop rows with no feature values in all four features
    features = ['number', 'gender', 'case', 'person']
    has_no_values = (result[features] == NA).all(axis=1)
    result = result[~has_no_values]
    return result[COLS]


if __name__ == '__main__':
    refresh(FEATURES_DIR)
    for name, code in LANGUAGES.items():
        features = prepare(name)
        file_name = os.path.join(FEATURES_DIR, f'{code}.csv')
        features.to_csv(file_name, index=False)
        print(f'Prepared features for {name}')
