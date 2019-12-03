"""Prepare cloze examples from UD datasets.

We look through all UD corpora to find examples of agreement relations between
two tokens. When we find one, we mask out one of the words, note the feature
values shared and save to disk.

This module is intended to be run as a script:
    $ python src/cloze.py

"""
import os
from glob import glob
from operator import xor

import pandas as pd
import pyconll

from constants import AGREEMENT_TYPES, DISAGREE, LANGUAGES, MASK, MISSING, NA
from features import feature_value
from filenames import CLOZE_DIR, UNIVERSAL_DEPENDENCIES_DIR
from utils import refresh


def mask(sentence, mask_id):
    """Mask the token at `mask_id` in `sentence`.

    This is complicated by multiword tokens in the Universal Dependencies
    schema. If the form to be masked out is actually part of a multiword token,
    then we need to mask out the multiword token. Any time we see a mutliword
    token (whether we mask it or not), we need to not include the component
    words in the masked out sentence. We don't worry about preserving the
    lack of spaces after certain tokens (i.e. the SpaceAfter field in the
    schema) because BERT's tokenizer splits off punctuation anyway.

    Parameters
    ----------
    sentence : pyconll Sentence
    mask_id : pyconll id

    Returns
    -------
    str

    """
    result = []
    ids_to_skip = []  # ids of the component words of multiword tokens
    for token in sentence:
        if token.form and token.form != '_':
            if not token.is_multiword():
                if token.id not in ids_to_skip:
                    if token.id == mask_id:
                        result.append(MASK)
                    else:
                        result.append(token.form)
                else:  # the word is actually part of a multiword token
                    continue
            else:  # the token is a multiword token
                start, end = token.id.split('-')
                # ids of all the componenet words in the multiword token
                ids = [str(i) for i in range(int(start), int(end) + 1)]
                if mask_id in ids:
                    result.append(MASK)
                else:
                    result.append(token.form)
                ids_to_skip.extend(ids)  # make sure to skip component words
        else:  # there's no form to add
            continue
    return ' '.join(result)


def agreement_value(token1, token2, feature):
    """Return the agreement value of `token1` and `token2` in `feature`.

    I'm using the term agreement value to mean a couple of different things.
    First, it could be a feature value (e.g. singular or first person). The
    agreement value will be a feature value like this if the two tokens have
    the same feature value for that feature. For example, the English words
    "these" and "dogs" will have an agreement value of plural for number
    because they are both plural. Second, it could be the fact that one of the
    tokens is missing a feature value. In this case, we return MISSING. An
    example of this would be English "he" and "tall" for gender, in which only
    "he" is marked for gender. Finally, it could be the fact that the two
    tokens disagree in that feature. This could be an incorrectly annotated
    token. See the comments in constants.py for more discussion.

    We end up removing any cloze examples where one of the values is DISAGREE.

    Parameters
    ----------
    token1, token2 : pyconll Token
    feature : str

    Returns
    -------
    str
        Either a feature value or MISSING or DISAGREE

    """
    value1 = feature_value(token1, feature)
    value2 = feature_value(token2, feature)
    # the two values will be equal if they are geniunely the same feature value
    # (e.g. both "Sing") or if they are both NA.
    if value1 == value2:
        return value1
    value1_is_missing = (value1 == NA)  # don't rely on NA being falsey
    value2_is_missing = (value2 == NA)
    if xor(value1_is_missing, value2_is_missing):
        return MISSING
    return DISAGREE


def intervening_noun(sentence, target, controller):
    """Return True if a noun intervenes between `target` and `controller`."""
    slice = sentence[target.id:controller.id][1:]
    for token in slice:
        if token.upos == 'NOUN':
            return True
    return False


def agree(token1, token2):
    """Return True if `token1` and `token2` agree."""
    for feature in ['number', 'gender', 'case', 'person']:
        value = agreement_value(token1, token2, feature)
        if value == DISAGREE:
            return False
    return True


def count_distractors(sentence, token):
    """Return the number of nouns in `sentence` that don't match `token`."""
    nouns = [t for t in sentence if t.upos == 'NOUN']
    return sum([agree(noun, token) for noun in nouns])


def extract(sentence, target, controller, type_, reverse=False):
    """Extract relevant information for the cloze example.

    Parameters
    ----------
    sentence : pyconll Sentence
    target, controller : pyconll Token
    type_ : str
        Type of agreement relation
    reverse : bool
        We normally mask out the controller, but setting this to True masks out the target

    Returns
    -------
    dict

    """
    example = {'type': type_, 'uid': sentence.id}
    if reverse:
        word_to_mask = target
        other_word = controller
    else:
        word_to_mask = controller
        other_word = target
    example['masked'] = mask(sentence, word_to_mask.id)
    example['pos'] = word_to_mask.upos
    # we want to cloze examples that involve agreement between two tokens,
    # but we also want to allow for missing feature values on the tokens
    # when a language doesn't mark that feature on that token (e.g. English
    # subjects don't mark person). So we note the feature values of the masked
    # word and also note if there's any disagreement between the two tokens.
    # If there is, we'll filter them out.
    example['agree'] = agree(target, controller)
    for feature in ['number', 'gender', 'case', 'person']:
        example[feature] = feature_value(word_to_mask, feature)
    # the data below is used to understand the lingiustic contexts in which
    # performance is affected. we use `other_masked` to calcuate the linear
    # distance between the two words in experiment.py. We may want to only look
    # at distance if there's no intervening noun, as in Linzen et al. (2016).
    # We also count the number of "incorrect" nouns in the sentence.
    example['other_masked'] = mask(sentence, other_word.id)
    example['intervening_noun'] = intervening_noun(sentence, target,
                                                   controller)
    example['num_distractors'] = count_distractors(sentence, word_to_mask)
    return example


def is_determiner_relation(token1, token2):
    """Return True if `token1` is the determiner of `token2`."""
    return (token1.upos == 'DET') and \
           (token1.deprel in ['det', 'det:predet']) and \
           (token2.upos == 'NOUN') and \
           (token1.head == token2.id)


def is_modifying_adjective_relation(token1, token2):
    """Return True if `token1` is an adjective modifying noun `token2`."""
    return (token1.upos == 'ADJ') and \
           (token1.deprel == 'amod') and \
           (token2.upos == 'NOUN') and \
           (token1.head == token2.id)


def is_predicated_adjective_relation(token1, token2):
    """Return True if `token1` is an adjective predicated of `token2`."""
    return (token1.upos == 'ADJ') and \
           (token2.upos in ['NOUN', 'PRON']) and \
           (token2.deprel == 'nsubj') and \
           (token2.head == token1.id)


def is_verb_relation(token1, token2):
    """Return True if `token1` is a verb with subject `token2`."""
    return (token1.upos == 'VERB') and \
           (token2.upos in ['NOUN', 'PRON']) and \
           (token2.deprel == 'nsubj') and \
           (token2.head == token1.id)


def is_copula_relation(token1, token2):
    """Return True if `token1` is a copula dependent of `token2`.

    We don't want to capture cases where `token2` is an adjective, because
    we capture those in `is_predicated_adjective_relation()`.

    """
    return (token1.deprel == 'cop') and \
           (token2.upos != 'ADJ') and \
           (token1.head == token2.id)


def is_auxiliary_relation(token1, token2):
    """Return True if `token1` is an auxiliary dependent of `token2`."""
    return (token1.upos == 'AUX') and \
           (token1.deprel in ['aux', 'aux:pass']) and \
           (token2.upos == 'VERB') and \
           (token1.head == token2.id)


def is_subject(token1, token2):
    """Return True if `token1` is the subject of `token2`."""
    return (token1.upos in ['NOUN', 'PRON']) and \
           (token1.deprel == 'nsubj') and \
           (token1.head == token2.id)


def find_subject(token, sentence):
    """Return the subject of `token` in sentence."""
    for potential_subject in sentence:
        if is_subject(potential_subject, token):
            return potential_subject
    return None


def collect_agreement_relations(fname):
    """Prepare cloze examples from `fname`.

    Agreement relations are an overt morphophonological co-variance of feature
    values between two tokens. We say that one of the tokens (the target)
    agrees with the other (the controller) in a set of features. In this work,
    we are interested in four features which are commonly involved in agreement
    relations. We are interested in four types of cross-linguistically common
    agreement relations. In the list below, the target comes first and the
    controller second:

        * determiner ~ noun
        * (modifying) adjective ~ noun
        * (predicated) adjective ~ (subject) noun
        * verb(-like) ~ (subject) noun

    Not all languages will exhibit agreement relations in all four types, and
    even when they do the tokens may not agree in all four features (and indeed
    may agree in other features that we're not looking at).

    To collect agreement relations, we loop over the sentences in `file_name`
    looking for instances of the four types listed above (e.g. we look for a
    determiner and its head noun, a predicated adjective and its subject).
    Each instance we find is a potential agreement relation. For every instance
    we find, we extract the agreement values of the two tokens (see the
    agreement_value function for exactly what I mean by this). Provided the
    instance has at least one genuine agreement value then we will keep it.

    Parameters
    ----------
    fname : str

    Returns
    -------
    pd.DataFrame
        Contains the type of agreement relation, POS of the masked word, the
        agreement values for the four features, and the masked sentence

    """
    conll = pyconll.iter_from_file(fname)
    result = []
    for sentence in conll:
        for token in sentence:
            try:
                head = sentence[token.head]
            except (KeyError, ValueError):
                # problem with the underlying file or annotation
                continue
            if is_determiner_relation(token, head):
                instance = extract(sentence, token, head, 'determiner')
                result.append(instance)
                instance = extract(
                    sentence, token, head, 'determiner', reverse=True
                )  # quick fix to get examples with both maskings
                result.append(instance)
            elif is_modifying_adjective_relation(token, head):
                instance = extract(sentence, token, head, 'modifying')
                result.append(instance)
                instance = extract(sentence,
                                   token,
                                   head,
                                   'modifying',
                                   reverse=True)
                result.append(instance)
            # The Universal Dependency schema annotates a predicated adjective
            # or a verb as the head of a nominal. However, syntactically the
            # adjective/verb is the target of agreement with the nominal. To
            # account for this, if we find one of the next two functions, we
            # pass in `head` as the `token1` and `token` as `token2`.
            elif is_predicated_adjective_relation(head, token):
                instance = extract(sentence, head, token, 'predicated')
                result.append(instance)
                instance = extract(sentence,
                                   head,
                                   token,
                                   'predicated',
                                   reverse=True)
                result.append(instance)
            elif is_verb_relation(head, token):
                instance = extract(sentence, head, token, 'verb')
                result.append(instance)
                instance = extract(sentence, head, token, 'verb', reverse=True)
                result.append(instance)
            # The Universal Dependencies schema annotates copulas as dependents
            # of the predicate, and auxiliaries as dependents of the main verb.
            # However, we want to extract the subjects in these cases, so once
            # we find a copula or auxiliary, we have to go looking for the
            # subject too. The subject is the controller of the agreement,
            # while the copula/auxiliary is the target.
            elif is_copula_relation(token, head):
                subject = find_subject(token, sentence)
                if subject:  # maybe we didn't find a subject
                    instance = extract(sentence, token, subject, 'verb')
                    result.append(instance)
                    instance = extract(sentence,
                                       token,
                                       subject,
                                       'verb',
                                       reverse=True)
                    result.append(instance)
            elif is_auxiliary_relation(token, head):
                subject = find_subject(token, sentence)
                if subject:
                    instance = extract(sentence, token, subject, 'verb')
                    result.append(instance)
                    instance = extract(sentence,
                                       token,
                                       subject,
                                       'verb',
                                       reverse=True)
                    result.append(instance)
    result = pd.DataFrame(result)
    # remove instances with tokens that disagree or have no values for all
    # four features.
    features = ['number', 'gender', 'case', 'person']
    agree = result['agree']
    has_no_values = (result[features] == NA).all(axis=1)
    result = result[agree & ~has_no_values]
    # order columns
    cols = [
        'uid', 'type', 'pos', 'number', 'gender', 'case', 'person', 'masked',
        'other_masked', 'intervening_noun', 'num_distractors'
    ]
    return result[cols]


def prepare(language):
    """Prepare cloze examples for `language`.

    We source cloze examples from the Universal Dependencies corpora for that
    language.

    Parameters
    ----------
    language : str

    Returns
    -------
    pd.DataFrame

    """
    pattern = f'UD_{language}*/*.conllu'
    file_names = glob(os.path.join(UNIVERSAL_DEPENDENCIES_DIR, pattern))
    result = [collect_agreement_relations(fname) for fname in file_names]
    result = pd.concat(result, ignore_index=True, sort=False)
    result.drop_duplicates(inplace=True, subset=['masked', 'type'])
    # filter out automatically harvested cloze examples that are not valid,
    # because we know that this language doesn't have those agreement relations
    valid_types = AGREEMENT_TYPES[language]
    total_num = len(result)
    result = result[result['type'].isin(valid_types)]
    valid_num = len(result)
    return result, total_num, valid_num


if __name__ == '__main__':
    refresh(CLOZE_DIR)
    total = 0
    valid = 0
    for name, code in LANGUAGES.items():
        try:
            cloze, total_num, valid_num = prepare(name)
            total += total_num
            valid += valid_num
            file_name = os.path.join(CLOZE_DIR, f'{code}.csv')
            cloze.to_csv(file_name, index=False)
            print(f'Prepared cloze examples for {name}')
        except ValueError:  # the pd.concat in `prepare` errored
            print(f'No cloze examples found for {name}')
    print(f"{total:,}, {valid:,}, {round(valid/total,3)}")
