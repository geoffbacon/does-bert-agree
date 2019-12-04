"""Constant values used across different modules."""

MASK = "[MASK]"  # BERT's mask token

# agreement value used when neither of the two tokens has a feature. This is
# almost certainly because the language does not mark that feature on these
# tokens, but it's also possible that the tokens should have values but the
# annotation is wrong. We use a string value that Pandas won't interpret as a
# NaN because that causes problems in experiment.py (specifically, NaNs don't
# compare equal in Pandas (Python more generally?)).
NA = "NO VALUE"

# agreement value used when the two tokens both have values for a feature but
# the values are different. This is most often because the annotation is wrong,
# but it does happen that languages have grammatical disagreement. One
# phenomenon that results in this is called anti-agreement, in which verbs
# whose subjects have been extracted from their usual position do not agree
# (see Baier 2018).
DISAGREE = "DISAGREE"

# agreement value used when one of the two tokens is missing a feature that
# the other has. This could be because the language does not mark that feature
# on that token (e.g. English pronouns mark gender but predicated adjectives
# do not), or because the annotation is missing.
MISSING = "MISSING"

# mapping from language name to ISO code. The Universal Dependencies data and
# my code uses the language name, while UniMorph uses the ISO code. It turns
# out there's a nice package called pycountry that would have been a more
# flexible solution, but it's not worth re-writing this now.
LANGUAGES = {
    "Afrikaans": "afr",
    "Arabic": "ara",
    "Armenian": "hye",
    "Basque": "eus",
    # 'Belarusian': 'bel',
    "Breton": "bre",
    # 'Bulgarian': 'bul',
    "Catalan": "cat",
    "Croatian": "hrv",
    "Czech": "ces",
    "Danish": "dan",
    "Dutch": "nld",
    "English": "eng",
    # 'Estonian': 'est',
    "Finnish": "fin",
    "French": "fra",
    # 'Galician': 'gal',
    "German": "deu",
    "Greek": "ell",
    "Hebrew": "heb",
    "Hindi": "hin",
    "Hungarian": "hun",
    # 'Indonesian': 'ind',
    "Irish": "gle",
    "Italian": "ita",
    # 'Kazakh': 'kaz',
    # 'Korean': 'kor',
    "Latin": "lat",
    # 'Latvian': 'lav',
    # 'Lithuanian': 'lit',
    # 'Marathi': 'mar',
    # 'Norwegian-Bokmal': 'nob',
    "Norwegian-Nynorsk": "nno",
    "Persian": "fas",
    "Polish": "pol",
    "Portuguese": "por",
    "Romanian": "ron",
    "Russian": "rus",
    # 'Serbian': 'srp',
    # 'Slovak': 'slk',
    # 'Slovenian': 'slv',
    "Spanish": "spa",
    "Swedish": "swe",
    # 'Tagalog': 'tgl',
    "Tamil": "tam",
    "Telugu": "tel",
    # 'Thai': 'tha',
    "Turkish": "tur",
    "Ukrainian": "ukr",
    "Urdu": "urd",
    # 'Yoruba': 'yor'
}

# we use a language-agnostic method of harvesting agreement examples which
# sometimes results in false positives. For example, it yields examples of
# agreement between nouns and modifying adjectives in English which is wrong.
# I believe these come from incorrect annotations in the Universal Dependencies
# data. To remove these, we list here the types of agreement relations present
# in each language sourced from reference grammars and the linguistics
# literature where possible. I'd rather be more conservative with this so I've
# assumed a language doesn't have an agreement type unless it's explicitly
# mentioned somewhere. The possible types are: determiner, modifying,
# predicated and verb.
AGREEMENT_TYPES = {
    "Afrikaans": ["modifying"],  #
    "Arabic": ["modifying", "predicated", "verb"],  #
    "Armenian": ["verb"],  #
    "Basque": ["determiner", "modifying", "predicated", "verb"],  #
    "Belarusian": ["modifying", "predicated", "verb"],
    "Breton": ["verb"],  #
    "Bulgarian": ["determiner", "modifying", "predicated", "verb"],
    "Catalan": ["determiner", "modifying", "predicated", "verb"],  #
    "Croatian": ["modifying", "predicated", "verb"],  #
    "Czech": ["modifying", "predicated", "verb"],  #
    "Danish": ["determiner", "modifying", "predicated"],  #
    "Dutch": ["determiner", "modifying", "verb"],  #
    "English": ["determiner", "verb"],  #
    "Estonian": ["determiner", "modifying", "predicated", "verb"],
    "Finnish": ["determiner", "modifying", "predicated", "verb"],  #
    "French": ["determiner", "modifying", "predicated", "verb"],  #
    "Galician": ["determiner", "modifying", "predicated", "verb"],
    "German": ["determiner", "modifying", "predicated", "verb"],  #
    "Greek": ["determiner", "modifying", "predicated", "verb"],  #
    "Hebrew": ["modifying", "predicated", "verb"],  #
    "Hindi": ["modifying", "predicated", "verb"],  #
    "Hungarian": ["predicated"],  #
    "Indonesian": [],
    "Irish": ["determiner", "modifying", "verb"],  #
    "Italian": ["determiner", "modifying", "predicated", "verb"],  #
    "Kazakh": ["determiner", "modifying", "predicated", "verb"],
    "Korean": ["determiner", "modifying", "predicated", "verb"],
    "Latin": ["determiner", "modifying", "predicated", "verb"],  #
    "Latvian": ["determiner", "modifying", "predicated", "verb"],
    "Lithuanian": ["determiner", "modifying", "predicated", "verb"],
    "Marathi": ["determiner", "modifying", "predicated", "verb"],
    "Norwegian-Bokmal": ["determiner", "modifying", "predicated"],  #
    "Norwegian-Nynorsk": ["determiner", "modifying", "predicated"],  #
    "Persian": ["modifying", "predicated", "verb"],  #
    "Polish": ["modifying", "predicated", "verb"],  #
    "Portuguese": ["determiner", "modifying", "predicated", "verb"],  #
    "Romanian": ["modifying", "predicated", "verb"],  #
    "Russian": ["modifying", "predicated", "verb"],  #
    "Serbian": ["determiner", "modifying", "predicated", "verb"],
    "Slovak": ["determiner", "modifying", "predicated", "verb"],
    "Slovenian": ["determiner", "modifying", "predicated", "verb"],
    "Spanish": ["determiner", "modifying", "predicated", "verb"],  #
    "Swedish": ["modifying", "predicated"],  #
    "Tagalog": [],
    "Tamil": ["modifying", "verb"],  #
    "Telugu": ["modifying", "predicated", "verb"],  #
    "Thai": [],
    "Turkish": ["modifying", "predicated"],  #
    "Ukrainian": ["modifying", "predicated", "verb"],  #
    "Urdu": ["modifying", "predicated", "verb"],  #
    "Yoruba": [],
}
