"""Predict masked words using BERT.

The code in this module was heavily borrowed from Yoav Goldberg's code on
assessing BERT's syntactic abilities: https://github.com/yoavg/bert-syntax/
Thanks to Yoav for making his code available.

"""
import pandas as pd
from transformers import BertForMaskedLM, BertTokenizer
from torch import LongTensor  # pylint: disable=E0611

from constants import MASK

START = ['[CLS]']
END = ['[SEP]']


class BERT:
    """High-level interface for getting word predictions from BERT."""
    def __init__(self, name):
        """Initialize BERT instance.

        Parameters
        ----------
        name : str
            Name of the pre-trained BERT model to use. In this project, either
            'bert-base-multilingual-cased' or 'bert-base-cased'

        """
        self.model = BertForMaskedLM.from_pretrained(name)
        self.model.eval()
        self.model.to("cuda:0")
        tokenizer = BertTokenizer.from_pretrained(name)
        self.tokenize = tokenizer.tokenize
        self.tokens_to_ids = tokenizer.convert_tokens_to_ids
        self.ids_to_tokens = tokenizer.ids_to_tokens
        # tokenizer.vocab is a collections.OrderedDict, not a regular Python
        # dictionary, so its keys always come out in the same order.
        self.vocab = list(tokenizer.vocab.keys())
        self.index = pd.Index(self.vocab, name='word')

    def predict(self, masked_sentence, fold_case=False):
        """Predict the masked word in `masked_sentence`.

        Note that the output probability distribution is unnormalized.

        Parameters
        ----------
        masked_sentence : str
            Sentence with one token masked out
        fold_case : bool
            Whether or not to average predictions over different casings.

        Returns
        -------
        pd.DataFrame
            The unnormalized probability distribution over BERT's vocab of
            each word in the masked position.

        """
        tokens = START + self.tokenize(masked_sentence) + END
        target_index = tokens.index(MASK)
        token_ids = self.tokens_to_ids(tokens)
        tensor = LongTensor(token_ids).unsqueeze(0)
        probs = self.model(tensor)[0][0, target_index]
        probs = pd.DataFrame(probs.data.numpy(),
                             index=self.index,
                             columns=['p'])
        if fold_case:
            probs.index = probs.index.str.lower()
            return probs.groupby('word').mean()
        return probs
