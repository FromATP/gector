import logging
import re

from random import random
from typing import Dict, List

from allennlp.data.fields import TextField, SequenceLabelField, MetadataField, Field
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token

from overrides import overrides
from utils.helpers import SEQ_DELIMETERS, START_TOKEN, STOP_TOKEN

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("seq2seq_datareader")
class Seq2SeqDataReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:

    WORD###TAG [TAB] WORD###TAG [TAB] ..... \n

    and converts it into a ``Dataset`` suitable for sequence tagging. You can also specify
    alternative delimiters in the constructor.

    Parameters
    ----------
    delimiters: ``dict``
        The dcitionary with all delimeters.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenised in the data file.
    max_len: if set than will truncate long sentences
    """
    # fix broken sentences mostly in Lang8
    BROKEN_SENTENCES_REGEXP = re.compile(r'\.[a-zA-RT-Z]')

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 delimeters: dict = SEQ_DELIMETERS,
                 lazy: bool = False,
                 max_len: int = None,
                 test_mode: bool = False,
                 broken_dot_strategy: str = "keep") -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._delimeters = delimeters
        self._max_len = max_len
        self._broken_dot_strategy = broken_dot_strategy
        self._test_mode = test_mode

    @overrides
    def _read(self, file_path):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                # skip blank and broken lines
                if not line or (not self._test_mode and self._broken_dot_strategy == 'skip'
                                and self.BROKEN_SENTENCES_REGEXP.search(line) is not None):
                    continue

                src, tgt = line.split(self._delimeters["sents"])
                src = src.split(self._delimeters["tokens"])
                tgt = tgt.split(self._delimeters["tokens"])
                
                try:
                    src_tokens = [Token(token) for token in src]
                    tgt_tokens = [Token(token) for token in tgt]
                except ValueError:
                    src_tokens = [Token(token) for token in src]
                    tgt_tokens = None

                if src_tokens and src_tokens[0] != Token(START_TOKEN):
                    src_tokens = [Token(START_TOKEN)] + src_tokens
                if tgt_tokens and tgt_tokens[0] != Token(START_TOKEN):
                    tgt_tokens = [Token(START_TOKEN)] + tgt_tokens
                if src_tokens and src_tokens[-1] != Token(STOP_TOKEN):
                    src_tokens = [Token(STOP_TOKEN)] + src_tokens
                if tgt_tokens and tgt_tokens[-1] != Token(STOP_TOKEN):
                    tgt_tokens = [Token(STOP_TOKEN)] + tgt_tokens

                src_words = [x.text for x in src_tokens]
                tgt_words = [x.text for x in tgt_tokens]
                if self._max_len is not None:
                    src_tokens = src_tokens[:self._max_len]
                    tgt_tokens = None if tgt_tokens is None else tgt_tokens[:self._max_len]
                instance = self.text_to_instance(src_tokens, tgt_tokens, src_words, tgt_words)
                if instance:
                    yield instance


    def text_to_instance(self, src_tokens: List[Token], tgt_tokens: List[Token],
                         src_words: List[str] = None, tgt_words: List[str] = None) -> Instance:  # type: ignore
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}

        src_sequence = TextField(src_tokens, self._token_indexers)
        fields["tokens"] = src_sequence
        fields["src_metadata"] = MetadataField({"words": src_words})

        tgt_sequence = TextField(tgt_tokens, self._token_indexers)
        fields["labels"] = tgt_sequence
        fields["tgt_metadata"] = MetadataField({"words": tgt_words})

        return Instance(fields)
