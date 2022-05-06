from errno import EMEDIUMTYPE
from re import A
from typing import Dict, Optional, List, Any

import numpy
import torch

import torch.nn.utils.rnn as R
import torch.nn.functional as F
from torch.nn.modules.linear import Linear

from torchcrf import CRF

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules import TimeDistributed, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask

from overrides import overrides
from bert_crf.encoders import LinearEncoder, BiLSTMEncoder, AttentionEncoder

@Model.register("seq2seq")
class Seq2Seq(Model):
    """

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and predicting output tags.
    calculate_span_f1 : ``bool``, optional (default=``None``)
        Calculate span-level F1 metrics during training. If this is ``True``, then
        ``label_encoding`` is required. If ``None`` and
        label_encoding is specified, this is set to ``True``.
        If ``None`` and label_encoding is not specified, it defaults
        to ``False``.
    label_encoding : ``str``, optional (default=``None``)
        Label encoding to use when calculating span f1.
        Valid options are "BIO", "BIOUL", "IOB1", "BMES".
        Required if ``calculate_span_f1`` is true.
    labels_namespace : ``str``, optional (default=``labels``)
        This is needed to compute the SpanBasedF1Measure metric, if desired.
        Unless you did something unusual, the default value should be what you want.
    verbose_metrics : ``bool``, optional (default = False)
        If true, metrics will be returned per label class in addition
        to the overall statistics.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, ged_model: Model,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 predictor_dropout=0.0,
                 labels_namespace: str = "labels",
                 verbose_metrics: bool = False,
                 label_smoothing: float = 0.0,
                 confidence: float = 0.0,
                 del_confidence: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 encoder_type = "Linear",
                 lstm_dropout_rate = 0.5,
                 hidden_dim = 256) -> None:
        super(Seq2Seq, self).__init__(vocab, regularizer)

        self.ged_model = ged_model

        self.label_namespaces = [labels_namespace]
        self.text_field_embedder = text_field_embedder
        self.num_labels_classes = self.vocab.get_vocab_size(labels_namespace)
        self.label_smoothing = label_smoothing
        self.confidence = confidence
        self.del_conf = del_confidence

        self._verbose_metrics = verbose_metrics
        self.predictor_dropout = TimeDistributed(torch.nn.Dropout(predictor_dropout))
        output_dim = text_field_embedder._token_embedders['bert'].get_output_dim()
        self.ged_embedder = AttentionEncoder(input_dim=self.ged_model.num_labels_classes, output_dim=output_dim)
        self.param_learning_layer = LinearEncoder(label_size=1, input_dim=2*output_dim)

        if encoder_type == "Linear":
            self.encoder = LinearEncoder(label_size=self.num_labels_classes, input_dim=output_dim)
        else:
            self.encoder = BiLSTMEncoder(label_size=self.num_labels_classes, input_dim=output_dim,
                                        hidden_dim=hidden_dim, drop_lstm=lstm_dropout_rate)
            
        self.crf_layer = CRF(self.num_labels_classes, batch_first=True)
        

        self.metrics = {"accuracy": CategoricalAccuracy()}

        self.alpha_labels = [0.7 for i in range(self.num_labels_classes)]
        self.alpha_labels[0] = 0.2
        self.alpha_labels[-1] = self.alpha_labels[-2] = 0.05
        
        self.gamma = 2

        initializer(self)
        

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                labels: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        labels : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            metadata containing the original words in the sentence to be tagged under a 'words' key.

        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.

        """
        ged_output = self.ged_model(tokens)["class_probabilities_labels"]
        embedded_ged_res = self.ged_embedder(ged_output)
        # print(embedded_ged_res.size())

        embedded_text = self.text_field_embedder(tokens)
        mask = get_text_field_mask(tokens)
        # print(embedded_text.size())
        concat_embedding = torch.cat([embedded_text, embedded_ged_res], dim=2)
        alpha = F.sigmoid(self.param_learning_layer(concat_embedding))
        # print(alpha)
        final_embedding = alpha * embedded_text + (1 - alpha) * embedded_ged_res
        # print(final_embedding.size())

        output_dict = {}
        
        if labels is not None:
            word_lens = torch.sum(mask, dim=1)
            final_embedding = self.encoder(final_embedding, word_lens)
            final_embedding = self.predictor_dropout(final_embedding)
            loss = self.crf_layer(final_embedding, labels, mask=mask.type(torch.bool))
            output_dict["loss"] = loss
        
        if metadata is not None:
            output_dict["words"] = [x["words"] for x in metadata]
        return output_dict

    @overrides
    def decode(self, tokens: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple position-wise argmax over each token, converts indices to string labels, and
        adds a ``"tags"`` key to the dictionary with the result.
        """

        ged_output = self.ged_model(tokens)["class_probabilities_labels"]
        embedded_ged_res = self.ged_embedder(ged_output)

        embedded_text = self.text_field_embedder(tokens)
        concat_embedding = torch.cat([embedded_text, embedded_ged_res], dim=2)
        alpha = F.sigmoid(self.param_learning_layer(concat_embedding))
        final_embedding = alpha * embedded_text + (1 - alpha) * embedded_ged_res
        batch_size, sequence_length, _ = final_embedding.size()

        word_indices = self.crf_layer.decode(final_embedding)
        
        for idx in range(batch_size):
            word_indices[idx][:sequence_length[idx]] = word_indices[idx][:sequence_length[idx]].flip([0])
        
        return word_indices

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {metric_name: metric.get_metric(reset) for
                             metric_name, metric in self.metrics.items()}
        return metrics_to_return
