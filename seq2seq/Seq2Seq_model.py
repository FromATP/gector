"""Basic model. Predicts tags for every token"""
from typing import Dict, Optional, List, Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TimeDistributed, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides

from utils.helpers import START_TOKEN, STOP_TOKEN, PAD
from seq2seq.utils import get_src_mask, get_tgt_mask, remove_redudant
from seq2seq.modules import AttentionalEncoder, SelfAttentionLayer, AttentionalDecoder, LinearLayer

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
                 local_vocab: List,
                 labels_namespace: str = "labels",
                 verbose_metrics: bool = False,
                 label_smoothing: float = 0.0,
                 max_seq_len: int = 300,
                 hidden_size: int=512,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(Seq2Seq, self).__init__(vocab, regularizer)

        self.ged_model = ged_model

        self.label_namespaces = [labels_namespace]
        self.num_labels_classes = len(local_vocab)
        self.label_smoothing = label_smoothing
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.local_vocab = local_vocab

        self.id_to_vocab = dict(zip(local_vocab.values(), local_vocab.keys()))

        self.start_id = self.local_vocab['start']
        self.stop_id = self.local_vocab['stop']

        self._verbose_metrics = verbose_metrics
        self.ged_encoder = AttentionalEncoder(self.ged_model.num_labels_classes,
                                                self.hidden_size)
        self.gec_encoder = AttentionalEncoder(self.num_labels_classes,
                                                self.hidden_size)
        self.self_attn = SelfAttentionLayer(self.num_labels_classes,
                                                self.hidden_size)
        self.ged_decoder = AttentionalDecoder(self.hidden_size)
        self.gec_decoder = AttentionalDecoder(self.hidden_size)
        self.param_learning_layer = TimeDistributed(torch.nn.Linear(2*self.hidden_size, hidden_size))
        self.generator = LinearLayer(self.hidden_size, self.num_labels_classes)

        self.metrics = {"accuracy": CategoricalAccuracy()}

        initializer(self)


    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                labels: Dict[str, torch.LongTensor] = None,
                src_metadata: List[Dict[str, Any]] = None,
                tgt_metadata: List[Dict[str, Any]] = None,
                src_local:torch.LongTensor = None,
                tgt_local:torch.LongTensor = None) -> Dict[str, torch.Tensor]:
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
        # ESSENTIAL: pytorch transformer的padding mask是“true表示ignore”，与get_text_field_mask的意义相反
        src_padding_mask = 1 - get_text_field_mask(tokens)
        src_padding_mask = src_padding_mask.to(dtype=torch.bool)
        src_local_padding = src_local == 0
        src_mask = get_src_mask(src_local)

        # ESSENTIAL：ged模型的indexer将$START拆成了两个token：$和START。在embedder之后只保留了其中一个的结果
        # 这导致ged output和tokens['bert']的seq_len不一 样，后者多2（两个$）
        # 按照逻辑，在做gec任务的时候只需要保留start和stop，把$去掉。
        with torch.no_grad():
            ged_output = self.ged_model(tokens)["class_probabilities_labels"]
            ged_output = torch.argmax(ged_output, dim=2)
        encoded_ged_res = self.ged_encoder(ged_output, src_mask, src_padding_mask)
        encoded_text = self.gec_encoder(src_local, src_mask, src_local_padding)

        if labels is not None:
            tgt_padding_mask = 1 - get_text_field_mask(labels)
            tgt_padding_mask = tgt_padding_mask.to(dtype=torch.bool)
            tgt_local_padding = tgt_local == 0
            tgt_mask = get_tgt_mask(tgt_local)

            batch_size, sequence_length, _ = encoded_text.size()
            tgt_input = tgt_local[:, :-1]
            tgt_input_mask = tgt_mask[:-1, :-1]
            tgt_input_padding_mask = tgt_local_padding[:, :-1]
            tgt_output = tgt_local[:, 1:].contiguous()
            tgt_output_padding_mask = tgt_local_padding[:, 1:]

            tgt_attn = self.self_attn(tgt_input, tgt_input_mask, tgt_input_padding_mask)
            decoded_ged_res = self.ged_decoder(tgt_attn, encoded_ged_res, tgt_input_mask, tgt_input_padding_mask)
            decoded_text = self.gec_decoder(tgt_attn, encoded_text, tgt_input_mask, tgt_input_padding_mask)

            concat_res = torch.cat([decoded_text, decoded_ged_res], dim=2)
            alpha = F.sigmoid(self.param_learning_layer(concat_res))
            # print(alpha)
            final_res = alpha * decoded_text + (1 - alpha) * decoded_ged_res
            output_logits = self.generator(final_res)

            output_prob = F.softmax(output_logits, dim=-1)

            output_dict = {"logits": output_logits,
                        "probabilities": output_prob}
            loss = sequence_cross_entropy_with_logits(output_logits, tgt_output, ~tgt_output_padding_mask,
                                                        label_smoothing=self.label_smoothing)
            
            for metric in self.metrics.values():
                metric(output_logits, tgt_output, tgt_output_padding_mask.float())
            output_dict["loss"] = loss
        
        if src_metadata is not None:
            output_dict["words"] = [x["words"] for x in src_metadata]
        if tgt_metadata is not None:
            output_dict["words"] = [x["words"] for x in tgt_metadata]
        return output_dict 

    @overrides
    def decode(self, tokens: Dict[str, torch.Tensor], src_local: torch.LongTensor):
       
        src_padding_mask = 1 - get_text_field_mask(tokens)
        src_padding_mask = src_padding_mask.to(dtype=torch.bool)
        src_local_padding = src_local == 0
        src_mask = get_src_mask(src_local)

        with torch.no_grad():
            ged_output = self.ged_model(tokens)["class_probabilities_labels"]
            ged_output = torch.argmax(ged_output, dim=2)

            encoded_ged_res = self.ged_encoder(ged_output, src_mask, src_padding_mask)
            encoded_text = self.gec_encoder(src_local, src_mask, src_local_padding)
            
            batch_size, sequence_length, _ = encoded_text.size()

            ids = []
            words = []
            for k in tqdm(range(batch_size)):
                cur_ged_res = encoded_ged_res[k][None, :, :]
                cur_text = encoded_text[k][None, :, :]
                cur_tgt = torch.ones(1, 1).fill_(self.start_id).type(torch.long).to(encoded_text.device)
                for i in range(self.max_seq_len):
                    cur_tgt_mask = (get_tgt_mask(cur_tgt).type(torch.bool)).to(cur_tgt.device)
                    tgt_attn = self.self_attn(cur_tgt, cur_tgt_mask)
                    decoded_ged_res = self.ged_decoder(tgt_attn, cur_ged_res, cur_tgt_mask)
                    decoded_text = self.gec_decoder(tgt_attn, cur_text, cur_tgt_mask)
                    
                    concat_res = torch.cat([decoded_text, decoded_ged_res], dim=2)
                    alpha = F.sigmoid(self.param_learning_layer(concat_res))
                    final_res = alpha * decoded_text + (1 - alpha) * decoded_ged_res
                    cur_logits = self.generator(final_res[:, -1:, :])
                    next_word = torch.argmax(cur_logits, dim=2)[0][0]
                    cur_tgt = torch.cat([cur_tgt, torch.ones(1, 1).type_as(src_local.data).fill_(next_word)], dim=1)
                    if next_word == self.stop_id:
                        break
                    
                cur_tgt = cur_tgt.cpu().tolist()[0]
                decoded_sent = [self.id_to_vocab[i] for i in cur_tgt]
                ids.append(cur_tgt)
                words.append(decoded_sent)

        return words, ids

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {metric_name: metric.get_metric(reset) for
                             metric_name, metric in self.metrics.items()}
        return metrics_to_return
