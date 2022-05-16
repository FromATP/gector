import argparse

import torch
import os
from pathlib import Path

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN

from gector.bert_token_embedder import PretrainedBertEmbedder
from gector.tokenizer_indexer import PretrainedBertIndexer
from gector.seq2labels_model import Seq2Labels
from utils.helpers import get_weights_name
from seq2seq.Seq2Seq_model import Seq2Seq
from seq2seq.gec_trainer import Trainer
from seq2seq.reader import Seq2SeqDataReader
from seq2seq.utils import get_smaller_vocab
 
def get_embbeder(weigths_name, special_tokens_fix, take_grad=False):
    embedders = {'bert': PretrainedBertEmbedder(
        pretrained_model=weigths_name,
        requires_grad=take_grad,
        top_layer_only=True,
        special_tokens_fix=special_tokens_fix)
    }
    text_field_embedder = BasicTextFieldEmbedder(
        token_embedders=embedders,
        embedder_to_indexer_map={"bert": ["bert", "bert-offsets"]},
        allow_unmatched_keys=True)
    return text_field_embedder


def get_token_indexers(model_name, max_pieces_per_token=5, special_tokens_fix=0):
    bert_token_indexer = PretrainedBertIndexer(
        pretrained_model=model_name,
        max_pieces_per_token=max_pieces_per_token,
        special_tokens_fix=special_tokens_fix
    )
    return {'bert': bert_token_indexer}


def get_data_reader(model_name, max_len, test_mode=False,
                    max_pieces_per_token=3, special_tokens_fix=0,):
    token_indexers = get_token_indexers(model_name,
                                        max_pieces_per_token=max_pieces_per_token,
                                        special_tokens_fix=special_tokens_fix,
                                        )
    reader = Seq2SeqDataReader(token_indexers=token_indexers,
                                max_len=max_len,
                                test_mode=test_mode,
                                lazy=True)
    return reader


def get_gec_model(vocab, ged_model, vocab_list,
                    max_seq_len = 300,
                    label_smoothing=0.0):
    model = Seq2Seq(ged_model=ged_model,
                    vocab=vocab,
                    vocab_list=vocab_list,
                    label_smoothing=label_smoothing,
                    max_seq_len=max_seq_len)
    return model


def main(args):
    ged_vocab = Vocabulary.from_files(args.ged_vocab_path)
    weights_name = get_weights_name(args.ged_model_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    ged_model = Seq2Labels(vocab=ged_vocab,
                        text_field_embedder=get_embbeder(weights_name, special_tokens_fix=1),
                        confidence=args.additional_confidence,
                        del_confidence=args.additional_del_confidence,
                    ).to(device)
    if torch.cuda.is_available():
        ged_model.load_state_dict(torch.load(args.ged_model_path[0]), strict=False)
    else:
        ged_model.load_state_dict(torch.load(args.ged_model_path[0], map_location=torch.device('cpu')), strict=False)
    ged_model.eval()
    print("GED model is set.")

    default_tokens = [DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN]
    namespaces = ['tokens', 'labels']
    tokens_to_add = {x: default_tokens for x in namespaces}
    reader = get_data_reader(weights_name, args.max_len,
                             test_mode=False,
                             max_pieces_per_token=args.pieces_per_token,
                             special_tokens_fix=1)

    train_dataset = reader.read(args.train_set)
    dev_dataset = reader.read(args.dev_set)
    gec_vocab = Vocabulary.from_instances(train_dataset,
                                        max_vocab_size={'tokens': 30000,
                                                        'labels': args.target_vocab_size},
                                        tokens_to_add=tokens_to_add)

    iterator = BucketIterator(batch_size=args.batch_size,
                              sorting_keys=[("tokens", "num_tokens")],
                              biggest_batch_first=True)
    iterator.index_with(gec_vocab)
    val_iterator = BucketIterator(batch_size=args.batch_size,
                                  sorting_keys=[("tokens", "num_tokens")], 
                                  instances_per_epoch=None)
    val_iterator.index_with(gec_vocab)

    vocab_list = get_smaller_vocab(iterator, train_dataset)
    vocab_path = Path(args.gec_model_dir) / 'vocabulary'
    vocab_path.mkdir(exist_ok=True, parents=True)
    with open(vocab_path / 'vocab.txt', "w", encoding="utf-8") as outputfd:
        outputfd.write(str(vocab_list))
    print(f'vocab size: {len(vocab_list)}')

    gec_model = get_gec_model(gec_vocab, ged_model, vocab_list,
                      max_seq_len = args.max_len,
                      label_smoothing=args.label_smoothing)
    gec_model.to(device)

    print("GEC model is set.")

    optimizer = torch.optim.Adam(gec_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            cuda_device = list(range(torch.cuda.device_count()))
        else:
            cuda_device = 0
    else:
        cuda_device = -1

    trainer = Trainer(model=gec_model,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      iterator=iterator,
                      validation_iterator=val_iterator,
                      train_dataset=train_dataset,
                      validation_dataset=dev_dataset,
                      serialization_dir=args.gec_model_dir,
                      patience=args.patience,
                      num_epochs=args.n_epoch,
                      cuda_device=cuda_device,
                      shuffle=False,
                      accumulated_batch_count=args.accumulation_size,
                      cold_step_count=args.cold_steps_count,
                      cold_lr=args.cold_lr)
    print("Start GEC training")
    trainer.train()


if __name__ == '__main__':
    # read parameters
    parser = argparse.ArgumentParser()
    # args for ged model
    parser.add_argument('--ged_model_path',
                        help='Path to the model file.', nargs='+',
                        required=True)
    parser.add_argument('--ged_vocab_path',
                        help='Path to the model file.',
                        default='./ged_model/vocabulary'  # to use pretrained models
                        )
    parser.add_argument('--additional_confidence',
                        type=float,
                        help='How many probability to add to $KEEP token.',
                        default=0)
    parser.add_argument('--additional_del_confidence',
                        type=float,
                        help='How many probability to add to $DELETE token.',
                        default=0)
    parser.add_argument('--ged_model_name',
                        choices=['bert', 'gpt2', 'transformerxl', 'xlnet', 'distilbert', 'roberta', 'albert'
                                 'bert-large', 'roberta-large', 'xlnet-large', 'bert-chn'],
                        help='Name of the transformer model.',
                        default='bert-chn')
                        
    # args for gec dataset
    parser.add_argument('--train_set',
                        help='Path to the train data', required=True)
    parser.add_argument('--dev_set',
                        help='Path to the dev data', required=True)
    parser.add_argument('--max_len',
                        type=int,
                        help='The max sentence length'
                             '(all longer will be truncated)',
                        default=300)
    parser.add_argument('--target_vocab_size',
                        type=int,
                        help='The size of target vocabularies.',
                        default=30000)  # 1000
    parser.add_argument('--pieces_per_token',
                        type=int,
                        help='The max number for pieces per token.',
                        default=5)

    # args for gec model
    parser.add_argument('--gec_model_dir',
                        help='Path to the model dir', required=True)
    parser.add_argument('--lr',
                        type=float,
                        help='Set initial learning rate.',
                        default=1e-5)
    parser.add_argument('--batch_size',
                        type=int,
                        help='The size of hidden unit cell.',
                        default=32)
    parser.add_argument('--patience',
                        type=int,
                        help='The number of epoch with any improvements'
                             ' on validation set.',
                        default=3)
    parser.add_argument('--label_smoothing',
                        type=float,
                        help='The value of parameter alpha for label smoothing.',
                        default=0.0)
    parser.add_argument('--n_epoch',
                        type=int,
                        help='The number of epoch for training model.',
                        default=20)
    parser.add_argument('--accumulation_size',
                        type=int,
                        help='How many batches do you want accumulate.',
                        default=4)
    parser.add_argument('--cold_steps_count',
                        type=int,
                        help='Whether to train only classifier layers first.',
                        default=4)
    parser.add_argument('--cold_lr',
                        type=float,
                        help='Learning rate during cold_steps.',
                        default=1e-3)

    args = parser.parse_args()
    main(args)
