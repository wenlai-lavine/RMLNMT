# _*_ coding: utf-8 _*_

import torch
from torchtext import data
from torchtext.vocab import Vectors, GloVe


def load_DA_data(args, path):
    """
    tokenizer : Breaks sentences into a list of words. If sequential=False, no tokenization is applied
    Field : A class that stores information about the way of preprocessing
    fix_length : An important property of TorchText is that we can let the input to be variable length, and TorchText will
                 dynamically pad each sequence to the longest sequence in that "batch". But here we are using fi_length which
                 will pad each sequence to have a fix length of 200.

    build_vocab : It will first make a vocabulary or dictionary mapping all the unique words present in the train_data to an
                  idx and then after it will use GloVe word embedding to map the index to the corresponding word embedding.

    vocab.vectors : This returns a torch tensor of shape (vocab_size x embedding_dim) containing the pre-trained word embeddings.
    BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.

    """

    tokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True,
                      fix_length=200)
    LABEL = data.LabelField(tensor_type=torch.FloatTensor)

    train_data, dev_data = data.TabularDataset.splits(
        path=args.input_path, train='train.tsv', validation='dev.tsv', format='tsv', skip_header=True,
        fields=[('labels', LABEL), ('sentences', TEXT)]
    )

    test_data = data.TabularDataset(args.input_path + '/test.tsv', format='tsv', skip_header=True,
                               fields=[('labels', LABEL), ('sentences', TEXT)])

    TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=args.emd_dim))
    LABEL.build_vocab(train_data)

    word_embeddings = TEXT.vocab.vectors
    print("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print("Label Length: " + str(len(LABEL.vocab)))

    train_iter = data.BucketIterator(train_data, batch_size=args.batch_size, sort_key=lambda x: len(x.text),
                                     repeat=False, shuffle=True)

    dev_iter = data.BucketIterator(dev_data, batch_size=args.batch_size, sort_key=lambda x: len(x.text),
                                     repeat=False, shuffle=True)

    test_iter = data.BucketIterator(test_data, batch_size=args.batch_size, sort_key=lambda x: len(x.text),
                                     repeat=False)

    vocab_size = len(TEXT.vocab)

    return TEXT, vocab_size, word_embeddings, train_iter, dev_iter, test_iter