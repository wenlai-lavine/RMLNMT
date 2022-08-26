import os
import time
import argparse
import load_data
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from models.LSTM import LSTMClassifier
from models.CNN import CNN
from models.LSTM_Attn import AttentionModel
from models.RCNN import RCNN
from models.RNN import RNN
from models.selfAttention import SelfAttention


def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

def train_model(args, model, train_iter, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.cuda()
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        text = batch.sentences[0]
        target = batch.labels
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()
        if (text.size()[0] is not args.batch_size):# One of the batch returned by BucketIterator has length different than 32.
            continue
        optim.zero_grad()
        prediction = model(text)
        loss = loss_fn(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/len(batch)
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1

        if steps % 1000 == 0:
            print (f'Epoch: {epoch+1}, Idx: {idx+1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')

        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()

    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)

def eval_model(args, model, val_iter):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text = batch.sentences[0]
            if (text.size()[0] is not args.batch_size):
                continue
            target = batch.labels
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()
            prediction = model(text)
            loss = loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter)


def args_parser():
    parser = argparse.ArgumentParser(description="sentence classification for Domain Adaptation in NMT")
    ## general param
    parser.add_argument("--input_path", required=True, type=str,
                        help="corpus path for your domain data")
    parser.add_argument("--output_path", required=True, type=str,
                        help="corpus path for your domain data")
    parser.add_argument("--model", required=True, type=str,
                        help="choose the model in your training")
    parser.add_argument("--batch_size", required=True, type=int,
                        help="batch size for model training")
    parser.add_argument("--lr", required=True, type=float,
                        help="learning rate")
    parser.add_argument("--label_nums", required=True, type=int,
                        help="num of labels using in your classifier")
    parser.add_argument("--hidden_size", type=int, default=256,
                        help="hidden size")
    parser.add_argument("--emd_dim", required=True, type=int,
                        help="dim of your embeddings")
    parser.add_argument("--epoch", required=True, type=int,
                        help="nums of epoch in the training step")
    ### CNN model param
    parser.add_argument("--in_kernel", type=int, default=1,
                        help="nums of input kernels")
    parser.add_argument("--out_kernel", type=int, default=100,
                        help="nums of output kernels")
    parser.add_argument("--kernel_size", type=list, default=[3,4,5],
                        help="height of kernels")
    parser.add_argument("--stride", type=int, default=1,
                        help="stride in the kernel")
    parser.add_argument("--padding", type=int, default=0,
                        help="height of kernels")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="height of kernels")

    return parser.parse_args()


if __name__ == "__main__":
    args = args_parser()

    TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_data.load_DA_data(args, args.input_path)

    if args.model == 'lstm':
        model = LSTMClassifier(
            args.batch_size,
            args.label_nums,
            args.hidden_size,
            vocab_size,
            args.emd_dim,
            word_embeddings
        )
    elif args.model == 'lstm_attn':
        model = AttentionModel(
            args.batch_size,
            args.label_nums,
            args.hidden_size,
            vocab_size,
            args.emd_dim,
            word_embeddings
        )
    elif args.model == 'cnn':
        model = CNN(
            args.batch_size,
            args.label_nums,
            args.in_kernel,
            args.out_kernel,
            args.kernel_size,
            args.stride,
            args.padding,
            args.dropout,
            vocab_size,
            args.emd_dim,
            word_embeddings
        )
    elif args.model == 'rcnn':
        model = RCNN(
            args.batch_size,
            args.label_nums,
            args.hidden_size,
            vocab_size,
            args.emd_dim,
            word_embeddings
        )
    elif args.model == 'rnn':
        model = RNN(
            args.batch_size,
            args.label_nums,
            args.hidden_size,
            vocab_size,
            args.emd_dim,
            word_embeddings
        )
    elif args.model == 'self_attn':
        model = SelfAttention(
            args.batch_size,
            args.label_nums,
            args.hidden_size,
            vocab_size,
            args.emd_dim,
            word_embeddings
        )
    else:
        raise IOError("please check the model name .... ...")

    loss_fn = F.cross_entropy

    for epoch in range(args.epoch):
        train_loss, train_acc = train_model(args, model, train_iter, epoch)
        val_loss, val_acc = eval_model(args, model, valid_iter)

        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')

        test_loss, test_acc = eval_model(args, model, test_iter)
        print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')

        with open(os.path.join(args.output_path, 'results_acc.txt'), 'a', encoding='utf-8') as res_acc_file:
            res_acc_file.write(f'Epoch: {epoch+1:02}, Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%' + '\n')

    ### save the model
    torch.save(model, os.path.join(args.output_path, "model_{}.pt".format(args.model)))
    print('endding ... ...')