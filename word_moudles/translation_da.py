# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import os

import ipdb
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.data import (
    ConcatDataset,
    data_utils,
    Dictionary,
    IndexedCachedDataset,
    IndexedDataset,
    IndexedRawTextDataset,
    LanguagePairDataset,
    FairseqDataset,
    IndexedRawLinesDataset
)

from fairseq.tasks import FairseqTask, register_task
from fairseq.tasks.translation import TranslationTask

### NEW CRIETERION ###################################################

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.cross_entropy import CrossEntropyCriterion
@register_criterion('cross_entropy_da')
class CrossEntropyCriterionDA(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.da_reweightloss = args.da_reweightloss

        if args.domain_nums==2:
            self.clsfunc = "binary" #F.binary_cross_entropy_with_logits
        else:
            self.clsfunc = "multidomain" #F.cross_entropy

        if args.damethod == "naive":
            self.forward = self.forward_naive
        elif args.damethod in ["multitask_classifier", "pure_adv", "partial_adv", "bayesian"]:
            self.forward = self.forward_multitask_classifier
        elif args.damethod == "multitask_tag":
            raise NotImplementedError("multitask_tag is not implemented")
        else:
            raise NotImplementedError(args.damethod+" is not implemented")

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward_naive(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True, mask_att=None):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1)) # lprobs: (LB)xD
        target = model.get_targets(sample, net_output).view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        eps_i = self.eps / lprobs.size(-1)
        if mask_att is None:
            if reduce:
                nll_loss = nll_loss.sum()
                smooth_loss = smooth_loss.sum()
            loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        else:
            mask_att = mask_att.detach().view(-1,1)[non_pad_mask]
            loss_wei = mask_att+1
            wei_nll_loss = nll_loss*loss_wei
            wei_smooth_loss = smooth_loss*loss_wei
            if reduce:
                nll_loss = nll_loss.sum()
                wei_nll_loss = wei_nll_loss.sum()
                wei_smooth_loss = smooth_loss.sum()
            loss = (1. - self.eps) * wei_nll_loss + eps_i * wei_smooth_loss
        return loss, nll_loss

    def forward_multitask_classifier(self, model, sample, reduce=True, getproportion=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if getproportion:
            if self.da_reweightloss:
                mask_att, idx_output, net_output = model(**sample['net_input'],classify = True)
            else:
                idx_output, net_output = model(**sample['net_input'],classify = True)
            idx_output = idx_output.squeeze()
            if self.clsfunc == "binary":
                if len(idx_output.shape) == 1:
                    idx_output.unsqueeze(0)
                x = F.sigmoid(idx_output).unsqueeze(-1)
                domain_prop = torch.cat([x,1-x],dim=-1)
            else:
                if len(idx_output.shape) == 2:
                    idx_output.unsqueeze(0)
                domain_prop = F.softmax(idx_output,dim=-1)
            return domain_prop.to("cpu").numpy()

        if self.da_reweightloss:
            mask_att, idx_output, net_output = model(**sample['net_input'],classify = True)
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce, mask_att=mask_att)
        else:
            idx_output, net_output = model(**sample['net_input'], classify=True)
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        idx_output = idx_output.squeeze()
        cls_parts = 1
        if self.clsfunc == "binary":
            dataset_idx = sample['dataset_idx'].float()
            clsfunc = F.binary_cross_entropy_with_logits
            if(len(idx_output.shape)==2):
                cls_parts = idx_output.shape[0]
                dataset_idx = dataset_idx.repeat(cls_parts,1).view(-1)
                idx_output = idx_output.view(-1)
            cls_accu = (dataset_idx==(idx_output.squeeze()>=0.5).float()).float().sum().item()/cls_parts
        else:
            dataset_idx = sample['dataset_idx'].long()
            clsfunc = F.cross_entropy
            if(len(idx_output.shape)==3):
                cls_parts = idx_output.shape[0]
                dataset_idx = dataset_idx.repeat(cls_parts,1).view(-1)
                idx_output = idx_output.view(cls_parts*idx_output.shape[1],-1)
            _,label_out = idx_output.max(dim=1)
            cls_accu = (dataset_idx==label_out).float().sum().item()/cls_parts
        if dataset_idx.shape[0] == idx_output.shape[0] and idx_output.shape[1] == self.args.domain_nums:
            cls_loss = clsfunc(idx_output, dataset_idx)
            loss = loss + cls_loss
        # else:
        #     loss = loss + torch.tensor(2.5).float().cuda()
        #     print('出错了... ...')

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'cls_accu': cls_accu,
            'batch_size': sample['dataset_idx'].shape[0],
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        try:
            cls_accu = sum(log.get('cls_accu', 0) for log in logging_outputs)
            batch_size = sum(log.get('batch_size', 0) for log in logging_outputs)
            agg_output['cls_accu'] = cls_accu / batch_size
        except:
            pass
        return agg_output


### NEW Dataset ######################################################
import bisect
class ConcatDataset_IDX(ConcatDataset):
    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        sample_idx = sample_idx % self.real_sizes[dataset_idx]
        return [dataset_idx, self.datasets[dataset_idx][sample_idx]]

    def get_original_text(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        sample_idx = sample_idx % self.real_sizes[dataset_idx]
        return self.datasets[dataset_idx].get_original_text(sample_idx)

def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    dataset_idx = torch.LongTensor([s['dataset_idx'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    dataset_idx = dataset_idx.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'dataset_idx': dataset_idx,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


class LanguagePairDataset_IDX(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for input feeding/teacher forcing
            (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
    """

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True, remove_eos_from_source=False, append_eos_to_target=False,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        tgt_item = tgt_item[1]
        src_item = self.src[index]
        dataset_idx = src_item[0]
        src_item = src_item[1]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and tgt_item[-1] != eos:
                tgt_item = torch.cat([tgt_item, torch.LongTensor([eos])])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if src_item[-1] == eos:
                src_item = src_item[:-1]

        return {
            'id': index,
            'dataset_idx': dataset_idx,
            'source': src_item,
            'target': tgt_item,
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=128):
        """Return a dummy batch with a given number of tokens."""
        src_len, tgt_len = utils.resolve_max_positions(
            (src_len, tgt_len),
            max_positions,
            (self.max_source_positions, self.max_target_positions),
        )
        bsz = max(num_tokens // max(src_len, tgt_len), 1)
        return self.collater([
            {
                'id': i,
                'dataset_idx': i%2,
                'source': self.src_dict.dummy_sentence(src_len),
                'target': self.tgt_dict.dummy_sentence(tgt_len) if self.tgt_dict is not None else None,
            }
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and getattr(self.tgt, 'supports_prefetch', False)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        self.tgt.prefetch(indices)



### NEW TASKS ########################################################
@register_task('translation_da')
class TranslationTaskDA(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (Dictionary): dictionary for the source language
        tgt_dict (Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', nargs='+', help='path(s) to data directorie(s)')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        # fmt: on

    @staticmethod
    def load_pretrained_model(path, src_dict_path, tgt_dict_path, arg_overrides=None):
        model = utils.load_checkpoint_to_cpu(path)
        args = model['args']
        state_dict = model['model']
        args = utils.override_model_args(args, arg_overrides)
        src_dict = Dictionary.load(src_dict_path)
        tgt_dict = Dictionary.load(tgt_dict_path)
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()

        task = TranslationTask(args, src_dict, tgt_dict)
        model = task.build_model(args)
        model.upgrade_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=True)
        return model

    def __init__(self, args, src_dict, tgt_dict, user_data_frame, task_score):
        super().__init__(args)
        self.is_curriculum = args.is_curriculum
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.user_data_frame = user_data_frame
        # self.meta_dev = meta_dev
        self.task_score = task_score

    def plot_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            proportion = criterion(model, sample, getproportion=True)
        return proportion

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()

        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(args.data[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        user_data_frame = kwargs['user_data_frame']
        task_score = kwargs['task_score']
        # meta_dev = kwargs['meta_dev']

        # load dictionaries
        # src_dict = cls.load_dictionary(os.path.join(args.data[0], 'dict.{}.txt'.format(args.source_lang)))
        # tgt_dict = cls.load_dictionary(os.path.join(args.data[0], 'dict.{}.txt'.format(args.target_lang)))
        src_dict = kwargs['src_dict']
        tgt_dict = kwargs['tgt_dict']
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict, user_data_frame=user_data_frame, task_score=task_score)

    def load_word_dataset(self, split, combine=False, fine_tune=False, bucket=False, **kwargs):

        src_datasets = []
        tgt_datasets = []
        # split_data = self.user_data_frame[self.user_data_frame.task_group == split].reset_index()
        # query_df = self.user_data_frame.loc[self.user_data_frame['task_group'] == self.args.valid_subset]
        # if split == 'train':
        #     support_df = self.user_data_frame.loc[self.user_data_frame['task_group'] == self.args.train_subset]
        #     df_by_domain = support_df.groupby('domain_group')
        #     # df_by_domain = self.user_data_frame.groupby('domain_group')
        # else:
        #     # support_df = self.user_data_frame.loc[self.user_data_frame.task_group.str.startswith('support')]
        #     df_by_domain = self.user_data_frame.groupby('domain_group')
        df_by_domain = self.user_data_frame.groupby('domain_group')
        df_group_list = []
        for name, df_group in df_by_domain:
            df_group_list.append(df_group)

        for i, dgl in enumerate(df_group_list):
            source_lines = dgl.src
            target_lines = dgl.tgt
            src_datasets.append(IndexedRawLinesDataset(lines=source_lines, dictionary=self.src_dict))
            tgt_datasets.append(IndexedRawLinesDataset(lines=target_lines, dictionary=self.tgt_dict))
            print('| {} {} {} examples'.format('', split, len(source_lines)))

        # auto adjust domain nums
        self.args.domain_nums = 5

        assert len(src_datasets) == len(tgt_datasets)

        if len(src_datasets) == 1:
            src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
        else:
            sample_ratios = [1] * len(src_datasets)
            sample_ratios[0] = self.args.upsample_primary
            src_dataset = ConcatDataset_IDX(src_datasets, sample_ratios)
            tgt_dataset = ConcatDataset_IDX(tgt_datasets, sample_ratios)

        self.datasets[split] = LanguagePairDataset_IDX(
            src_dataset, src_dataset.sizes, self.src_dict,
            tgt_dataset, tgt_dataset.sizes, self.tgt_dict,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
        )

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def split_exists(split, src, tgt, lang, data_path):
            filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
            if self.args.raw_text and IndexedRawTextDataset.exists(filename):
                return True
            elif not self.args.raw_text and IndexedDataset.exists(filename):
                return True
            return False

        def indexed_dataset(path, dictionary):
            if self.args.raw_text:
                return IndexedRawTextDataset(path, dictionary)
            elif IndexedDataset.exists(path):
                if self.args.lazy_load:
                    return IndexedDataset(path, fix_lua_indexing=True)
                else:
                    return IndexedCachedDataset(path, fix_lua_indexing=True)
            return None

        src_datasets = []
        tgt_datasets = []

        data_paths = self.args.data

        for dk, data_path in enumerate(data_paths):
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')

                # infer langcode
                src, tgt = self.args.source_lang, self.args.target_lang
                if split_exists(split_k, src, tgt, src, data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
                elif split_exists(split_k, tgt, src, src, data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
                else:
                    if k > 0 or dk > 0:
                        break
                    else:
                        raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

                src_datasets.append(indexed_dataset(prefix + src, self.src_dict))
                tgt_datasets.append(indexed_dataset(prefix + tgt, self.tgt_dict))

                print('| {} {} {} examples'.format(data_path, split_k, len(src_datasets[-1])))

                if not combine:
                    break

        # auto adjust domain nums
        self.args.domain_nums = k

        assert len(src_datasets) == len(tgt_datasets)

        if len(src_datasets) == 1:
            src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
        else:
            sample_ratios = [1] * len(src_datasets)
            sample_ratios[0] = self.args.upsample_primary
            src_dataset = ConcatDataset_IDX(src_datasets, sample_ratios)
            tgt_dataset = ConcatDataset_IDX(tgt_datasets, sample_ratios)

        self.datasets[split] = LanguagePairDataset_IDX(
            src_dataset, src_dataset.sizes, self.src_dict,
            tgt_dataset, tgt_dataset.sizes, self.tgt_dict,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
        )

    # def load_word_dataset(self, split, num_task, combine=False, **kwargs):
    #     """Load a given dataset split.
    #
    #     Args:
    #         split (str): name of the split (e.g., train, valid, test)
    #     """
    #
    #     def split_exists(split, src, tgt, lang, data_path):
    #         filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
    #         if self.args.raw_text and IndexedRawTextDataset.exists(filename):
    #             return True
    #         elif not self.args.raw_text and IndexedDataset.exists(filename):
    #             return True
    #         return False
    #
    #     def indexed_dataset(path, dictionary):
    #         if self.args.raw_text:
    #             return IndexedRawTextDataset(path, dictionary)
    #         elif IndexedDataset.exists(path):
    #             if self.args.lazy_load:
    #                 return IndexedDataset(path, fix_lua_indexing=True)
    #             else:
    #                 return IndexedCachedDataset(path, fix_lua_indexing=True)
    #         return None
    #
    #     src_datasets = []
    #     tgt_datasets = []
    #
    #     data_paths = self.args.data + 'CL_' + str(num_task)
    #
    #     for dk, data_path in enumerate(data_paths):
    #         for k in itertools.count():
    #             split_k = split + (str(k) if k > 0 else '')
    #
    #             # infer langcode
    #             src, tgt = self.args.source_lang, self.args.target_lang
    #             if split_exists(split_k, src, tgt, src, data_path):
    #                 prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
    #             elif split_exists(split_k, tgt, src, src, data_path):
    #                 prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
    #             else:
    #                 if k > 0 or dk > 0:
    #                     break
    #                 else:
    #                     raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))
    #
    #             src_datasets.append(indexed_dataset(prefix + src, self.src_dict))
    #             tgt_datasets.append(indexed_dataset(prefix + tgt, self.tgt_dict))
    #
    #             print('| {} {} {} examples'.format(data_path, split_k, len(src_datasets[-1])))
    #
    #             if not combine:
    #                 break
    #
    #     # auto adjust domain nums
    #     self.args.domain_nums = k
    #
    #     assert len(src_datasets) == len(tgt_datasets)
    #
    #     if len(src_datasets) == 1:
    #         src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
    #     else:
    #         sample_ratios = [1] * len(src_datasets)
    #         sample_ratios[0] = self.args.upsample_primary
    #         src_dataset = ConcatDataset_IDX(src_datasets, sample_ratios)
    #         tgt_dataset = ConcatDataset_IDX(tgt_datasets, sample_ratios)
    #
    #     self.datasets[split] = LanguagePairDataset_IDX(
    #         src_dataset, src_dataset.sizes, self.src_dict,
    #         tgt_dataset, tgt_dataset.sizes, self.tgt_dict,
    #         left_pad_source=self.args.left_pad_source,
    #         left_pad_target=self.args.left_pad_target,
    #         max_source_positions=self.args.max_source_positions,
    #         max_target_positions=self.args.max_target_positions,
    #     )

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict
