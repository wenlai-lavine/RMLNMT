import ipdb
import pandas
import pandas as pd

from fairseq.data import LanguagePairDataset, IndexedRawLinesDataset
from fairseq.data import data_utils
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from word_moudles.translation_da import ConcatDataset_IDX, LanguagePairDataset_IDX


@register_task('word_adapt_balance')
class WordAdapt_balance(TranslationTask):

    @staticmethod
    def add_args(parser):
        int_inf = 1000000000000
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default=True, type=bool, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default=False, type=bool, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        # Limit train, test, dev for downstream task
        # parser.add_argument('--train-limit', type=int, default=int_inf, help='maximum size for downstream train split')
        # parser.add_argument('--valid-limit', type=int, default=int_inf, help='maximum size for downstream dev split')
        # parser.add_argument('--test-limit', type=int, default=int_inf, help='maximum size for downstream test split')
        parser.add_argument('--support-tokens', type=int, default=int_inf, help='maximum size for single task support split')
        parser.add_argument('--query-tokens', type=int, default=int_inf, help='maximum size for single task query split')
        parser.add_argument('--is-curriculum', action='store_true', help='inner curriclum learning option')

        # fmt: on

    def __init__(self, args, src_dict, tgt_dict, user_data_frame, task_score):
        super().__init__(args, src_dict, tgt_dict)
        self.is_curriculum = args.is_curriculum
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.user_data_frame = user_data_frame
        self.task_score = task_score
        # self.user_data_frame = user_data_frame[user_data_frame.split == args.train_subset].head(args.train_limit)
        # self.user_data_frame = self.user_data_frame.append(user_data_frame[user_data_frame.split == args.valid_subset].head(args.valid_limit))
        # self.user_data_frame = self.user_data_frame.append(user_data_frame[user_data_frame.split == args.test_subset].head(args.test_limit))

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(args.data[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        user_data_frame = kwargs['user_data_frame']
        task_score = kwargs['task_score']
        src_dict = kwargs['src_dict']
        tgt_dict = kwargs['tgt_dict']
        return cls(args=args, src_dict=src_dict, tgt_dict=tgt_dict, user_data_frame=user_data_frame, task_score=task_score)

    # def dataset_buckets(self, split):
        #  split_data = self.datasets[split].src

    def load_dataset(self, split, combine=False, fine_tune=False, bucket=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        if split not in self.datasets:
            if not fine_tune:
                src_datasets = []
                tgt_datasets = []

                if split == 'support_dev':
                    split_data = self.user_data_frame.loc[self.user_data_frame.task_group.str.startswith('support')].reset_index()
                elif split == 'query_dev':
                    split_data = self.user_data_frame.loc[self.user_data_frame.task_group.str.startswith('query')].reset_index()
                else:
                    split_data = self.user_data_frame[self.user_data_frame.task_group == split].reset_index()


                if self.args.domain_nums == 5 and self.args.translation_task == 'en2de':
                    en2de_seen = ['General', 'EMEA', 'GlobalVoices', 'JRC', 'KDE']
                elif self.args.domain_nums == 6 and self.args.translation_task == 'en2de':
                    en2de_seen = ['General', 'EMEA', 'GlobalVoices', 'JRC', 'KDE', 'WMT']

                src_datasets = []
                tgt_datasets = []

                df_by_domain = split_data.groupby('domain_group')
                exist_domain_list = []
                df_domain_list = []
                domain_dict = dict()

                for name, df_group in df_by_domain:
                    exist_domain_list.append(name)
                    df_domain_list.append(df_group)
                    domain_dict[name] = df_group

                for e2ds in en2de_seen:
                    if e2ds in exist_domain_list:
                        source_lines = domain_dict[e2ds].src
                        target_lines = domain_dict[e2ds].tgt
                        src_datasets.append(IndexedRawLinesDataset(lines=source_lines, dictionary=self.src_dict))
                        tgt_datasets.append(IndexedRawLinesDataset(lines=target_lines, dictionary=self.tgt_dict))
                    else:
                        src_datasets.append(IndexedRawLinesDataset(lines='', dictionary=self.src_dict))
                        tgt_datasets.append(IndexedRawLinesDataset(lines='', dictionary=self.tgt_dict))

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

            else:
                print('-----------------')
                print(self.args.domains)
                en2de_unseen = ['Covid', 'Bible', 'Books', 'ECB', 'TED']
                en2de_seen = ['EMEA', 'General', 'GlobalVoices', 'JRC', 'KDE', 'WMT']
                en2zh_unseen = ['Education', 'Microblog', 'Science', 'Subtitles']
                en2zh_seen = ['General', 'Laws', 'News', 'Spoken', 'Thesis']
                src_datasets = []
                tgt_datasets = []
                split_data = self.user_data_frame[self.user_data_frame.task_group.str.contains(split)]

                if self.args.domains[0] in en2zh_unseen or self.args.domains[0] in en2de_unseen:
                    for i in range(self.args.domain_nums):
                        if i == 0:
                            source_lines = split_data.src
                            target_lines = split_data.tgt
                            src_datasets.append(IndexedRawLinesDataset(lines=source_lines, dictionary=self.src_dict))
                            tgt_datasets.append(IndexedRawLinesDataset(lines=target_lines, dictionary=self.tgt_dict))
                        else:
                            src_datasets.append(IndexedRawLinesDataset(lines='', dictionary=self.src_dict))
                            tgt_datasets.append(IndexedRawLinesDataset(lines='', dictionary=self.tgt_dict))
                elif self.args.translation_task == 'en2de' and self.args.domains[0] in en2de_seen:
                    for i in en2de_seen:
                        if self.args.domains[0] == i:
                            source_lines = split_data.src
                            target_lines = split_data.tgt
                            src_datasets.append(IndexedRawLinesDataset(lines=source_lines, dictionary=self.src_dict))
                            tgt_datasets.append(IndexedRawLinesDataset(lines=target_lines, dictionary=self.tgt_dict))
                        else:
                            src_datasets.append(IndexedRawLinesDataset(lines='', dictionary=self.src_dict))
                            tgt_datasets.append(IndexedRawLinesDataset(lines='', dictionary=self.tgt_dict))
                elif self.args.translation_task == 'en2zh' and self.args.domains[0] in en2zh_seen:
                    for i in en2zh_seen:
                        if self.args.domains[0] == i:
                            source_lines = split_data.src
                            target_lines = split_data.tgt
                            src_datasets.append(IndexedRawLinesDataset(lines=source_lines, dictionary=self.src_dict))
                            tgt_datasets.append(IndexedRawLinesDataset(lines=target_lines, dictionary=self.tgt_dict))
                        else:
                            src_datasets.append(IndexedRawLinesDataset(lines='', dictionary=self.src_dict))
                            tgt_datasets.append(IndexedRawLinesDataset(lines='', dictionary=self.tgt_dict))


                # df_list = []
                # for i in range(self.args.domain_nums):
                #     df_list.append(split_data)
                # if 'dev' in split:
                #     for i in range(self.args.domain_nums):
                #         df_list.append(split_data)
                # else:
                #     for i in range(self.args.domain_nums):
                #         df_list.append(pd.concat([split_data] * 10).sample(frac=1))


                # for i, dgl in enumerate(df_list):
                #     if i == 2:
                #         source_lines = dgl.src
                #         target_lines = dgl.tgt
                #         src_datasets.append(IndexedRawLinesDataset(lines=source_lines, dictionary=self.src_dict))
                #         tgt_datasets.append(IndexedRawLinesDataset(lines=target_lines, dictionary=self.tgt_dict))
                #     else:
                #         src_datasets.append(IndexedRawLinesDataset(lines='', dictionary=self.src_dict))
                #         tgt_datasets.append(IndexedRawLinesDataset(lines='', dictionary=self.tgt_dict))
                #     # print('| {} {} {} examples'.format('', split, len(source_lines)))
                #
                # self.args.domain_nums = i + 1

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

                if bucket:
                    split = split + '_bucket'
                    self.datasets[split] = []

                    for i in range(3):
                        src_datasets = []
                        tgt_datasets = []
                        bucket_data = self.user_data_frame[self.user_data_frame.task_group.str.contains('B' + str(i))].reset_index()
                        if self.args.domains[0] in en2zh_unseen or self.args.domains[0] in en2de_unseen:
                            for i in range(self.args.domain_nums):
                                if i == 0:
                                    source_lines = bucket_data.src
                                    target_lines = bucket_data.tgt
                                    src_datasets.append(IndexedRawLinesDataset(lines=source_lines, dictionary=self.src_dict))
                                    tgt_datasets.append(IndexedRawLinesDataset(lines=target_lines, dictionary=self.tgt_dict))
                                else:
                                    src_datasets.append(IndexedRawLinesDataset(lines='', dictionary=self.src_dict))
                                    tgt_datasets.append(IndexedRawLinesDataset(lines='', dictionary=self.tgt_dict))
                        elif self.args.translation_task == 'en2de' and self.args.domains[0] in en2de_seen:
                            for i in en2de_seen:
                                if self.args.domains[0] == i:
                                    source_lines = bucket_data.src
                                    target_lines = bucket_data.tgt
                                    src_datasets.append(IndexedRawLinesDataset(lines=source_lines, dictionary=self.src_dict))
                                    tgt_datasets.append(IndexedRawLinesDataset(lines=target_lines, dictionary=self.tgt_dict))
                                else:
                                    src_datasets.append(IndexedRawLinesDataset(lines='', dictionary=self.src_dict))
                                    tgt_datasets.append(IndexedRawLinesDataset(lines='', dictionary=self.tgt_dict))
                        elif self.args.translation_task == 'en2zh' and self.args.domains[0] in en2zh_seen:
                            for i in en2zh_seen:
                                if self.args.domains[0] == i:
                                    source_lines = bucket_data.src
                                    target_lines = bucket_data.tgt
                                    src_datasets.append(IndexedRawLinesDataset(lines=source_lines, dictionary=self.src_dict))
                                    tgt_datasets.append(IndexedRawLinesDataset(lines=target_lines, dictionary=self.tgt_dict))
                                else:
                                    src_datasets.append(IndexedRawLinesDataset(lines='', dictionary=self.src_dict))
                                    tgt_datasets.append(IndexedRawLinesDataset(lines='', dictionary=self.tgt_dict))



                        # df_list = []
                        # for i in range(self.args.domain_nums):
                        #     df_list.append(bucket_data)
                        # # for i in range(self.args.domain_nums):
                        # #     df_list.append(pd.concat([bucket_data] * 10).sample(frac=1))
                        #
                        # for i, dgl in enumerate(df_list):
                        #     if i == 2:
                        #         source_lines = dgl.src
                        #         target_lines = dgl.tgt
                        #         src_datasets.append(IndexedRawLinesDataset(lines=source_lines, dictionary=self.src_dict))
                        #         tgt_datasets.append(IndexedRawLinesDataset(lines=target_lines, dictionary=self.tgt_dict))
                        #     else:
                        #         src_datasets.append(IndexedRawLinesDataset(lines='', dictionary=self.src_dict))
                        #         tgt_datasets.append(IndexedRawLinesDataset(lines='', dictionary=self.tgt_dict))
                        #     # print('| {} {} {} examples'.format('', split, len(source_lines)))
                        #
                        # # auto adjust domain nums
                        # self.args.domain_nums = i + 1

                        assert len(src_datasets) == len(tgt_datasets)

                        if len(src_datasets) == 1:
                            src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
                        else:
                            sample_ratios = [1] * len(src_datasets)
                            sample_ratios[0] = self.args.upsample_primary
                            src_dataset = ConcatDataset_IDX(src_datasets, sample_ratios)
                            tgt_dataset = ConcatDataset_IDX(tgt_datasets, sample_ratios)

                        self.datasets[split].append(LanguagePairDataset_IDX(
                            src_dataset, src_dataset.sizes, self.src_dict,
                            tgt_dataset, tgt_dataset.sizes, self.tgt_dict,
                            left_pad_source=self.args.left_pad_source,
                            left_pad_target=self.args.left_pad_target,
                            max_source_positions=self.args.max_source_positions,
                            max_target_positions=self.args.max_target_positions,
                        )
                        )

            # else:
            #     src_datasets = []
            #     tgt_datasets = []
            #     split_data = self.user_data_frame[self.user_data_frame.task_group.str.contains(split)]
            #     df_list = []
            #     for i in range(self.args.domain_nums):
            #         df_list.append(split_data)
            #     # if 'dev' in split:
            #     #     for i in range(self.args.domain_nums):
            #     #         df_list.append(split_data)
            #     # else:
            #     #     for i in range(self.args.domain_nums):
            #     #         df_list.append(pd.concat([split_data] * 10).sample(frac=1))
            #
            #     for i, dgl in enumerate(df_list):
            #         source_lines = dgl.src
            #         target_lines = dgl.tgt
            #         src_datasets.append(IndexedRawLinesDataset(lines=source_lines, dictionary=self.src_dict))
            #         tgt_datasets.append(IndexedRawLinesDataset(lines=target_lines, dictionary=self.tgt_dict))
            #         print('| {} {} {} examples'.format('', split, len(source_lines)))
            #
            #     self.args.domain_nums = i + 1
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
            #
            #     if bucket:
            #         split = split + '_bucket'
            #         self.datasets[split] = []
            #
            #         for i in range(3):
            #             src_datasets = []
            #             tgt_datasets = []
            #             bucket_data = self.user_data_frame[self.user_data_frame.task_group.str.contains('B' + str(i))].reset_index()
            #             df_list = []
            #             for i in range(self.args.domain_nums):
            #                 df_list.append(bucket_data)
            #             # for i in range(self.args.domain_nums):
            #             #     df_list.append(pd.concat([bucket_data] * 10).sample(frac=1))
            #
            #             for i, dgl in enumerate(df_list):
            #                 source_lines = dgl.src
            #                 target_lines = dgl.tgt
            #                 src_datasets.append(IndexedRawLinesDataset(lines=source_lines, dictionary=self.src_dict))
            #                 tgt_datasets.append(IndexedRawLinesDataset(lines=target_lines, dictionary=self.tgt_dict))
            #                 print('| {} {} {} examples'.format('', split, len(source_lines)))
            #
            #             # auto adjust domain nums
            #             self.args.domain_nums = i + 1
            #
            #             assert len(src_datasets) == len(tgt_datasets)
            #
            #             if len(src_datasets) == 1:
            #                 src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
            #             else:
            #                 sample_ratios = [1] * len(src_datasets)
            #                 sample_ratios[0] = self.args.upsample_primary
            #                 src_dataset = ConcatDataset_IDX(src_datasets, sample_ratios)
            #                 tgt_dataset = ConcatDataset_IDX(tgt_datasets, sample_ratios)
            #
            #             self.datasets[split].append(LanguagePairDataset_IDX(
            #                 src_dataset, src_dataset.sizes, self.src_dict,
            #                 tgt_dataset, tgt_dataset.sizes, self.tgt_dict,
            #                 left_pad_source=self.args.left_pad_source,
            #                 left_pad_target=self.args.left_pad_target,
            #                 max_source_positions=self.args.max_source_positions,
            #                 max_target_positions=self.args.max_target_positions,
            #             )
            #             )
