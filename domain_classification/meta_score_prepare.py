#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 09.09.21 20:58
# @Author  : Wen Lai
# @Site    : 
# @File    : meta_score_prepare.py
# @Usage information: 

# Copyright (c) 2021-present, CIS, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


"""
Example: 
python meta_score_prepare.py \
--num_labels 11 \
--device_id 0 \
--model_name bert-base-uncased \
--input_path /mounts/data/proj/lavine/domain_adaptation/paper_data/paper/data/repro_data/filter_data/2_deduplicated \
--cls_data /mounts/data/proj/lavine/domain_adaptation/paper_data/paper/data/repro_data/classifier/11_labels \
--out_data /mounts/data/proj/lavine/domain_adaptation/paper_data/paper/experiments/repro_exp/classification/BERT_11_labels \
--script_path /mounts/work/lavine/lavine_code/DA-NMT/classification

"""
import argparse
import os
import subprocess


def main(args):
    domain_list = ['General', 'EMEA', 'GlobalVoices', 'JRC', 'KDE', 'WMT']

    for dl in domain_list:
        print('preprocess ' + dl + ' domain ....')
        if args.num_labels == "11":
            gen_csv_query = "sed " + "'s/^/" + dl + "\t/'" + " " \
                            + os.path.join(args.input_path, dl + '.de-en.en') \
                            + " > " + os.path.join(args.cls_data, 'test.tsv')
        elif args.num_labels == "2":
            if dl == 'General':
                gen_csv_query = "sed " + "'s/^/" + "indomain" + "\t/'" + " " \
                                + os.path.join(args.input_path, dl + '.de-en.en') \
                                + " > " + os.path.join(args.cls_data, 'test.tsv')
            else:
                gen_csv_query = "sed " + "'s/^/" + "outdomain" + "\t/'" + " " \
                                + os.path.join(args.input_path, dl + '.de-en.en') \
                                + " > " + os.path.join(args.cls_data, 'test.tsv')

        rm_cache = 'rm ' + os.path.join(args.cls_data, 'cached_test_' + args.model_name + '_128_domain_adapt')

        gen_score = 'CUDA_VISIBLE_DEVICES=' + args.device_id + ' python ' \
                    + os.path.join(args.script_path, 'eval_score.py') \
                    + ' --model_type bert --model_name_or_path bert-base-uncased' \
                    + ' --task_name domain_adapt --do_eval --do_lower_case ' \
                    + ' --data_dir ' + args.cls_data \
                    + ' --max_seq_length 128 --per_gpu_eval_batch_size=1024' \
                    + ' --per_gpu_train_batch_size=32 --learning_rate 2e-5' \
                    + ' --num_train_epochs 20' \
                    + ' --num_labels ' + args.num_labels \
                    + ' --output_dir ' + args.out_data

        cp_final_score = 'cp ' + os.path.join(args.out_data, 'score.txt') \
                         + ' ' + os.path.join(args.out_data, 'score', dl + '.score')

        print('---------' + 'gen_csv_query' + '---------')
        subprocess.call(gen_csv_query, shell=True)
        print('---------' + 'rm_cache' + '---------')
        subprocess.call(rm_cache, shell=True)
        print('---------' + 'gen_score' + '---------')
        subprocess.call(gen_score, shell=True)
        print('---------' + 'cp_final_score' + '---------')
        subprocess.call(cp_final_score, shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, type=str, help='input path')
    parser.add_argument("--num_labels", required=True, type=str, help='input path')
    parser.add_argument("--model_name", required=True, type=str, help='input path')
    parser.add_argument("--device_id", required=True, type=str, help='input path')
    parser.add_argument("--cls_data", required=True, type=str, help='input path')
    parser.add_argument("--out_data", required=True, type=str, help='input path')
    parser.add_argument("--script_path", required=True, type=str, help='input path')
    main(parser.parse_args())