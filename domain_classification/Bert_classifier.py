from __future__ import absolute_import, division, print_function

import argparse
import csv
import glob
import os
from collections import defaultdict
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
import coloredlogs, logging
from colorama import Fore, Style
from tqdm import tqdm, trange
from transformers import (
    WEIGHTS_NAME,
    BertConfig,
    BertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)

from utils import (
    compute_metrics,
    convert_examples_to_features,
    output_modes,
    processors,
)
from modeling_bert import BertForSequenceClassification

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = logging.getLogger(__name__)
coloredlogs.install(
    fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
    level="INFO",
    datefmt="%m/%d %H:%M:%S",
    logger=logger,
)

def highlight(input):
    return Fore.YELLOW + str(input) + Style.RESET_ALL


ALL_MODELS = tuple(BertConfig.pretrained_config_archive_map.keys())
MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def setup_bins(bin_size, adaptive=False):
    if adaptive:
        return "TODO"
    else:
        return np.linspace(0, 1, bin_size + 1)


def load_and_cache_examples(args, task, tokenizer, data_type="train", show_stats=True):
    #### 加载数据集
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    processor = processors[task]()
    output_mode = output_modes[task]
    label_list = processor.get_labels()

    if args.eval_dataset is not None:
        basename_eval_dataset = os.path.basename(args.eval_dataset).replace(".txt", "")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            args.data_dir,
            "../cached_{}={}_{}_{}".format(
                "test_input" if data_type == "test" else data_type,
                #'test_input' if evaluate else 'train',
                basename_eval_dataset,
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                str(args.max_seq_length),
            ),
        )
    else:
        # Load data features from cache or dataset file
        ### data_feature
        #### 数据的feature保存的文件路径
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}_{}".format(
                data_type,  ##'test' if evaluate else 'train',
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                str(args.max_seq_length),
                str(task),
            ),
        )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        ### 已存在cache——feature， 直接加载
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        data_dir_eval = (
            args.eval_dataset if args.eval_dataset is not None else args.data_dir
        )
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if data_type == "test":
            examples = processor.get_test_examples(args.data_dir)
        elif data_type == "train":
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == "dev":
            examples = processor.get_dev_examples(args.data_dir)

        features = convert_examples_to_features(
            examples,
            label_list,
            args.max_seq_length,
            tokenizer,
            output_mode,
            cls_token_at_end=False,
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            cls_token_segment_id=0,
            pad_on_left=False,
            pad_token_segment_id=0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if show_stats and args.eval_dataset is None:
        nonzero_cnt = 0
        for feature in features:
            nonzero_cnt += np.count_nonzero(feature.input_ids) - 2

        # Vocab size
        vocab_dict = defaultdict(int)
        for feature in features:
            for f in feature.input_ids:
                vocab_dict[f] += 1

        # feature distribution
        feature_dict = defaultdict(int)
        if output_mode == "classification":
            for feature in features:
                feature_dict[label_list[feature.label_id]] += 1
            total_c = sum([c for _, c in feature_dict.items()])

        stats = {
            "num_features": len(features),
            "avg_sent_len": nonzero_cnt / len(features),
            "vocab_size": len(vocab_dict),
            "feature_dist": feature_dict,
        }

        for key in stats.keys():
            logger.info(" {} = {}".format(key, highlight(stats[key])))

    if args.local_rank == 0:
        torch.distributed.barrier()

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    if args.eval_dataset is None:
        if output_mode == "classification":
            all_label_ids = torch.tensor(
                [f.label_id for f in features], dtype=torch.long
            )
        elif output_mode == "regression":
            all_label_ids = torch.tensor(
                [f.label_id for f in features], dtype=torch.float
            )
    else:
        all_label_ids = None

    if all_label_ids is None:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    else:
        dataset = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_label_ids
        )
    return dataset


def train(args, train_dataset, model, tokenizer, num_labels, valid_dataset=None):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size
    )

    if valid_dataset is not None:
        valid_sampler = (
            RandomSampler(valid_dataset)
            if args.local_rank == -1
            else DistributedSampler(valid_dataset)
        )
        valid_dataloader = DataLoader(
            valid_dataset, sampler=valid_sampler, batch_size=args.train_batch_size
        )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps
            // (len(train_dataloader) // args.gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    num_epochs = 0
    tr_loss, logging_loss = 0.0, 0.0

    ##setup bins
    bins = setup_bins(args.bin_size)
    p_emps = np.zeros((len(bins) - 1, num_labels))

    val_loss_per_steps = []
    model.zero_grad()
    train_iterator = trange(
        int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for epoch in train_iterator:
        num_epochs += 1
        epoch_iterator = tqdm(
            train_dataloader,
            desc="Iteration",
            disable=args.local_rank not in [-1, 0],
            ncols=8,
        )
        ### 三种损失
        epoch_loss, epoch_loss_mle, epoch_loss_cal = 0.0, 0.0, 0.0
        update_freq = len(epoch_iterator) // args.num_updates
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3],
                "empprob_ref": p_emps,
                "bins": bins,
                "calloss_yn": args.poscal_train,
                "calloss_type": args.calloss_type,
                "calloss_lambda": args.calloss_lambda,
                "caltrain_start_epochs": args.calloss_start_epochs,
                "curr_epoch": num_epochs,
                "device": args.device,
            }

            outputs = model(**inputs)
            loss = outputs[0][0]
            loss_mle = outputs[0][1]
            loss_cal = outputs[0][2]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            epoch_loss += loss.item()
            epoch_loss_cal += loss_cal.item()
            epoch_loss_mle += loss_mle.item()

            # logging
            epoch_iterator.desc = "[{}] Loss:{:.2f} lr:{:.1e}".format(
                epoch, tr_loss / (step + 1), scheduler.get_lr()[0]
            )

            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if (
                    args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and global_step % args.logging_steps == 0
                ):
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar(
                                "eval_{}".format(key), value, global_step
                            )
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        "loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_step,
                    )
                    logging_loss = tr_loss

                if (
                    args.local_rank in [-1, 0]
                    and args.save_steps > 0
                    and global_step % args.save_steps == 0
                ):
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    # logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

            ##Update p_emps
            if (
                num_epochs > args.calloss_start_epochs
                and args.poscal_train
                and ((step + 1) % update_freq == 0)
            ) or ((step + 1) == len(epoch_iterator)):

                freq_p_hats = np.zeros((len(bins) - 1, num_labels))
                freq_p_trues = np.zeros((len(bins) - 1, num_labels))
                acc_p_hats = np.zeros((len(bins) - 1, num_labels))

                print(
                    "Update p_emp at num_epochs %d, steps %d" % (num_epochs, step + 1)
                )
                for cal_step, cal_batch in enumerate(epoch_iterator):
                    cal_batch = tuple(t.to(args.device) for t in cal_batch)
                    labels = cal_batch[3]
                    inputs = {
                        "input_ids": cal_batch[0],
                        "attention_mask": cal_batch[1],
                        "token_type_ids": cal_batch[2],
                    }
                    with torch.no_grad():
                        logits = model.logits_from_current_epochs(**inputs)

                    p_hats = (
                        torch.nn.functional.softmax(logits, dim=1)
                        .cpu()
                        .detach()
                        .numpy()
                    )
                    p_binums = np.digitize(p_hats, bins) - 1
                    p_trues = np.eye(num_labels)[labels.cpu()]

                    for sample in np.dstack((p_binums, p_trues, p_hats)):
                        for col_idx, subsample in enumerate(sample):
                            row_idx = int(subsample[0])
                            indiv_true = subsample[1]
                            indiv_p_hat = subsample[2]
                            try:
                                freq_p_hats[row_idx][col_idx] += 1
                                freq_p_trues[row_idx][col_idx] += indiv_true
                                acc_p_hats[row_idx][col_idx] += indiv_p_hat
                            except IndexError:
                                freq_p_hats[row_idx - 1][col_idx] += 1
                                freq_p_trues[row_idx - 1][col_idx] += indiv_true
                                acc_p_hats[row_idx - 1][col_idx] += indiv_p_hats

                p_emps = freq_p_trues / freq_p_hats
                p_emps[np.isnan(p_emps)] = 0

                # ECE
                p_confs = acc_p_hats / freq_p_hats
                p_confs[np.isnan(p_confs)] = 0

            if valid_dataset is not None:
                val_loss = 0.0
                if num_epochs > 0 and global_step % 500 == 0:
                    for val_step, val_batch in enumerate(
                        tqdm(valid_dataloader, desc="Valid")
                    ):
                        val_batch = tuple(t.to(args.device) for t in val_batch)
                        inputs = {
                            "input_ids": val_batch[0],
                            "attention_mask": val_batch[1],
                            "token_type_ids": val_batch[2],
                            "labels": val_batch[3],
                            "empprob_ref": p_emps,
                            "bins": bins,
                            "calloss_yn": args.poscal_train,
                            "calloss_type": args.calloss_type,
                            "calloss_lambda": args.calloss_lambda,
                            "caltrain_start_epochs": args.calloss_start_epochs,
                            "curr_epoch": num_epochs,
                            "device": args.device,
                        }

                        # print("Update ValLoss %d" %(len(val_loss_per_steps)))
                        with torch.no_grad():
                            outputs = model(**inputs)
                        loss = outputs[0][0]
                        val_loss += loss.item()

                    if (
                        len(val_loss_per_steps) > 10
                        and np.mean(val_loss_per_steps[-10:]) < val_loss
                    ):
                        break
                    else:
                        val_loss_per_steps.append(val_loss)

            if args.max_steps > 0 and global_step > args.max_steps:
                train_iterator.close()
                break

        ##Save training loss
        try:
            train_ece = np.sum(freq_p_hats * (abs(p_confs - p_emps))) / np.sum(
                freq_p_hats[:, 0]
            )
        except UnboundLocalError:
            train_ece = -9999

        output_train_file = os.path.join(args.output_dir, "train.txt")
        with open(output_train_file, "a") as writer:
            logger.info("***** Save Train Loss *****")

            # task, model, epochs,total_loss,mle_loss,poscal_loss,train_ece
            if args.poscal_train:
                rtypes = "PosCal"
            else:
                rtypes = "MLE"

            writer.write("%s\t%s\t%d\t" % (args.task_name, rtypes, num_epochs))
            writer.write(
                "%.4f\t%.4f\t%.4f\t" % (epoch_loss, epoch_loss_mle, epoch_loss_cal)
            )
            writer.write("%.4f" % (train_ece))
            writer.write("\n")

        if valid_dataset is not None:
            if num_epochs > 1:
                if (
                    len(val_loss_per_steps) > 5
                    and np.mean(val_loss_per_steps[-5:]) < val_loss
                ):
                    print(
                        "Early stopping in num_epochs %d, steps %d"
                        % (num_epochs, step + 1)
                    )
                    train_iterator.close()
                    break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, num_labels, prefix=""):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}

    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(
            args, eval_task, tokenizer, data_type="test"
        )

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_sampler = (
            SequentialSampler(eval_dataset)
            if args.local_rank == -1
            else DistributedSampler(eval_dataset)
        )
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
        )

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        bins = setup_bins(args.bin_size)
        out_p_hats = []
        freq_p_hats = np.zeros((args.bin_size, num_labels))
        freq_p_trues = np.zeros((args.bin_size, num_labels))
        acc_p_hats = np.zeros((args.bin_size, num_labels))

        for batch in tqdm(eval_dataloader, desc="Evaluating", ncols=8):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3],
                    "eval_only": True,
                }

                # if args.eval_dataset is None:
                #    inputs['labels'] = batch[3]

                outputs = model(**inputs)

                if len(outputs) >= 2:
                    tmp_eval_loss, logits = outputs[:2]
                    eval_loss += tmp_eval_loss.mean().item()
                elif len(outputs) == 1:
                    logits = outputs[0]

            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
                )

            # To get ece
            p_hats = torch.nn.functional.softmax(logits, dim=1).cpu().detach().numpy()
            p_binums = np.digitize(p_hats, bins) - 1
            p_trues = np.eye(num_labels)[batch[3].cpu()]

            out_p_hats += p_hats.tolist()
            for sample in np.dstack((p_binums, p_trues, p_hats)):
                for col_idx, subsample in enumerate(sample):
                    row_idx = int(subsample[0])
                    indiv_true = subsample[1]
                    indiv_p_hat = subsample[2]
                    try:
                        freq_p_hats[row_idx][col_idx] += 1
                        freq_p_trues[row_idx][col_idx] += indiv_true
                        acc_p_hats[row_idx][col_idx] += indiv_p_hat
                    except IndexError:
                        freq_p_hats[row_idx - 1][col_idx] += 1
                        freq_p_trues[row_idx - 1][col_idx] += indiv_true
                        acc_p_hats[row_idx - 1][col_idx] += index_p_hat

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds_prob_score = np.max(preds, axis=1)
            preds_ids = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds_ids = np.squeeze(preds)

        if out_label_ids is not None:
            result = compute_metrics(eval_task, preds_ids, out_label_ids)
            results.update(result)

        ##Update p_emps
        p_emps = freq_p_trues / freq_p_hats
        p_emps[np.isnan(p_emps)] = 0

        # Get ECE
        p_confs = acc_p_hats / freq_p_hats
        p_confs[np.isnan(p_confs)] = 0

        eval_ece = np.sum(freq_p_hats * (abs(p_confs - p_emps))) / np.sum(
            freq_p_hats[:, 0]
        )

        # For Outfiles
        if args.poscal_train:
            rtypes = "PosCal"
        else:
            rtypes = "MLE"

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results{} *****".format(prefix))
            writer.write("%s\t%s\t" % (args.task_name, rtypes))
            for key in sorted(result.keys()):
                writer.write("%s \t %s \t" % (key, str(result[key])))

            writer.write("eval_ece \t %.6f" % (eval_ece))
            writer.write("\n")

        output_prob_label_file = os.path.join(args.output_dir, "prob_labels.tsv")
        with open(output_prob_label_file, "w") as prob_label_file:
            logger.info("***** prob plus labels file writing{} *****".format(prefix))
            tsv_w = csv.writer(prob_label_file, delimiter='\t')
            tsv_w.writerow(['labels', 'score'])
            for i in range(len(preds_ids)):
                tsv_w.writerow([preds_ids[i], preds_prob_score[i]])

    return results


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: "
        + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    ## Other parameters
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Rul evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight deay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )

    parser.add_argument(
        "--logging_steps", type=int, default=50, help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=10000,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )

    # added for external input for testing
    parser.add_argument(
        "--eval_dataset", default=None, type=str, help="Additional eval dataset "
    )

    # For Posterior Reinforced Training
    parser.add_argument("--num_updates", default=1, type=int)
    parser.add_argument("--bin_size", default=20, type=int)
    parser.add_argument("--calloss_start_epochs", default=0, type=int)
    parser.add_argument("--poscal_train", default=False, action="store_true")
    parser.add_argument("--calloss_lambda", default=1, type=float)
    parser.add_argument("--calloss_type", default="KL", type=str)

    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Set seed
    set_seed(args)

    # Prepare task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    ### 掉utils下面的具体的类(每个任务都有类)
    processor = processors[args.task_name]()
    ### 选择classification 还是 regression
    args.output_mode = output_modes[args.task_name]
    ### 获取label_list
    label_list = processor.get_labels()
    num_labels = len(label_list)
    logger.info(
        "Processor: {}, label: {} ({})".format(processor, label_list, num_labels)
    )

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        ### 加载数据集进行训练
        train_dataset = load_and_cache_examples(
            args, args.task_name, tokenizer, data_type="train"
        )
        valid_dataset = load_and_cache_examples(
            args, args.task_name, tokenizer, data_type="dev"
        )

        if not os.path.exists(os.path.join(args.output_dir, "training_args.bin")):
            ### 开始训练
            global_step, tr_loss = train(
                args,
                train_dataset,
                model,
                tokenizer,
                num_labels,
                valid_dataset=valid_dataset,
            )
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Load a trained model and vocabulary that you have fine-tuned
    model = model_class.from_pretrained(args.output_dir)
    tokenizer = tokenizer_class.from_pretrained(
        args.output_dir, do_lower_case=args.do_lower_case
    )
    model.to(args.device)

    # Evaluation
    result = evaluate(args, model, tokenizer, num_labels, prefix="")
    return result


if __name__ == "__main__":
    main()