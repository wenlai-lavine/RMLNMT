from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math
import os
import sys

import numpy as np

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, KLDivLoss

from transformers.modeling_utils import PreTrainedModel, prune_linear_layer
from transformers.configuration_bert import BertConfig
from transformers.file_utils import add_start_docstrings
from transformers.modeling_bert import BertPreTrainedModel, BertModel


class BertForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """

    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def logits_from_current_epochs(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
        empprob_ref=None,
        bins=None,
        calloss_yn=None,
        calloss_type=None,
        calloss_lambda=None,
        caltrain_start_epochs=None,
        curr_epoch=None,
        eval_only=None,
        device=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[
            2:
        ]  # add hidden states and attention if they are here
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                # Input : Logit (No probability Function) / Label
                loss_fct = CrossEntropyLoss()

                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if eval_only:
                outputs = (loss,) + outputs
                return outputs

            # Calibration Loss!
            # if calloss_yn:
            #     Initial loss set-up
            loss_mle, loss_cal = torch.zeros(1), torch.zeros(1)
            # if curr_epochs > caltrain_start_epoch:
            if np.sum(empprob_ref) > 0:
                pred = torch.nn.functional.softmax(logits, dim=1)
                empprob = np.zeros(pred.shape)
                pred_to_empprob_map = np.digitize(pred.cpu().detach().numpy(), bins) - 1
                for sample, empmaps in enumerate(pred_to_empprob_map):
                    for clas, empprob_idx in enumerate(empmaps):
                        try:
                            empprob[sample][clas] = empprob_ref[empprob_idx][clas]
                        except IndexError:
                            empprob[sample][clas] = empprob_ref[empprob_idx - 1][clas]

                empprob = torch.from_numpy(empprob).float().to(device)
                if "MSE" in calloss_type:
                    loss_cal_fct = MSELoss()
                elif calloss_type == "KL":
                    loss_cal_fct = KLDivLoss(reduction="batchmean")
                    pred = torch.log(pred)

                # Update original loss containing calibration loss
                loss_cal = loss_cal_fct(pred.view(-1), empprob.view(-1))
                if calloss_type == "RMSE":
                    loss_cal = torch.sqrt(loss_cal)

                loss_cal = calloss_lambda * loss_cal
                loss_mle += loss
                if calloss_yn:
                    loss += loss_cal
            outputs = ([loss, loss_mle, loss_cal],) + outputs

        return outputs