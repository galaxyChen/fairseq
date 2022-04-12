# Copyright (c) Microsoft. All rights reserved.
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F
from torch.nn import KLDivLoss

from fairseq import utils, metrics

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from dataclasses import dataclass, field
from omegaconf import II


@dataclass
class AdvLabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        non_pad_mask = target.ne(ignore_index)
        nll_loss = nll_loss[non_pad_mask]
        smooth_loss = smooth_loss[non_pad_mask]
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def KL(input, target, reduce=True):
    loss_func = KLDivLoss()
    input = F.log_softmax(input, dim=-1)
    target = F.softmax(target, dim=-1)
    loss = loss_func(input, target)
    return loss


@register_criterion('adv_label_smoothed_cross_entropy', dataclass=AdvLabelSmoothedCrossEntropyCriterionConfig)
class AdvLabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, task, label_smoothing, sentence_avg):
        super().__init__(task)
        self.task = task
        self.eps = label_smoothing
        self.sentence_avg = sentence_avg

    def adv_project(self, grad, norm_type='inf', eps=1e-6):
        if norm_type == 'l2':
            direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + eps)
        elif norm_type == 'l1':
            direction = grad.sign()
        else:
            direction = grad / (grad.abs().max(-1, keepdim=True)[0] + eps)
        return direction

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output, enc_emb = model(**sample['net_input'])
        loss, nll_loss, logits = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        if self.task.args.log_variance or 'adaptive-profiling' == self.task.args.admin_init_type:
            exit()
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss, lprobs

    def log_probs(self, model, net_output):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1)).float()
        return lprobs

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(
                2) if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(
                2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

    @classmethod
    def reduce_metrics(cls, logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        log_dict = {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(
                2) if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(
                2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

        for name in log_dict:
            metrics.log_scalar(name, log_dict[name])
        metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg))


@register_criterion('adv_label_smoothed_cross_entropy_kl', dataclass=AdvLabelSmoothedCrossEntropyCriterionConfig)
class AdvLabelSmoothedCrossEntropyKLCriterion(AdvLabelSmoothedCrossEntropyCriterion):

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output, teacher_output = model(**sample['net_input'])
        ce_loss, nll_loss, kl_loss, logits = self.compute_loss(model, net_output, teacher_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']

        loss = (ce_loss + kl_loss) / 2
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ce_loss': utils.item(ce_loss.data) if reduce else ce_loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            "kl_loss": utils.item(kl_loss.data) if reduce else kl_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        if self.task.args.log_variance or 'adaptive-profiling' == self.task.args.admin_init_type:
            exit()
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, teacher_output, sample, reduce=True):
        kl_loss = KL(net_output[0], teacher_output[0], reduce=reduce)
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss, kl_loss, lprobs

    @classmethod
    def reduce_metrics(cls, logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        log_dict = {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(
                2) if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(
                2) if ntokens > 0 else 0.,
            'kl_loss': sum(log.get('kl_loss', 0) for log in logging_outputs) / sample_size if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

        for name in log_dict:
            metrics.log_scalar(name, log_dict[name])
        metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg))
