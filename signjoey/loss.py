# coding: utf-8
"""
Module to implement training loss
"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.autograd import Variable


class XentLoss(nn.Module):
    """
    Cross-Entropy Loss with optional label smoothing
    """

    def __init__(self, pad_index: int, smoothing: float = 0.0):
        super(XentLoss, self).__init__()
        self.smoothing = smoothing
        self.pad_index = pad_index
        if self.smoothing <= 0.0:
            # standard xent loss
            self.criterion = nn.NLLLoss(ignore_index=self.pad_index, reduction="sum")
        else:
            # custom label-smoothed loss, computed with KL divergence loss
            self.criterion = nn.KLDivLoss(reduction="sum")

    def _smooth_targets(self, targets: Tensor, vocab_size: int):
        """
        Smooth target distribution. All non-reference words get uniform
        probability mass according to "smoothing".

        :param targets: target indices, batch*seq_len
        :param vocab_size: size of the output vocabulary
        :return: smoothed target distributions, batch*seq_len x vocab_size
        """
        # batch*seq_len x vocab_size
        smooth_dist = targets.new_zeros((targets.size(0), vocab_size)).float()
        # fill distribution uniformly with smoothing
        smooth_dist.fill_(self.smoothing / (vocab_size - 2))
        # assign true label the probability of 1-smoothing ("confidence")
        smooth_dist.scatter_(1, targets.unsqueeze(1).data, 1.0 - self.smoothing)
        # give padding probability of 0 everywhere
        smooth_dist[:, self.pad_index] = 0
        # masking out padding area (sum of probabilities for padding area = 0)
        padding_positions = torch.nonzero(targets.data == self.pad_index)
        # pylint: disable=len-as-condition
        if len(padding_positions) > 0:
            smooth_dist.index_fill_(0, padding_positions.squeeze(), 0.0)
        return Variable(smooth_dist, requires_grad=False)

    # pylint: disable=arguments-differ
    def forward(self, log_probs, targets):
        """
        Compute the cross-entropy between logits and targets.

        If label smoothing is used, target distributions are not one-hot, but
        "1-smoothing" for the correct target token and the rest of the
        probability mass is uniformly spread across the other tokens.

        :param log_probs: log probabilities as predicted by model
        :param targets: target indices
        :return:
        """
        if self.smoothing > 0:
            targets = self._smooth_targets(
                targets=targets.contiguous().view(-1), vocab_size=log_probs.size(-1)
            )
            # targets: distributions with batch*seq_len x vocab_size
            assert (
                log_probs.contiguous().view(-1, log_probs.size(-1)).shape
                == targets.shape
            )
        else:
            # targets: indices with batch*seq_len
            targets = targets.contiguous().view(-1)
        loss = self.criterion(
            log_probs.contiguous().view(-1, log_probs.size(-1)), targets
        )
        return loss


# Clément
class AnchoringLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.T = 2 # temperature
        self.cls_criterion = nn.KLDivLoss(reduction='sum')
        self.reg_criterion = nn.SmoothL1Loss(reduction='sum')

    def forward(self,
                scores,
                pose_deltas,
                body_scores,
                body_deltas,
                face_scores,
                face_deltas,
                hand_scores_1,
                hand_deltas_1,
                hand_scores_2,
                hand_deltas_2
                ):

        cls_loss_body = self.cls_criterion(F.log_softmax(scores['body'] / self.T, dim=-1),
                                           F.softmax(body_scores / self.T, dim=-1))
        cls_loss_face = self.cls_criterion(F.log_softmax(scores['face'] / self.T, dim=-1),
                                           F.softmax(face_scores / self.T, dim=-1))
        cls_loss_hand_1 = self.cls_criterion(F.log_softmax(scores['hand_1'] / self.T, dim=-1),
                                             F.softmax(hand_scores_1 / self.T, dim=-1))
        cls_loss_hand_2 = self.cls_criterion(F.log_softmax(scores['hand_2'] / self.T, dim=-1),
                                             F.softmax(hand_scores_2 / self.T, dim=-1))
        cls_loss = (cls_loss_body + cls_loss_face + cls_loss_hand_1 + cls_loss_hand_2) / 4

        reg_loss_body = self.reg_criterion(pose_deltas['body'], body_deltas)
        reg_loss_face = self.reg_criterion(pose_deltas['face'], face_deltas)
        reg_loss_hand_1 = self.reg_criterion(pose_deltas['hand_1'], hand_deltas_1)
        reg_loss_hand_2 = self.reg_criterion(pose_deltas['hand_2'], hand_deltas_2)
        reg_loss = (reg_loss_body + reg_loss_face + reg_loss_hand_1 + reg_loss_hand_2) / 4 / 1000

        return cls_loss, reg_loss
