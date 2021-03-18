import torch
import torch.nn as nn
import torch.nn.functional as F


class StableBCELoss(nn.modules.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, logit, label):
        neg_abs = - logit.abs()
        loss = logit.clamp(min=0) - logit * label + (1 + neg_abs.exp()).log()
        return loss.mean()


class BCELoss2d(nn.Module):
    def __init__(self):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        # self.bce_loss = StableBCELoss()

    def forward(self, logits, labels):

        # labels = (weight_labels==1).float()

        logits_flat = logits.view(-1)
        labels_flat = labels.view(-1)
        return self.bce_loss(logits_flat, labels_flat)


class BCELoss2d_posw(nn.Module):
    def __init__(self, w_pos):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w_pos))

    def forward(self, logits, labels):

        # labels = (weight_labels==1).float()

        logits_flat = logits.view(-1)
        labels_flat = labels.view(-1)

        return self.bce_loss(logits_flat, labels_flat)


class WeightedBCELoss2d(nn.Module):
    def __init__(self):
        super(WeightedBCELoss2d, self).__init__()

    def forward(self, logits, labels, weight_labels):

        w = weight_labels.view(-1)
        z = logits.view(-1)
        t = labels.view(-1)
        loss = w * z.clamp(min=0) - w * z * t + w * torch.log(1 + torch.exp(-z.abs()))
        loss = loss.sum() / (w.sum() + 1e-12)
        return loss


class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, labels):
        probs = torch.sigmoid(logits)

        smooth = 1.0

        m1 = probs.view(-1)
        m2 = labels.view(-1)

        intersection = (m1 * m2).sum()
        score = (2.0 * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

        return 1 - score


class SoftDiceLoss_vnet(nn.Module):
    def __init__(self):
        super(SoftDiceLoss_vnet, self).__init__()

    def forward(self, logits, labels):
        probs = torch.sigmoid(logits)

        smooth = 1.0

        m1 = probs.view(-1)
        m2 = labels.view(-1)

        intersection = (m1 * m2).sum()
        score = (2.0 * intersection + smooth) / ((m1 * m1).sum() + (m2 * m2).sum() + smooth)

        return 1 - score


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, labels):

        input = logits.view(-1)
        target = labels.view(-1)

        max_val = (-input).clamp(min=0)

        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.mean()


class BCEDice(nn.Module):
    def __init__(self, alpha, w_pos):
        super().__init__()
        self.alpha = alpha
        self.dice = SoftDiceLoss()
        self.bce = BCELoss2d_posw(w_pos=w_pos)

    def forward(self, logits, labels):
        loss = self.alpha * self.bce(logits, labels) + (1 - self.alpha) * self.dice(logits, labels)
        return loss


class DiceAccuracy(nn.Module):
    def __init__(self):
        super(DiceAccuracy, self).__init__()

    def forward(self, logits, labels):
        probs = torch.sigmoid(logits)
        # probs = (probs > 0.5).float()
        smooth = 1e-12

        m1 = probs.view(-1)
        m2 = labels.view(-1)

        intersection = (m1 * m2).sum()
        score = (2.0 * intersection) / (m1.sum() + m2.sum() + smooth)

        return score


class IouAccuracy(nn.Module):
    def __init__(self):
        super(IouAccuracy, self).__init__()

    def forward(self, logits, labels):
        probs = torch.sigmoid(logits)
        # probs = (probs > 0.5).float()
        smooth = 1e-12

        m1 = probs.view(-1)
        m2 = labels.view(-1)

        intersection = (m1 * m2).sum()
        union = (m1 + m2 - m1 * m2).sum()
        score = intersection / (union + smooth)

        return score


def numeric_score(logits, labels):
    """Computes scores:
    FP = False Positives
    FN = False Negatives
    TP = True Positives
    TN = True Negatives
    return: FP, FN, TP, TN"""

    probs = torch.sigmoid(logits)
    probs = (probs > 0.5).float()

    FP = ((probs == 1) & (labels == 0)).sum()
    FN = ((probs == 0) & (labels == 1)).sum()
    TP = ((probs == 1) & (labels == 1)).sum()
    TN = ((probs == 0) & (labels == 0)).sum()

    return FP, FN, TP, TN


class Precision(nn.Module):
    def __init__(self):
        super(Precision, self).__init__()

    def forward(self, logits, labels):

        FP, FN, TP, TN = numeric_score(logits, labels)
        precision = TP / (TP + FP + 1e-12)

        return precision


class Recall(nn.Module):
    def __init__(self):
        super(Recall, self).__init__()

    def forward(self, logits, labels):

        FP, FN, TP, TN = numeric_score(logits, labels)
        recall = TP / (TP + FN + 1e-12)

        return recall
