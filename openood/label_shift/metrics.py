import numpy as np
import logging
import torch
import torch.nn as nn
from scipy.stats import entropy
from typing import List
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import confusion_matrix


def get_ls_metrics(probs, labels, w, pi, w_gt):
    # print(list(set(labels)), probs.shape, len(w), len(pi), len(w_gt))
    metric = {}
    metric = get_acc_metrics(probs, labels)

    w_mse = get_mse(w, w_gt)

    metric['w_mse'] = w_mse
    metric['w'] = w
    metric['pi'] = pi

    # print('Estimated w_mse is {:.4f}'.format(w_mse))

    return metric


def get_mse(a, b):
    return np.mean(np.power(a - b, 2))


def round_val(metrics):
    # Round metric values to a more reader friendly form
    for k, v in metrics.items():
        if type(v) in [np.ndarray, list]:
            metrics[k] = [round(float(x), 4) for x in list(v)]
        else:
            metrics[k] = round(float(v), 4)

    return metrics


def get_acc_metrics(probs, labels):
    cls_num = probs.shape[-1]
    pred = np.argmax(probs, axis=-1)
    # print(list(set(pred)))
    acc = acc_cal(probs, labels, method='top1')
    # print('Evaluation Top1 Acc %.4f' % acc)
    ood_acc = np.sum([x == cls_num for x in pred]) / len(pred)
    # print('Evaluation Top1 OOD Acc %.4f' % ood_acc)

    matrix = confusion_matrix(labels, pred)
    per_cls_acc = matrix.diagonal() / matrix.sum(axis=1)
    # print(len(matrix.diagonal()), len(matrix.sum(axis=1)))
    avg_per_cls_acc = per_cls_acc.mean() * 100
    # print('Per-Class Top1 Acc %.4f' % (avg_per_cls_acc))

    # mmf_acc = list(mmf_acc_cal(probs, labels, cls_num_list))
    # print('Many Medium Few shot Top1 Acc: ' + str(mmf_acc))

    # precision, recall, f1, _ = prfs(labels, pred, average='micro')
    # print('(micro) Precision: %.4f, Recall: %.4f, F Score: %.4f' % (precision, recall, f1))

    precision, recall, f1, support = prfs(labels, pred, average='macro', zero_division=0)
    print('Acc: %.4f (macro) Precision: %.4f, Recall: %.4f, F Score: %.4f' % (acc, precision, recall, f1))

    # pc_ece = ece_loss(torch.Tensor(np.array(pc_probs)), torch.LongTensor(np.array(labels))).detach().cpu().numpy()
    ece = ECECal(np.array(probs), list(labels))
    sce = SCECal(np.array(probs), list(labels), cls_num)
    bier = BierCal(np.array(probs), list(labels))
    ent = EntropyCal(np.array(probs))
    print('ECE, SCE, Bier, Entropy of current model: %.4f, %.4f, %.4f, %.4f' % (ece, sce, bier, ent))


    result = {'acc': acc,
              'ood_acc': ood_acc,
              'sce': sce,
              'ece': ece,
              'bier': bier,
              'entropy': ent,
              #'mmf_acc': mmf_acc,
              'cls_acc': avg_per_cls_acc,
              'precision': precision,
              'recall': recall,
              'f1': f1}

    result = round_val(result)

    return result


def get_posterior_statistics(result_list):
    result = {
        'qy_all': np.array(result_list),
        'qy_mean': np.mean(result_list, axis=0),
        'qy_max': np.max(result_list, axis=0),
        'qy_min': np.min(result_list, axis=0),
        'qy_std': np.std(result_list, axis=0),
        'qy_median': np.median(result_list, axis=0),
    }

    qy_hist = []
    qy_hist_edges = []
    data_num = len(result_list)
    n_bins = 10
    for i, x in enumerate(np.array(result_list).T):
        mean = result['qy_mean'][i]
        std = result['qy_std'][i]
        hist_range = (max(mean - 3 * std, 0), min(mean + 3 * std, 1))
        hist, hist_edges = np.histogram(x, range=hist_range, bins=n_bins, density=True)
        qy_hist.append(hist)
        qy_hist_edges.append(hist_edges)

    result.update({'qy_hist': np.array(qy_hist),
                   'qy_hist_edges': np.array(qy_hist_edges),
                   'qy_n_bins': np.array([n_bins])}
                  )

    return result


def acc_cal(logits, label: List[int], method: str = 'top1'):
    if method == 'top1':
        label_pred = np.argsort(logits, -1).T[-1]
        correct = np.sum([i == j for i, j in zip(label_pred, label)])
        total = len(label)

    result = correct / total * 100
    return round(float(result), 4)


def ClsAccCal(logits, label: List[int], method: str = 'top1'):
    if method == 'top1':
        label_pred = np.argsort(logits, -1).T[-1]

        label_set = np.arange(logits.shape[-1])
        correct = np.zeros(len(label_set))
        total = np.zeros(len(label_set)) + 1e-6
        for i, j in zip(label_pred, label):
            correct[j] += (i == j)
            total[j] += 1

    result = np.array(correct) / np.array(total) * 100
    return result.round(4).tolist()


def mmf_acc_cal(logits, label: List[int], class_num_list: List[int], method: str = 'top1'):
    # many medium few - shot accuracy calculation
    correct = np.zeros(3)
    total = np.zeros(3) + 1e-6
    mmf_id = list(map(get_mmf_idx, list(class_num_list)))
    if method == 'top1':
        label_pred = np.argsort(logits, -1).T[-1]
        for i, j in zip(label_pred, label):
            correct[mmf_id[j]] += (i == j)
            total[mmf_id[j]] += 1

    result = np.array(correct) / np.array(total) * 100
    return result.round(4).tolist()


def get_mmf_idx(img_num: int):
    assert type(img_num) == int
    if img_num < 20:
        return 2
    elif img_num < 100:
        return 1
    else:
        return 0


# Original Code from https://github.com/gpleiss/temperature_scaling
class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, softmaxes, labels):
        softmaxes = torch.Tensor(softmaxes)
        labels = torch.LongTensor(labels)

        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=softmaxes.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


# Expected Calibration Error (ECE) numpy implementation
def ECECal(probs, labels: List[int], bins: int = 15, sum=True):
    conf = np.max(probs, axis=-1)
    acc = np.argmax(probs, axis=-1) == labels

    bin_upper_bounds = np.linspace(0, 1, bins + 1)[1:]
    # split_idx = np.searchsorted(bin_upper_bounds, conf, 'left')
    split_idx = np.digitize(conf, bin_upper_bounds, right=True)
    data_len = len(split_idx)

    ece = np.zeros(bins)
    for i in range(bins):
        idx = split_idx == i
        if np.sum(idx) > 0:
            bin_avg_conf = np.mean(conf[idx])
            bin_avg_acc = np.mean(acc[idx])
            bin_prob = np.sum(idx) / data_len

            ece[i] = np.abs(bin_avg_acc - bin_avg_conf) * bin_prob

        # print(bin_avg_acc, bin_avg_conf, bin_prob, ece[i])

    return ece.sum()


# Reliability Values numpy implementation
def ECEAccCal(probs, labels, bins: int = 15):
    conf = np.max(probs, axis=-1)
    acc = np.argmax(probs, axis=-1) == labels

    bin_upper_bounds = np.linspace(0, 1, bins + 1)[1:]
    # split_idx = np.searchsorted(bin_upper_bounds, conf, 'left')
    split_idx = np.digitize(conf, bin_upper_bounds, right=True)
    data_len = len(labels)

    bin_acc = np.zeros(bins)
    bin_prob = np.zeros(bins)
    for i in range(bins):
        idx = split_idx == i
        if np.sum(idx) > 0:
            bin_avg_acc = np.mean(acc[idx])
            bin_acc[i] = bin_avg_acc
            bin_prob[i] = np.sum(idx) / data_len

        # print(bin_avg_acc, bin_avg_conf, bin_prob, ece[i])

    return bin_acc, bin_prob


def BierCal(probs, labels: List[int]):
    probs_correct = np.array([x[i] for x, i in zip(probs, labels)])
    return np.mean(np.power(probs_correct - 1, 2))


def EntropyCal(prob):
    result = np.mean(entropy(prob, axis=-1))
    return result


def SCECal(probs, labels: List[int], cls_num, bins: int = 15):
    cls = np.arange(cls_num)
    conf_all = np.max(probs, axis=-1)
    acc_all = np.argmax(probs, axis=-1) == labels

    conf_group, acc_group = group_data((conf_all, acc_all), labels, cls)

    eces = []
    for conf, acc in zip(conf_group, acc_group):
        conf = np.array(conf)
        acc = np.array(acc)
        bin_upper_bounds = np.linspace(0, 1, bins + 1)[1:]
        # split_idx = np.searchsorted(bin_upper_bounds, conf, 'left')
        split_idx = np.digitize(conf, bin_upper_bounds, right=True)
        data_len = len(split_idx)

        ece = np.zeros(bins)
        for i in range(bins):
            idx = split_idx == i
            if np.sum(idx) > 0:
                bin_avg_conf = np.mean(conf[idx])
                bin_avg_acc = np.mean(acc[idx])
                bin_prob = np.sum(idx) / data_len

                ece[i] = np.abs(bin_avg_acc - bin_avg_conf) * bin_prob

            # print(bin_avg_acc, bin_avg_conf, bin_prob, ece[i])

        eces.append(ece.sum())

    return np.mean(eces)


def group_data(data: tuple, label: List[int], cls: List[int]):
    # the idx should be output of np.unique(data), which is sorted.
    assert len(set(label)) == len(cls)
    tuple_num = len(data)
    data_group = [[[] for _ in cls] for _ in range(tuple_num)]
    for i, l in enumerate(label):
        for j in range(tuple_num):
            data_group[j][l].append(data[j][i])

    return data_group


# from https://github.com/zhmiao/OpenLongTailRecognition-OLTR
def F_measure(preds, labels, openset=False, theta=None):
    from sklearn.metrics import f1_score
    if openset:
        # f1 score for openset evaluation
        true_pos = 0.
        false_pos = 0.
        false_neg = 0.

        for i in range(len(labels)):
            true_pos += 1 if preds[i] == labels[i] and labels[i] != -1 else 0
            false_pos += 1 if preds[i] != labels[i] and labels[i] != -1 else 0
            false_neg += 1 if preds[i] != labels[i] and labels[i] == -1 else 0

        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        return 2 * ((precision * recall) / (precision + recall + 1e-12))
    else:
        # Regular f1 score
        return f1_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro')
