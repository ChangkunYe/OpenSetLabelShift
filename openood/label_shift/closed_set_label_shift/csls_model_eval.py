import numpy as np
import time
from typing import List
from openood.label_shift.closed_set_label_shift.common import lsc, get_py
from openood.label_shift.closed_set_label_shift.mlls import mlls
from openood.label_shift.closed_set_label_shift.bbse_lsq import bbse
from openood.label_shift.closed_set_label_shift.rlls import rlls
from openood.label_shift.closed_set_label_shift.mapls import mapls
from openood.label_shift.metrics import acc_cal
import logging


def ls_metrics_eval(probs, labels,
                    train_probs, train_labels,
                    train_cls_num_list, ensemble_num=None, py_all=None):
    r"""
    Evaluating performance metrics for Label Shift Methods.
    (Note that "labels" and "val_labels" only used as ground truth for evaluation)
    """
    np.random.seed()
    max_iter = 100
    if ensemble_num is None:
        def f(x):
            return x
    else:
        def f(x):
            return x.mean(-2)

    def get_mse(a, b):
        return np.mean(np.power(a - b, 2))

    cls_num = len(train_cls_num_list)
    qy_gt = np.zeros(cls_num)
    for i in labels:
        if i < cls_num:
            qy_gt[i] += 1

    qy_gt = qy_gt / qy_gt.sum()
    if py_all is None:
        py_all = {'gt': get_py(f(train_probs), train_cls_num_list, mode="gt"),
                  'soft': get_py(f(train_probs), train_cls_num_list, mode="soft"),
                  'hard': get_py(f(train_probs), train_cls_num_list, mode="hard")}
    w_gt = qy_gt / py_all['gt']

    metrics = {}

    # -----------------Maximum Likelihood Label Shift Accuracy-------------------#
    print('#-----mlls Softmax Metric------#')
    for mode in ['gt', 'soft']:
        time_start = time.time()
        w, qy = get_label_shift_ratio(f(train_probs), train_labels, f(probs), train_cls_num_list,
                                      py=py_all[mode], qy_method='mlls', max_iter=max_iter)
        mlls_probs = lsc(f(probs), w)

        print('#Mode: %s' % (mode))
        acc = acc_cal(mlls_probs, labels)
        mlls_metric = {'acc': acc}
        del mlls_probs
        w_mse = get_mse(w, w_gt)
        print('w MSE: %.8f, acc: %.2f' % (w_mse, acc))

        mlls_metric['w_mse'] = w_mse
        mlls_metric['w'] = w
        mlls_metric['qy'] = qy

        metrics['mlls_' + mode] = mlls_metric

        print('Evaluation time: %.4f' % (time.time() - time_start))

    # -----------------Black-Box Label Shift Accuracy----------------------------#
    print('#-----bbse Softmax Metric------#')
    for mode in ['hard', 'soft']:
        time_start = time.time()
        w, qy = get_label_shift_ratio(f(train_probs), train_labels, f(probs), train_cls_num_list,
                                      py=py_all[mode], py_mode=mode, qy_mode='soft', qy_method="bbse")
        bbse_probs = lsc(f(probs), w)

        print('#Mode: %s' % (mode))
        acc = acc_cal(bbse_probs, labels)
        bbse_metric = {'acc': acc}
        del bbse_probs

        w_mse = get_mse(w, w_gt)
        print('w MSE: %.8f, acc: %.2f' % (w_mse, acc))

        bbse_metric['w_mse'] = w_mse
        bbse_metric['w'] = w
        bbse_metric['qy'] = qy

        metrics['bbse_' + mode] = bbse_metric

        print('Evaluation time: %.4f' % (time.time() - time_start))

    # -----------Regularized Learning Label Shift Accuracy-----------------------#
    print('#-----rlls Softmax Metric------#')
    for mode in ['hard', 'soft']:
        time_start = time.time()
        w, qy = get_label_shift_ratio(f(train_probs), train_labels, f(probs), train_cls_num_list,
                                      py=py_all[mode], py_mode=mode, qy_mode='soft', qy_method="rlls")
        rlls_probs = lsc(f(probs), w)

        print('#Mode: %s' % (mode))
        acc = acc_cal(rlls_probs, labels)
        rlls_metric = {'acc': acc}
        del rlls_probs

        w_mse = get_mse(w, w_gt)
        print('w MSE: %.8f, acc: %.2f' % (w_mse, acc))

        rlls_metric['w_mse'] = w_mse
        rlls_metric['w'] = w
        rlls_metric['qy'] = qy

        metrics['rlls_' + mode] = rlls_metric

        print('Evaluation time: %.4f' % (time.time() - time_start))


    # -----------Long-Tail Label Shift Accuracy-----------------------#
    print('#-----mapls Softmax Metric------#')
    # for lam in [1.0, 0.9, 0.7, 0.5, 0.3, 0.1, None]:
    for mode in ['gt', 'soft']:
        time_start = time.time()
        w, qy = get_label_shift_ratio(f(train_probs), train_labels, f(probs), train_cls_num_list,
                                      py=py_all[mode], qy_mode='soft', qy_method="mapls",
                                      max_iter=max_iter, lam=None)
        mapls_probs = lsc(f(probs), w)

        # print('lambda is: %.2f' % lam)
        acc = acc_cal(mapls_probs, labels)
        mapls_metric = {'acc': acc}
        del mapls_probs

        w_mse = get_mse(w, w_gt)
        print('w MSE: %.8f, acc: %.2f' % (w_mse, acc))

        mapls_metric['w_mse'] = w_mse
        mapls_metric['w'] = w
        mapls_metric['qy'] = qy

        # m_name = 'None' if lam is None else "%.2f" % lam
        metrics['mapls_' + mode] = mapls_metric

        print('Evaluation time: %.4f' % (time.time() - time_start))

    return metrics


def get_label_shift_ratio(train_probs, train_labels, val_probs,
                          cls_num_list, py: np.ndarray = None,
                          val_cls_num_list: List = None,
                          py_mode: str = "soft", qy_mode: str = 'soft',
                          qy_method: str = 'known', max_iter: int = 100,
                          lam: int = None):
    valid_methods = ['known', 'uniform', 'mlls', 'bbse', 'rlls', 'mapls', 'mapls2']
    assert qy_method in valid_methods
    cls_num = len(cls_num_list)
    if py is None:
        py = cls_num_list / np.sum(cls_num_list)

    if qy_method == 'known':
        assert val_cls_num_list is not None
        qy = get_py(val_probs, val_cls_num_list, mode=qy_mode)
        w = np.array(qy) / np.array(py)
    if qy_method == 'uniform':
        qy = np.ones(cls_num) / cls_num
        w = np.array(qy) / np.array(py)
    elif qy_method == 'mlls':
        qy = mlls(val_probs, py, max_iter=max_iter)
        w = np.array(qy) / np.array(py)
    elif qy_method == 'bbse':
        w = bbse(train_probs, train_labels, val_probs, cls_num, py_mode=py_mode, qy_mode=qy_mode)
        qy = w * np.array(py) / np.sum(w * np.array(py))
    elif qy_method == 'rlls':
        w = rlls(train_probs, train_labels, val_probs, cls_num, py_mode=py_mode, qy_mode=qy_mode)
        qy = w * np.array(py) / np.sum(w * np.array(py))
    elif qy_method == 'mapls':
        qy = mapls(train_probs, val_probs, py, qy_mode, max_iter=max_iter, lam=lam)
        w = np.array(qy) / np.array(py)

    # print(np.shape(py), np.shape(qy))
    return w, qy
