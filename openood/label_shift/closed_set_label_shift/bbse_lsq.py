import numpy as np
from typing import List
from openood.label_shift.closed_set_label_shift.common import get_confusion_matrix, get_marginal
import cvxpy as cp


# Estimation of Target Label Distribution P(Y_t=i) via BBSE-------------------#
def bbse(train_probs: np.ndarray,
         train_labels: List,
         test_probs: np.ndarray,
         cls_num: int,
         py_mode: str = 'soft',
         qy_mode: str = 'soft',
         solver: str = 'cvxpy'):
    r"""
    Implementation of Black Box Label Shift estimator (BBSE) with convex optimization support.

    Given source domain predicted p(z=i|x) = f(x), source domain ground truth P(Y_s=i).
    Given target domain predicted q(z=i|x) = f(x).
    Solve q(z) = \sum_y p(z,y) q(y)/p(y) for q(y)/p(y)
    Estimate target domain P(Y_t=i), returns w = P(Y_t=i)/P(Y_s=i)

    Args:
        train_probs:    Predicted probability of train set samples, which come from source domain.
        train_labels:   Ground truth labels of train set samples, follows source domain label distribution.
        test_probs:     Predicted probability of test set or validation set samples, which come from target domain.
        cls_num:        Total number of classes in the classification dataset.
        py_mode:        Mode for estimating source domain confusion matrix .etc, either 'soft' or 'hard'.
        qy_mode:        Mode for estimating target domain predicted label distribution, either 'soft' or 'hard'.
        solver:         Solver for the linear system, either "npy" or "cvxpy".
        result_type:    Support value in ['w', 'pi'], where 'w': P(Y_t=.) / P(Y_s=.) or 'pi': P(Y_t=.)

    Shape:
        * Input:
            train_probs:    N x C   (No. of samples in source domain set) x (No. of classes),
            train_labels:   N       (No. of samples in source domain set),
            test_probs:     M x C   (No. of samples in target domain set) x (No. of classes),
            cls_num:        1       (No. of classes C)

        * Output:
            w/pi:           C       Estimated w = P(Y_t=.) / P(Y_s=.) or P(Y_t=.)=\pi depend on "result_type"

    Reference:
        * Original paper:
        [ICML 2018] Detecting and Correcting for Label Shift with Black Box Predictors
        < http://proceedings.mlr.press/v80/lipton18a/lipton18a.pdf >

        * Official Code:
            < https://github.com/zackchase/label_shift >
        * Unofficial Code:
            < https://github.com/flaviovdf/label-shift >
    """

    assert (train_probs.shape[-1] == cls_num) and (test_probs.shape[-1] == cls_num)

    # Get Confusion Matrix
    pzy = get_confusion_matrix(train_probs, train_labels, cls_num, mode=py_mode)

    # Devided by marginal if estimate pi instead of w
    pzy = pzy.T / len(train_labels)

    qz = get_marginal(test_probs, cls_num, mode=qy_mode)

    # Solve the linear system
    if solver == 'npy':
        result = numpy_solve(pzy, qz)
    elif solver == 'cvxpy':
        # Obtain source label distribution
        pz = np.array(list(set(train_labels)))
        pz = pz / pz.sum()
        result = cvx_solve(pzy, qz, result_type="w", c=pz)
    else:
        raise ValueError('solver have to be either "npy" or "cvxpy".')

    # Clip and normalize the result.
    if np.sum(result < 0) > 0:
        print('[warning] Negative value exist in BBSE estimation of w, will be clip to 0')
        result = np.clip(result, 0, None)

    return result


# Just for consistency check, from BBSE unofficial Code implementation: https://github.com/flaviovdf/label-shift
def calculate_marginal(y, n_classes):
    mu = np.zeros(shape=(n_classes, 1))
    for i in range(n_classes):
        mu[i] = np.sum(y == i)
    return mu / y.shape[0]


def estimate_labelshift_ratio(y_true_val, y_pred_val, y_pred_trn, n_classes):
    from sklearn.metrics import confusion_matrix
    labels = np.arange(n_classes)
    C = confusion_matrix(y_true_val, y_pred_val, labels=labels).T
    C = C / y_true_val.shape[0]

    mu_t = calculate_marginal(y_pred_trn, n_classes)
    lamb = 1.0 / min(y_pred_val.shape[0], y_pred_trn.shape[0])

    I = np.eye(n_classes)
    wt = np.linalg.solve(np.matmul(C.T, C) + lamb * I, np.matmul(C.T, mu_t))
    return wt


def estimate_target_dist(wt, y_true_val, n_classes):
    mu_t = calculate_marginal(y_true_val, n_classes)
    return wt * mu_t


def numpy_solve(matrix, bias):
    r"""
    Solve Linear System with Numpy linalg package.
    """
    try:
        # Solve the Ax=b if A (pzy) is not singular
        x = np.linalg.solve(np.matmul(matrix.T, matrix), np.matmul(matrix.T, bias))
        # logging.info('Matrix is pseudo invertible, solved with np.linalg.solve.')
    except np.linalg.LinAlgError:
        # Go with least square solve if A (pzy) is singular
        print('Matrix is singular, solved with np.linalg.lstsq.')
        x = np.linalg.lstsq(np.matmul(matrix.T, matrix), np.matmul(matrix.T, bias), rcond=None)[0]
        print('Least Square Solver result: %s' % str(x))

    return x


def cvx_solve(matrix, bias, result_type: str = 'w', weighted: bool = False, c: np.ndarray = None):
    r"""
    Solve linear System with Cvxpy package, constraints: x >= 0 and np.sum(x) == 1
    """
    n = len(bias)
    assert matrix.shape == (n, n)

    if weighted:
        matrix = np.array([x / bias for x in matrix])
        bias = np.ones_like(bias)

    x = cp.Variable(n)

    if result_type == 'qy':
        r = np.ones(n)
        constraints = [x >= 0, r.T @ x == 1]
    elif result_type == 'w':
        assert c is not None
        constraints = [x >= 0, c.T @ x == 1]

    cost = cp.sum_squares(matrix @ x - bias)

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP, verbose=False)
    # prob.solve(solver=cp.SCS, use_indirect=False)

    return x.value