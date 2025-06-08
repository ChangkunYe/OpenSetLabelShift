import numpy as np
from .em import owls_em
from .metrics import get_ls_metrics


def owls_estimation(fx: np.ndarray, hx: np.ndarray, c: np.ndarray, rho_s: float,
                    fx_s: np.ndarray = None, hx_s: np.ndarray = None,
                    mode: str = 'MLE'):
    r"""
    Open World Label Shift Estimation (function implementation)
    Args:
        fx:         Test Data (target domain) ID classifier Softmax output
        hx:         Test Data (target domain) OOD classifier Binary output
        c:          Source ID label distribution
        rho_s:      Source ID data ratio (estimated by ls_retrieval())
        fx_s:       [Optional for MAP estimate] Train Data (source domain) ID classifier Softmax output
        hx_s:       [Optional for MAP estimate] Train Data (source domain) OOD classifier binary output
        mode:       "MLE" or "MAP" estimate.

    Returns:
        pi:         Target ID label distribution p_t(y=i|ID) = pi_i
        rho_t:      Target ID data ratio p_t(ID) = rho_t

    """
    pi, rho_t = owls_em(fx, hx, c, rho_s, fx_s=fx_s, hx_s=hx_s, estimate=mode)

    return pi, rho_t


def owls_evaluation(fx: np.ndarray, hx: np.ndarray, labels: np.ndarray,
                    c: np.ndarray, rho_s: float,
                    pi: np.ndarray, rho_t: np.ndarray):
    r"""
    Open World Label Shift model evaluation (function implementation)
    Args:
        fx:         Test Data (target domain) ID classifier Softmax output
        hx:         Test Data (target domain) OOD classifier Binary output
        labels:     Test Data (target domain) ground truth labels (ID [0,1,...,K-1] and OOD [K])
        c:          Source ID label distribution
        rho_s:      Source ID data ratio (estimated by ls_retrieval())
        pi:         Target ID label distribution p_t(y=i|ID) = pi_i
        rho_t:      Target ID data ratio p_t(ID) = rho_t

    Returns:
        metrics:    All metrics as dict

    """
    fx_tilde = np.concatenate((fx * hx[None, ...], (1 - hx)[None, ...]), axis=0)
    c_tilde = np.concatenate((c * rho_s, [1 - rho_s]))
    pi_tilde = np.concatenate((pi * rho_t, [1 - rho_t]))

    _, w_gt = np.unique(labels, return_counts=True)
    w_gt /= w_gt.sum()

    w = pi_tilde / c_tilde
    metrics = get_ls_metrics(fx_tilde, labels, w, pi, w_gt)
    return metrics


def owls_correction(fx_tilde: np.ndarray, pi_tilde: np.ndarray, c_tilde: np.ndarray):
    r"""
    Open Set Label Shift correction (function implementation)
    Args:
        fx_tilde:       Test Data (target domain) output [h(x)*f(x)_1, h(x)*f(x)_2,...,h(x)*f(x)_K,1-h(x)]
        pi_tilde:       Test Data (target domain) label distribution (include OOD class)
        c_tilde:        source domain label distribution (include OOD class)
    Returns:
        probs:          label shift corrected softmax prediction for K+1 classes (K ID + 1 OOD)
    """
    probs = fx_tilde * pi_tilde / c_tilde
    probs /= probs.sum(-1, keepdim=True)

    return probs


# Get rho_s
def ls_retrieval(hx_s_id, hx_refer_ood):
    r"""
    Source ID data ratio rho_s retrieval model
    Args:
        hx_s_id:        OOD classifier h(x) prediction on source ID dataset
        hx_refer_ood:   OOD classifier h(x) prediction on source (reference) OOD dataset
    Returns:
        rho_s:          Source ID data ratio

    """
    sigma_0 = hx_refer_ood.mean()
    # sigma_0 = np.clip((sigma_0 + 0.9), 0, 0.95)
    sigma_1 = hx_s_id.mean()
    print("sigma 0: {:.2f}, sigma 1: {:.2f}".format(sigma_0, sigma_1))
    return sigma_0 / (1 - sigma_1 + sigma_0)


# Get corrected rho_t
def rho_t_correction(rho_t: float, tpr_s: float, fpr_s: float = None):
    r"""
    Target ID data ratio calibration
    Args:
        rho_t:      Original prediction of rho_t with imperfect classifier h(x) != p_s(b=1|x)
        tpr_s:      Average confidence of h(x) on source ID dataset
        fpr_s:      Average confidence of h(x) on source (reference) OOD dataset

    Returns:
        rho_t:      Calibrated p_t(ID) = rho_t
    """
    # fpr_s = tpr_s / 2 if fpr_s is None else fpr_s

    if abs(tpr_s - fpr_s) <= 1e-3:
        # raise ValueError('TPR_s %.2f and FPR_s %.2f too close, method unstable.' % (tpr_s, fpr_s))
        print('[warning] TPR_s %.2f and FPR_s %.2f too close, method unstable.' % (tpr_s, fpr_s))

    result = (rho_t - fpr_s) / (tpr_s - fpr_s)
    if fpr_s > tpr_s:
        print("[warning] TPR < FPR, which suggests the OOD classifier need adjustment: "
              "change decision boundary.")
    return np.clip(result, 0, 1)



