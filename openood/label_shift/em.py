import numpy as np
from typing import Union, Tuple


# import logging


def owls_em(fx: np.ndarray,
            hx: np.ndarray,
            c: np.ndarray,
            rho_s: float,
            fx_s: np.ndarray = None,
            hx_s: np.ndarray = None,
            alpha_i: Tuple[np.ndarray, float] = None,
            alpha_o: Tuple[np.ndarray, float] = None,
            max_iter: int = 100,
            estimate: str = 'MLE',
            init_mode: str = 'identical',
            dvg_name='kl'):
    r"""
    Implementation of Open World Label Shift algorithm,
    for unknown target label distribution estimation.

    Method:
        * Given         source domain p_s(y|x) = f(x), p_s(y) = c,
                                  p_s(i|x) = h(x), p_s(i) = rho_s;
                        target domain D = {x_i}^N_{i=1};
        * Assume:       p_s(x|y=i) = p_t(x|y=i) for i\in{1,2,...,K,K+1}
        * Estimate:     target domain p_t(y) = pi, p_t(i) = rho_t
        
    Args:
        fx:             Close Set classification softmax output for all the target domain data (D)
        hx:             OOD classification [0,1] output for all the target domain data (D)
        c:              p_s(y) Source label distribution on seen classes
        rho_s:          p_s(i) Estimated source ID ratio given h(x)
        fx_s:           Source domain data prediction (used when compute MAP)
        hx_s:           Source domain ood prediction (used when compute MAP)
        alpha_i:        Dirichlet prior parameter for target ID label distribution p_t(y)
        alpha_o:        Beta prior parameter for target ID/OOD ratio p_t(i), p_t(o)
        max_iter:       Maximum iteration of EM algorithm
        estimate:       Whether to use "MLE" or "MAP" estimate
        init_mode:      Initialization method for pi and rho_t
        dvg_name:       Divergence used to determine prior automatically.

    Shapes:
        * Input:
            fx:         N x K       K-dimensional softmax predictions for N samples
            hx:         N           OOD classification results for N samples
            c:          K           K-dimensional source domain ID class label distribution
            rho_s:      1           (Pseudo) ratio of ID data on source domain
            alpha_i     (K, 1)      Prior ratio and concentration (scale) for p_t(y) = pi
            alpha_o     (2, 1)      Prior ratio and concentration (scale) for p_t(i), p_t(o)
        * Output:
            pi          K           Estimated p_t(y) = pi       (ID label distribution)
            rho_t       1           Estimated p_t(i) = rho_t    (ID / (ID + OOD) data ratio)

    References

    """
    # Sanity Check
    cls_num = len(c)
    assert fx.shape[-1] == cls_num
    if type(max_iter) != int or max_iter < 0:
        raise Exception('max_iter should be a positive integer, not ' + str(max_iter))

    # Construct fx_tilde and c_tilde given data
    c_tilde = np.concatenate((rho_s * c, [1 - rho_s]))
    fx_tilde = np.concatenate((fx * hx[..., None], 1 - hx[..., None]), axis=-1)

    # Setup d(p,q) measure
    if dvg_name == 'kl':
        dvg = kl_div
    elif dvg_name == 'js':
        dvg = js_div
    else:
        raise Exception('Unsupported distribution distance measure, expect kl or js.')

    # Set setup prior parameter for MLE or MAP estimate
    if estimate == "MLE":
        print("Estimation model is MLE, prior alpha_i, alpha_o are ignored.")
        alpha_i, alpha_o = None, None
    elif estimate == "MAP":
        print("Estimation model is MAP, prior alpha_i, alpha_o are either given or "
              "learned with divergence type {}.".format(dvg_name))
        fx_s_tilde = np.concatenate((fx_s * hx_s[..., None], 1 - hx_s[..., None]), axis=-1)
        alpha_o = (np.ones(2), 1) if alpha_o is None else alpha_o
        alpha_i = get_id_prior(fx_s_tilde, c_tilde, dvg=dvg, max_iter=max_iter) if alpha_i is None else alpha_i
        # print('Prior parameters alpha_i, alpha_o are: {}, {}'.format(str(alpha_i), str(alpha_o)))

    # EM Algorithm Computation
    pi, rho_t = o_em(fx_tilde, c_tilde, alpha_i, alpha_o, init_mode=init_mode, max_iter=max_iter)

    # print('MLE/MAP pi: {}, rho_t {:.2f}'.format(str(pi), rho_t))

    return pi, rho_t


def o_em(fx_tilde: np.ndarray,
         c_tilde: np.ndarray,
         alpha_i: Tuple[np.ndarray, float] = None,
         alpha_o: Tuple[np.ndarray, float] = None,
         init_mode: str = 'uniform',
         max_iter: int = 100):
    r"""
    The Open World Label Shift EM algorithm implementation.

    Args:
        fx_tilde:       N x (K+1)   Softmax Probability [hx x fx_1, hx x fx2, ..., hx x fxK, 1 - hx]^T
        c_tilde:        K+1         Source label distribution [rho_s x c1, rho_s x c2, ..., rho_s x cK, 1 - rho_s]^T
        alpha_i:        (K, 1)      pi prior with K dim ratio and scalar scale: (ratio, scale)
        alpha_o:        (2, 1)      rho_t prior with 2 dim ratio and scalar scale: (ratio, scale)
        init_mode:      str         method to initialize pi, rho_t, support: "identical", "uniform"
        max_iter:       int         maximum iteration of EM algorithm

    Returns:
        pi:             K           Estimated p_t(y) = pi
        rho_t:          1           Estimated p_t(i) = rho_t

    """
    # Sanity check
    assert len(fx_tilde.shape) == 2 and fx_tilde.shape[-1] == len(c_tilde)
    sample_num, cls_num = fx_tilde.shape
    cls_num -= 1

    # Switch to MLE if no prior alpha_i or alpha_o is given.
    ratio_i, lam_i = (np.ones(cls_num) / cls_num, 1) if alpha_i is None else alpha_i
    ratio_o, lam_o = (np.ones(2) / 2, 1) if alpha_o is None else alpha_o
    assert len(ratio_o) == 2 and len(ratio_i) == cls_num

    # Normalize Source Label Distribution c
    c_tilde = np.array(c_tilde) / np.sum(c_tilde)
    # Initialize Target Label Distribution pi, rho_t
    if init_mode == 'uniform':
        pi = np.ones(cls_num) / cls_num
        rho_t = 0.5
    elif init_mode == 'identical':
        pi = c_tilde.copy()[:-1]
        pi /= pi.sum()
        rho_t = 1 - c_tilde[-1]
    else:
        raise ValueError('init_mode should be either "uniform" or "identical"')

    # Initialize weight
    pi_tilde = np.concatenate((pi * rho_t, [1 - rho_t]))
    w = pi_tilde / np.array(c_tilde)

    # logging
    pi_history = [pi_tilde[:-1] / np.sum(pi_tilde[:-1])]
    rho_t_history = [1 - pi_tilde[-1]]

    # EM algorithm with MAP estimation----------------------------------------#
    for i in range(max_iter):
        # print('w shape ', w.shape)
        # E-Step--------------------------------------------------------------#
        g = fx_tilde * w
        g /= np.sum(g, axis=-1, keepdims=True)

        # M-Step--------------------------------------------------------------#
        pi_tilde = np.mean(g, axis=0)
        # print(m.shape, alpha_i.shape, sample_num)

        # Update pi, rho_t with data and, if applicable, prior parameters.
        pi = lam_i * pi_tilde[:-1] / (1 - pi_tilde[-1]) + (1 - lam_i) * ratio_i
        rho_t = lam_o * (1 - pi_tilde[-1]) + (1 - lam_o) * ratio_o[0]
        pi_tilde = np.concatenate((pi * rho_t, [1 - rho_t]))

        w = pi_tilde / c_tilde

        pi_history.append(pi)
        rho_t_history.append(rho_t)

    # print('pi estimate history {}'.format(str(pi_history)))
    # print(('rho_t estimate history [' + ', '.join(['%.2f']*len(rho_t_history)) + ']') % tuple(rho_t_history))

    return pi, rho_t


def mapls(fx: np.ndarray,
          c: np.ndarray,
          alpha: Tuple[np.ndarray, float] = None,
          init_mode: str = 'uniform',
          max_iter: int = 100):
    r"""
    The Close World Label Shift EM (MAPLS) algorithm implementation.

    Args:
        fx:             N x K       Softmax Probability [hx x fx_1, hx x fx2, ..., hx x fxK, 1 - hx]^T
        c:              K           Source label distribution [rho_s x c1, rho_s x c2, ..., rho_s x cK, 1 - rho_s]^T
        alpha:          (K, 1)      pi prior with K dim ratio and scalar scale: (ratio, scale)
        init_mode:      str         method to initialize pi, rho_t, support: "identical", "uniform"
        max_iter:       int         maximum iteration of EM algorithm

    Returns:
        pi:             K           Estimated p_t(y) = pi

    """
    # Sanity check
    assert len(fx.shape) == 2 and fx.shape[-1] == len(c)
    sample_num, cls_num = fx.shape

    # Switch to MLE if no prior alpha_i or alpha_o is given.
    ratio, lam = (np.ones(cls_num) / cls_num, 1) if alpha is None else alpha
    assert len(ratio) == cls_num

    # Normalize Source Label Distribution c
    c = np.array(c) / np.sum(c)
    # Initialize Target Label Distribution pi, rho_t
    if init_mode == 'uniform':
        pi = np.ones(cls_num) / cls_num
    elif init_mode == 'identical':
        pi = c.copy()[:-1]
        pi /= pi.sum()
    else:
        raise ValueError('init_mode should be either "uniform" or "identical"')

    # Initialize weight
    w = pi / np.array(c)

    # logging
    pi_history = [pi]

    # EM algorithm with MAP estimation----------------------------------------#
    for i in range(max_iter):
        # print('w shape ', w.shape)
        # E-Step--------------------------------------------------------------#
        g = fx * w
        g /= np.sum(g, axis=-1, keepdims=True)

        # M-Step--------------------------------------------------------------#
        pi = np.mean(g, axis=0)

        # Update pi, rho_t with data and, if applicable, prior parameters.
        pi = lam * pi + (1 - lam) * ratio
        w = pi / c

        pi_history.append(pi)
    # print('pi estimate history {}'.format(str(pi_history)))

    return pi


def get_id_prior(fx_s_tilde: np.ndarray,
                 c_tilde: np.ndarray,
                 dvg, max_iter=100):
    r"""
    Adaptive prior learning model for ID classes

    Args:
        fx_s_tilde:     Source Data ID + OOD prediction
        c_tilde:        Source ID + OOD label distribution
        dvg:            Divergence type
        max_iter:       Max iteration of EM algorithm

    Returns:
        ratio:          alpha / sum(alpha)
        lam:            weight of data contribution in the M-step in EM
    """
    N, K = fx_s_tilde.shape
    K = K - 1
    c = c_tilde[:-1] / c_tilde[:-1].sum()
    u = np.ones(K) / K

    # MLLS estimation of source and target domain label distribution
    pi, rho_t = o_em(fx_s_tilde, c_tilde, max_iter=max_iter)
    # print('MLE pi: {}, rho_t {:.2f}'.format(str(pi), rho_t))

    d_tu = dvg(pi, u) + 1e-6
    d_ts = dvg(pi, c) + 1e-6
    d_su = dvg(c, u) + 1e-6
    print('weights are, TU_div %.4f, TS_div %.4f, SU_div %.4f' % (d_tu, d_ts, d_su))

    lam = 1 - lam_forward(d_su, lam_inv(dpq=0.5, lam=0.2))

    w = (1 / d_tu) / (1 / d_tu + 1 / d_ts)
    # w = (d_tu * d_ts) / ((d_tu + d_ts) * d_tu)
    # print('Uniform prior weight: %.4f, Source prior weight: %.4f' % (w, 1 - w))
    # ratio = u * w + c * (1 - w)
    ratio = u
    # alpha_o = (N / lam - N + K) * ratio

    return ratio, np.clip(lam, 0, 1)


def get_ood_prior(fx_tilde, c_tilde):
    ratio, lam = 1, 1
    return ratio, lam


def lam_inv(dpq, lam):
    return (1 / (1 - lam) - 1) / dpq


def lam_forward(dpq, gamma):
    return gamma * dpq / (1 + gamma * dpq)


def kl_div(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q + 1e-8, dtype=float)

    return np.sum(np.where(p > 0, p * np.log(p / q), 0))


def js_div(p, q):
    assert (np.abs(np.sum(p) - 1) < 1e-6) and (np.abs(np.sum(q) - 1) < 1e-6)
    m = (p + q) / 2
    return kl_div(p, m) / 2 + kl_div(q, m) / 2
