import numpy as np
import torch
from time import time
from tqdm import tqdm
from typing import Union, List, Any, Dict
from .em import owls_em
from .metrics import get_ls_metrics, get_acc_metrics, acc_cal, get_mse
from .logistic_regression import train_logistic_regression
from .models import ls_retrieval, rho_t_correction
from openood.label_shift.closed_set_label_shift.csls_model_eval import ls_metrics_eval as leval


class OpenSetLabelShift:
    def __init__(self,
                 f: torch.nn.Module,
                 h: Union[torch.nn.Module, Any],
                 s_id_dataloader: torch.utils.data.DataLoader,
                 s_ood_dataloader: torch.utils.data.DataLoader,
                 train_cls_num_list: Union[np.ndarray, torch.Tensor],
                 progress: bool = False,
                 rescale: str = 'auto',
                 T: float = 2):
        r"""
        The Open World Label Shift model for label shift estimation and correction in
        the 'train K ID classes --> test K ID classes + 1 OOD class' setting.

        Args:
            f:                      ID Multi-class classifier
            h:                      OOD Binary Classifier (based on f)
            s_id_dataloader:        Source Validation DataLoader
            s_ood_dataloader:       Source Pseudo OOD DataLoader
            train_cls_num_list:     Source ID label distribution
            progress:               If enabling tqdm
            rescale:                Rescale h in [0,1] via "threshold", "logistic" or "auto"
            T:                      parameter to rescale \hat{sigma}_0
        """

        # self.baseline_id_metrics, self.baseline_all_metrics = None, None
        # self.all_metrics, self.id_metrics = None, None

        self.cls_num = len(train_cls_num_list)
        # Source ID label distribution
        self.train_cls_num_list = train_cls_num_list
        self.c = torch.Tensor(train_cls_num_list / train_cls_num_list.sum())
        self.T = T
        self.diagnostics = {'source': {}, 'target': {}}
        self._raw_score_warning_emitted = False

        if (rescale == "logistic") or (rescale == "auto" and self.cls_num > 100):
            self.rescale_logistic = True
            self.rescale_mode = 'logistic'
        elif (rescale == 'threshold') or (rescale == "auto" and self.cls_num <= 100):
            self.rescale_logistic = False
            self.rescale_mode = 'threshold'
        else:
            raise ValueError("[warning] rescale should be either 'threshold', 'logistic' or 'auto'.")
        print('Using {} rescale for {} classes.'.format(self.rescale_mode, self.cls_num))


        # Inference Source ID data and Source pseudo OOD data
        print(f'Performing inference on val and fake ood test set...')
        if f:
            self._source_inference('id', f, s_id_dataloader, progress=progress)
            self._source_inference('ood', f, s_ood_dataloader, progress=progress)
            self.fx_s = torch.cat((self.source_id_probs, self.source_ood_probs), axis=0)
            self.labels_s = torch.cat((self.source_id_labels, self.source_ood_labels))
        else:
            self.fx_s = None

        if h:
            ref_id_pred, hx_s_id, ref_id_gt = h(f, s_id_dataloader, progress)
            fake_ood_pred, hx_s_ood, fake_ood_gt = h(f, s_ood_dataloader, progress)
            raw_hx_s_id = torch.Tensor(hx_s_id)
            raw_hx_s_ood = torch.Tensor(hx_s_ood)
            self._warn_raw_ood_score_usage('source', raw_hx_s_id, raw_hx_s_ood)
            self._record_score_diagnostics('source', 'raw_postprocessor', raw_hx_s_id, raw_hx_s_ood)

            hx_s_id, hx_s_ood = raw_hx_s_id + 1, raw_hx_s_ood + 1
            self._record_score_diagnostics(
                'source',
                'shifted_for_owls',
                hx_s_id,
                hx_s_ood,
                self.source_id_labels,
            )
            self.hx_s = torch.cat((hx_s_id, hx_s_ood))
        else:
            raise ValueError("OOD classifier is required.")

        print('Length of source id conf {:g}, source id prob {:g}, ood conf {:g}, ood prob {:g}'.format(
            len(hx_s_id), len(self.source_id_probs), len(hx_s_ood), len(self.source_ood_probs)))
        print("Inference Complete.")

        # Rescale OOD Binary classifier's output of Source Domain data to [0, 1]
        print("Train ID median: {:.2f}, Train ID mean: {:.2f}, "
              "Pseudo OOD median: {:.2f}, Pseudo ood mean: {:.2f}".format(
            hx_s_id.median(), hx_s_id.mean(), hx_s_ood.median(), hx_s_ood.mean()))

        # Compute rescaling parameters (aim at better separating ID/OOD classes in h(x) space)
        # self._compute_scale_thresholding(hx_s_id, hx_s_ood)  # statistical median based approach
        # self._compute_scale_logistic(hx_s_id, hx_s_ood) # logistic regression based approach
        if self.rescale_logistic:
            self._compute_scale_logistic(hx_s_id, hx_s_ood)
        else:
            self._compute_scale_thresholding(hx_s_id, hx_s_ood)

        print('Estimation threshold is {:.2f}'.format(self.mid))
        # Source domain OOD classifier TPR, FPR evaluation
        source_rescaled_id = self._rescale(hx_s_id)
        source_rescaled_ood = self._rescale(hx_s_ood)
        self.sigma_1, self.sigma_0 = source_rescaled_id.mean(), source_rescaled_ood.mean()
        # Source ID data ratio retrival
        self.rho_s = ls_retrieval(source_rescaled_id, source_rescaled_ood / self.T)
        print("[Rescaled] Train ID median: {:.2f}, Train ID mean: {:.2f}, "
              "Pseudo OOD median: {:.2f}, Pseudo ood mean: {:.2f}".format(
            source_rescaled_id.median(), source_rescaled_id.mean(),
            source_rescaled_ood.median(), source_rescaled_ood.mean()))
        self._record_score_diagnostics(
            'source',
            'rescaled',
            source_rescaled_id,
            source_rescaled_ood,
            self.source_id_labels,
        )

        # Source ID + OOD label distribution
        self.c_tilde = torch.cat((self.c * self.rho_s, torch.Tensor([1 - self.rho_s])))
        print("Estimated c_tilde is: ", self.c_tilde)

        # Initialize estimate values
        self.pi, self.rho_t, self.pi_tilde = None, None, None
        self.pi_gt, self.rho_t_gt, self.pi_tilde_gt, self.w_gt = None, None, None, None

        # Initialize Model Predictions
        self.target_cls_probs, self.target_cls_labels = {}, {}
        # self.corrected_target_cls_probs, self.corrected_target_cls_labels = {}, {}

        print("Estimated rho_s is: {:.2f}".format(self.rho_s))

    def _rescale(self, x: torch.Tensor):
        if self.rescale_logistic:
            result = torch.sigmoid(self.rescale_w * x + self.rescale_b)
        else:
            result = (x > self.mid).to(torch.float)
        return result

    def _summary_stats(self, x: torch.Tensor) -> Dict[str, float]:
        values = x.detach().cpu().to(torch.float)
        return {
            'min': float(values.min()),
            'max': float(values.max()),
            'mean': float(values.mean()),
            'median': float(values.median()),
            'std': float(values.std(unbiased=False)) if len(values) > 1 else 0.0,
        }

    def _classwise_mean_summary(self, values: torch.Tensor,
                                labels: torch.Tensor) -> Dict[str, float]:
        values = values.detach().cpu().to(torch.float)
        labels = labels.detach().cpu().to(torch.int64)
        class_means = []
        for cls in torch.unique(labels):
            cls_idx = int(cls.item())
            if cls_idx < 0 or cls_idx >= self.cls_num:
                continue
            mask = labels == cls
            if mask.any():
                class_means.append(float(values[mask].mean()))

        if not class_means:
            return {}

        class_means = np.asarray(class_means, dtype=np.float64)
        return {
            'class_count': int(len(class_means)),
            'min': float(class_means.min()),
            'max': float(class_means.max()),
            'mean': float(class_means.mean()),
            'std': float(class_means.std()),
        }

    def _record_score_diagnostics(self,
                                  split_name: str,
                                  stage_name: str,
                                  id_scores: torch.Tensor,
                                  ood_scores: torch.Tensor,
                                  id_labels: torch.Tensor = None):
        id_stats = self._summary_stats(id_scores)
        ood_stats = self._summary_stats(ood_scores)
        stage = {
            'id': id_stats,
            'ood': ood_stats,
            'mean_gap': id_stats['mean'] - ood_stats['mean'],
        }
        if id_labels is not None:
            classwise = self._classwise_mean_summary(id_scores, id_labels)
            if classwise:
                stage['id_classwise_mean'] = classwise

        self.diagnostics.setdefault(split_name, {})[stage_name] = stage
        print('[diag][{}][{}] ID mean {:.4f}, OOD mean {:.4f}, gap {:.4f}'.format(
            split_name, stage_name, id_stats['mean'], ood_stats['mean'], stage['mean_gap']))
        if 'id_classwise_mean' in stage:
            classwise = stage['id_classwise_mean']
            print('[diag][{}][{}] ID classwise mean min {:.4f}, max {:.4f}, std {:.4f} across {} classes'.format(
                split_name,
                stage_name,
                classwise['min'],
                classwise['max'],
                classwise['std'],
                classwise['class_count']))

    def _warn_raw_ood_score_usage(self, split_name: str,
                                  id_scores: torch.Tensor,
                                  ood_scores: torch.Tensor):
        if self._raw_score_warning_emitted:
            return

        min_score = float(torch.min(torch.cat((id_scores, ood_scores))))
        max_score = float(torch.max(torch.cat((id_scores, ood_scores))))
        if min_score < 0.0 or max_score > 1.0:
            print('[warning] {} OOD score range [{:.4f}, {:.4f}] is not probability-like. '
                  'Current OWLS implementation still shifts scores with `1 + conf` before rescaling, '
                  'which differs from the paper assumption h(x) ~= p(ID|x) in [0, 1].'.format(
                      split_name, min_score, max_score))
        else:
            print('[warning] {} OOD scores are treated as h(x) via `1 + conf` before rescaling. '
                  'Even when the raw score falls in [0, 1], this code path is still a heuristic '
                  'approximation of the paper assumption.'.format(split_name))
        self._raw_score_warning_emitted = True

    def _record_rho_diagnostics(self):
        rho_info = {
            'rho_s': float(self.rho_s),
            'rho_t0': float(self.rho_t0),
            'rho_t': float(self.rho_t),
            'rho_t_gt': float(self.rho_t_gt),
            'sigma_1': float(self.sigma_1),
            'sigma_0': float(self.sigma_0),
            'sigma_0_adjusted': float(self.sigma_0 / self.T),
        }
        self.diagnostics.setdefault('target', {})['rho'] = rho_info
        print('[diag][target][rho] rho_s {:.4f}, rho_t0 {:.4f}, rho_t {:.4f}, gt {:.4f}, '
              'sigma_1 {:.4f}, sigma_0 {:.4f}, sigma_0/T {:.4f}'.format(
                  rho_info['rho_s'],
                  rho_info['rho_t0'],
                  rho_info['rho_t'],
                  rho_info['rho_t_gt'],
                  rho_info['sigma_1'],
                  rho_info['sigma_0'],
                  rho_info['sigma_0_adjusted']))

    def _separation_check(self, hx_s_id: torch.Tensor, hx_s_ood: torch.Tensor,
                          std_num: float = 0, mode: str = 'median'):
        if mode == 'median':
            max_scale = hx_s_id.median()
            min_scale = hx_s_ood.median()
        elif mode == 'mean':
            max_scale = hx_s_id.median() + std_num * hx_s_id.std()
            min_scale = hx_s_ood.median() - std_num * hx_s_ood.std()
        else:
            raise ValueError('Only support thresholding with mode = "median" or "mean".')

        diff = (max_scale - min_scale) / 2
        mid = (max_scale + min_scale) / 2

        if diff <= 1e-4:
            print('[warning] ID & OOD data are not well separated by the OOD classifier, '
                             'with ID %s = %.2f and OOD %s = %.2f, '
                             'please consider the following: \n'
                             '1) use the other mode \n'
                             '2) use the logistic regression rescale method (_compute_scale_logistic()) \n'
                             '3) increase --ood_noise_gamma to generate more different OOD data.'
                             % (mode, max_scale, mode, min_scale))

        return mid, diff

    # Rescaling technique based on median of h(x) on ID and OOD data
    def _compute_scale_thresholding(self, hx_s_id: torch.Tensor, hx_s_ood: torch.Tensor,
                       peak: float = 1, bias: float = 0, std_num: float = 0, mode: str = 'median'):
        # Rescale the input conf to [0,1]
        mid, diff = self._separation_check(hx_s_id, hx_s_ood, std_num, mode)
        self.mid = mid

        self.rescale_w = peak / diff
        self.rescale_b = (bias - self.mid) / diff * peak
        print('Rescale parameters, w: {:.2f}, b: {:.2f}'.format(self.rescale_w, self.rescale_b))

    # Rescaling technique based on logistic regression of h(x) on ID and OOD data
    def _compute_scale_logistic(self, hx_s_id: torch.Tensor, hx_s_ood: torch.Tensor,
                        bias: float = 0, std_num: float = 0, mode: str = 'median'):
        # Rescale the input conf to [0,1]
        mid, diff = self._separation_check(hx_s_id, hx_s_ood, std_num, mode)

        data = torch.cat((hx_s_id, hx_s_ood)).unsqueeze(-1) - mid
        labels = torch.cat((torch.ones_like(hx_s_id), torch.zeros_like(hx_s_ood)))
        self.rescale_w, self.rescale_b = train_logistic_regression(data, labels, 1, 1)
        self.rescale_b = self.rescale_b - self.rescale_w * mid + bias

        self.mid = - self.rescale_b / self.rescale_w
        print('Rescale parameters, w: {:.2f}, b: {:.2f}'.format(self.rescale_w, self.rescale_b))

    def _source_inference(self, name, net, data_loader,
                          msg: str = 'classify source data',
                          progress: bool = False):
        net.eval()

        all_probs = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=msg, disable=not progress):
                data = batch['data'].cuda()
                probs = torch.softmax(net(data), dim=1)
                all_probs.append(probs.cpu())
                all_labels.append(batch['label'])

        if name == 'id':
            self.source_id_probs = torch.cat(all_probs)
            self.source_id_labels = torch.cat(all_labels)
        else:
            self.source_ood_probs = torch.cat(all_probs)
            self.source_ood_labels = torch.ones(len(all_labels)) * self.cls_num

    def target_inference(self, name, net, data_loader,
                         msg: str = 'classify target data ',
                         progress: bool = False):
        net.eval()

        all_probs = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=msg, disable=not progress):
                data = batch['data'].cuda()
                probs = torch.softmax(net(data), dim=1)
                all_probs.append(probs.cpu())
                all_labels.append(batch['label'])

        self.target_cls_probs[name] = torch.cat(all_probs)
        if name == 'id':
            print('Keep original ID data ground truth label.')
            self.target_cls_labels[name] = torch.cat(all_labels)
        else:
            print('Use cls_num for OOD data ground truth label.')
            self.target_cls_labels[name] = \
                torch.ones(len(self.target_cls_probs[name])) * self.cls_num
        # print("target inference labels:", self.target_cls_labels[name])

    def target_multi_inference(self, net, data_loader_dict: dict,
                               msg: str = 'target classification',
                               progress: bool = True):
        for name, data_loader in data_loader_dict.items():
            self.target_inference(name, net, data_loader, msg, progress)

    def estimation(self, fx: torch.Tensor, hx: torch.Tensor, mode: str = 'MLE'):
        self.pi, self.rho_t0 = owls_em(fx.numpy(), hx.numpy(), self.c.numpy(), self.rho_s,
                                       fx_s=self.fx_s.numpy(), hx_s=self.hx_s.numpy(),
                                       estimate=mode)

        self.rho_t = rho_t_correction(self.rho_t0, self.sigma_1, self.sigma_0 / self.T)
        self.pi = torch.Tensor(self.pi)
        self.pi_tilde = torch.cat((self.pi * self.rho_t, torch.Tensor([1 - self.rho_t])))

    def correction(self, fx: torch.Tensor, hx: torch.Tensor = None) -> torch.Tensor:
        if hx is not None:
            assert len(fx.shape) == 2 and len(hx.shape) == 1
            hx = hx.unsqueeze(-1)
            probs = torch.cat((fx * hx, 1 - hx), axis=-1)

            probs = probs * self.pi_tilde / self.c_tilde
            probs /= probs.sum(-1, keepdim=True)

            return probs
        else:
            probs = fx * self.pi / self.c
            probs /= probs.sum(-1, keepdim=True)

            return probs

    def evaluation(self, f, ood_name: str,
                   hx_id: torch.Tensor, hx_ood: torch.Tensor,
                   id_data_loader=None, ood_data_loader=None,
                   mode: str = 'MAP',
                   msg: str = 'target classification',
                   progress: bool = True):
        print('Test id median: {:.2f}, Test id mean: {:.2f}, Test ood median: {:.2f}, Test ood mean: {:.2f}'.format(
            hx_id.median(), hx_id.mean(), hx_ood.median(), hx_ood.mean()))
        print(
            '[Rescale]Test id median: {:.2f}, Test id mean: {:.2f}, Test ood median: {:.2f}, Test ood mean: {:.2f}'.format(
                self._rescale(hx_id).median(), self._rescale(hx_id).mean(),
                self._rescale(hx_ood).median(), self._rescale(hx_ood).mean()))

        if id_data_loader is not None:
            if 'id' in self.target_cls_probs.keys():
                print('ID test data already inferred, use existing one.')
            else:
                self.target_inference('id', f, id_data_loader, msg + ' id', progress)

        if ood_data_loader is not None:
            if ood_name in self.target_cls_probs.keys():
                print('[warning] OOD {} data already inferred.'.format(ood_name))
            self.target_inference(ood_name, f, ood_data_loader, msg + ' ' + ood_name, progress)

        assert len(hx_id) == len(self.target_cls_probs['id'])
        assert len(hx_ood) == len(self.target_cls_probs[ood_name])

        # Obtain Ground Truth Values
        # _, count = torch.unique(self.target_cls_labels['id'], return_counts=True)
        count = torch.zeros_like(self.target_cls_probs['id'][-1])
        for i in self.target_cls_labels['id']:
            count[i] += 1
        self.pi_gt = count / count.sum()

        self.w_gt = self.pi_gt / self.c
        self.rho_t_gt = len(hx_id) / (len(hx_id) + len(hx_ood))
        self.pi_tilde_gt = torch.cat((self.pi_gt * self.rho_t_gt, torch.Tensor([1 - self.rho_t_gt])))
        w_tilde_gt = self.pi_tilde_gt / self.c_tilde

        # Concatenate Prediction for ID & OOD samples
        fx = torch.cat((self.target_cls_probs['id'], self.target_cls_probs[ood_name]), axis=0)
        hx = torch.cat((hx_id, hx_ood))
        labels = torch.cat((self.target_cls_labels['id'], self.target_cls_labels[ood_name]))

        # get_acc_metrics(self.target_cls_probs['id'].cpu().numpy(),
        #                 self.target_cls_labels['id'].cpu().numpy())

        # Rescaled hx for label shift correction task
        # hx_rescaled = self._correction_rescale(hx)
        hx_rescaled = self._rescale(hx)
        self._record_score_diagnostics(
            'target',
            'shifted_for_owls',
            hx_id,
            hx_ood,
            self.target_cls_labels['id'],
        )
        self._record_score_diagnostics(
            'target',
            'rescaled',
            hx_rescaled[:len(hx_id)],
            hx_rescaled[len(hx_id):],
            self.target_cls_labels['id'],
        )
        # Construct the fx_tilde for K+1 classes of ID & OOD
        fx_tilde = torch.cat((fx * hx_rescaled.unsqueeze(-1), 1 - hx_rescaled.unsqueeze(-1)), axis=-1)

        # Open World Label Shift Estimation of pi, rho_t
        c_t = time()
        self.estimation(fx, hx_rescaled, mode=mode)
        self._record_rho_diagnostics()
        print("Estimation time %.4f" % (time() - c_t))
        # Open World Label Shift Correction of predictions
        new_fx_tilde = self.correction(fx, hx_rescaled)

        # ID classes label shift correction according to pi estimated in OSLS setting
        new_fx = self.correction(fx)
        new_hx = hx.clone()

        # Metrics Evaluation
        u = torch.ones_like(self.c) / len(self.c)
        w = self.pi / self.c
        w_u = u / self.c
        w_tilde = self.pi_tilde / self.c_tilde
        w_tilde_u = torch.cat((u * self.rho_t, torch.Tensor([1 - self.rho_t]))) / self.c_tilde

        print("Ground truth rho_t {:.2f}, Predicted rho_t {:.2f}".format(self.rho_t_gt, self.rho_t))
        # print("Ground truth w_tilde {}, Predicted w {}".format(str(self.pi_tilde_gt / self.c_tilde), str(w_tilde)))

        self.baseline_metrics = {'id_acc': acc_cal(fx_tilde.numpy()[:len(hx_id)],
                                                   labels.numpy()[:len(hx_id)].astype(int)),
                                 'ood_acc': acc_cal(fx_tilde.numpy()[len(hx_id):],
                                                    labels.numpy()[len(hx_id):].astype(int)),
                                 'all_acc': acc_cal(fx_tilde.numpy(),
                                                    labels.numpy().astype(int)),
                                 'w_mse': get_mse(w_u.numpy(), self.w_gt.numpy())
                                 }

        self.metrics = {'id_acc': acc_cal(new_fx_tilde.numpy()[:len(hx_id)],
                                          labels.numpy()[:len(hx_id)].astype(int)),
                        'ood_acc': acc_cal(new_fx_tilde.numpy()[len(hx_id):],
                                           labels.numpy()[len(hx_id):].astype(int)),
                        'all_acc': acc_cal(new_fx_tilde.numpy(),
                                           labels.numpy().astype(int)),
                        'w_mse': get_mse(w.numpy(), self.w_gt.numpy())
                        }

        self.cs_metrics = leval(fx.cpu().numpy(), list(labels.numpy().astype(int)),
                                self.source_id_probs.detach().cpu().numpy(),
                                self.source_id_labels.detach().cpu().numpy(),
                                train_cls_num_list=self.train_cls_num_list)
