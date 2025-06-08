from typing import Callable, List, Type

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partialmethod

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

from openood.evaluators.metrics import compute_all_metrics
from openood.postprocessors import BasePostprocessor
from openood.networks.ash_net import ASHNet
from openood.networks.react_net import ReactNet
from openood.networks.scale_net import ScaleNet
from openood.label_shift import OpenSetLabelShift

from .datasets import DATA_INFO, data_setup, get_id_ood_dataloader
from .postprocessor import get_postprocessor
from .preprocessor import get_default_preprocessor


class Evaluator:
    def __init__(
            self,
            net: nn.Module,
            id_name: str,
            data_root: str = './data',
            config_root: str = './configs',
            preprocessor: Callable = None,
            postprocessor_name: str = None,
            postprocessor: Type[BasePostprocessor] = None,
            batch_size: int = 200,
            shuffle: bool = False,
            num_workers: int = 4,
            test_subset_config: dict = None,
            train_subset_config: dict = None,
            ood_noise_gamma: float = 0.1,
    ) -> None:
        """A unified, easy-to-use API for evaluating (most) discriminative OOD
        detection methods.

        Args:
            net (nn.Module):
                The base classifier.
            id_name (str):
                The name of the in-distribution dataset.
            data_root (str, optional):
                The path of the data folder. Defaults to './data'.
            config_root (str, optional):
                The path of the config folder. Defaults to './configs'.
            preprocessor (Callable, optional):
                The preprocessor of input images.
                Passing None will use the default preprocessor
                following convention. Defaults to None.
            postprocessor_name (str, optional):
                The name of the postprocessor that obtains OOD score.
                Ignored if an actual postprocessor is passed.
                Defaults to None.
            postprocessor (Type[BasePostprocessor], optional):
                An actual postprocessor instance which inherits
                OpenOOD's BasePostprocessor. Defaults to None.
            batch_size (int, optional):
                The batch size of samples. Defaults to 200.
            shuffle (bool, optional):
                Whether shuffling samples. Defaults to False.
            num_workers (int, optional):
                The num_workers argument that will be passed to
                data loaders. Defaults to 4.

        Raises:
            ValueError:
                If both postprocessor_name and postprocessor are None.
            ValueError:
                If the specified ID dataset {id_name} is not supported.
            TypeError:
                If the passed postprocessor does not inherit BasePostprocessor.
        """
        # check the arguments
        self.owls = None
        if postprocessor_name is None and postprocessor is None:
            raise ValueError('Please pass postprocessor_name or postprocessor')
        if postprocessor_name is not None and postprocessor is not None:
            print(
                'Postprocessor_name is ignored because postprocessor is passed'
            )
        if id_name not in DATA_INFO:
            raise ValueError(f'Dataset [{id_name}] is not supported')

        self.num_classes = DATA_INFO[id_name]['num_classes']

        # get data preprocessor
        if preprocessor is None:
            preprocessor = get_default_preprocessor(id_name if id_name[-3:] != "_lt" else id_name[:-3])

        # set up config root
        if config_root is None:
            filepath = os.path.dirname(os.path.abspath(__file__))
            config_root = os.path.join(*filepath.split('/')[:-2], 'configs')

        # get postprocessor
        if postprocessor is None:
            postprocessor = get_postprocessor(config_root, postprocessor_name, id_name)
        if not isinstance(postprocessor, BasePostprocessor):
            raise TypeError(
                'postprocessor should inherit BasePostprocessor in OpenOOD')

        # load data
        data_setup(data_root, id_name)
        loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'test_subset_config': test_subset_config,
            'train_subset_config': train_subset_config,
            'ood_noise_gamma': ood_noise_gamma,
        }
        dataloader_dict = get_id_ood_dataloader(id_name, data_root,
                                                preprocessor, **loader_kwargs)

        # wrap base model to work with certain postprocessors
        if postprocessor_name == 'react':
            net = ReactNet(net)
        elif postprocessor_name == 'ash':
            net = ASHNet(net)
        elif postprocessor_name == 'scale':
            net = ScaleNet(net)

        # postprocessor setup
        postprocessor.setup(net, dataloader_dict['id'], dataloader_dict['ood'])

        self.id_name = id_name
        self.net = net
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.dataloader_dict = dataloader_dict
        self.metrics = {
            'id_acc': None,
            'csid_acc': None,
            'ood': None,
            'fsood': None
        }
        self.scores = {
            'id': {
                'train': None,
                'val': None,
                'test': None
            },
            'csid': {k: None
                     for k in dataloader_dict['csid'].keys()},
            'ood': {
                'val': None,
                'near':
                    {k: None
                     for k in dataloader_dict['ood']['near'].keys()},
                'far': {k: None
                        for k in dataloader_dict['ood']['far'].keys()},
            },
            'id_preds': None,
            'id_labels': None,
            'csid_preds': {k: None
                           for k in dataloader_dict['csid'].keys()},
            'csid_labels': {k: None
                            for k in dataloader_dict['csid'].keys()},
        }
        # perform hyperparameter search if have not done so
        if (self.postprocessor.APS_mode
                and not self.postprocessor.hyperparam_search_done):
            self.hyperparam_search()

        self.net.eval()

        # how to ensure the postprocessors can work with
        # models whose definition doesn't align with OpenOOD

        self.log_file = "./eval_%s_%s_%s_%s.txt" % \
                   (id_name, train_subset_config["name"], test_subset_config['name'],
                    test_subset_config['ood']['ratio'])

    def _classifier_inference(self,
                              data_loader: DataLoader,
                              msg: str = 'Acc Eval',
                              progress: bool = True):
        self.net.eval()

        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=msg, disable=not progress):
                data = batch['data'].cuda()
                logits = self.net(data)
                preds = logits.argmax(1)
                all_preds.append(preds.cpu())
                all_labels.append(batch['label'])

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        return all_preds, all_labels

    def eval_acc(self, data_name: str = 'id') -> float:
        if data_name == 'id':
            if self.metrics['id_acc'] is not None:
                return self.metrics['id_acc']
            else:
                if self.scores['id_preds'] is None:
                    all_preds, all_labels = self._classifier_inference(
                        self.dataloader_dict['id']['test'], 'ID Acc Eval', progress=False)
                    self.scores['id_preds'] = all_preds
                    self.scores['id_labels'] = all_labels
                else:
                    all_preds = self.scores['id_preds']
                    all_labels = self.scores['id_labels']

                assert len(all_preds) == len(all_labels)
                correct = (all_preds == all_labels).sum().item()
                acc = correct / len(all_labels) * 100
                self.metrics['id_acc'] = acc
                return acc
        elif data_name == 'csid':
            if self.metrics['csid_acc'] is not None:
                return self.metrics['csid_acc']
            else:
                correct, total = 0, 0
                for _, (dataname, dataloader) in enumerate(
                        self.dataloader_dict['csid'].items()):
                    if self.scores['csid_preds'][dataname] is None:
                        all_preds, all_labels = self._classifier_inference(
                            dataloader, f'CSID {dataname} Acc Eval')
                        self.scores['csid_preds'][dataname] = all_preds
                        self.scores['csid_labels'][dataname] = all_labels
                    else:
                        all_preds = self.scores['csid_preds'][dataname]
                        all_labels = self.scores['csid_labels'][dataname]

                    assert len(all_preds) == len(all_labels)
                    c = (all_preds == all_labels).sum().item()
                    t = len(all_labels)
                    correct += c
                    total += t

                if self.scores['id_preds'] is None:
                    all_preds, all_labels = self._classifier_inference(
                        self.dataloader_dict['id']['test'], 'ID Acc Eval')
                    self.scores['id_preds'] = all_preds
                    self.scores['id_labels'] = all_labels
                else:
                    all_preds = self.scores['id_preds']
                    all_labels = self.scores['id_labels']

                correct += (all_preds == all_labels).sum().item()
                total += len(all_labels)

                acc = correct / total * 100
                self.metrics['csid_acc'] = acc
                return acc
        else:
            raise ValueError(f'Unknown data name {data_name}')

    def eval_ood(self, fsood: bool = False, progress: bool = True):
        id_name = 'id' if not fsood else 'csid'
        task = 'ood' if not fsood else 'fsood'

        print('#' * 70)
        print('id_name: {}, task: {}, fsood: {}'.format(id_name, task, fsood))
        print('#' * 70)

        if self.metrics[task] is None:
            self.net.eval()

            # id score
            if self.scores['id']['test'] is None:
                print(f'Performing inference on {self.id_name} test set...',
                      flush=True)
                id_pred, id_conf, id_gt = self.postprocessor.inference(
                    self.net, self.dataloader_dict['id']['test'], progress)
                self.scores['id']['test'] = [id_pred, id_conf, id_gt]
            else:
                id_pred, id_conf, id_gt = self.scores['id']['test']

            # Initialize OWLS model
            self.owls = OpenSetLabelShift(self.net, self.postprocessor.inference,
                                            self.dataloader_dict['id']['val'],
                                            self.dataloader_dict['fake_ood'],
                                            torch.Tensor(self.dataloader_dict['train_cls_num_list']),
                                            progress=progress)

            if fsood:
                csid_pred, csid_conf, csid_gt = [], [], []
                for i, dataset_name in enumerate(self.scores['csid'].keys()):
                    if self.scores['csid'][dataset_name] is None:
                        print(
                            f'Performing inference on {self.id_name} '
                            f'(cs) test set [{i + 1}]: {dataset_name}...',
                            flush=True)
                        temp_pred, temp_conf, temp_gt = \
                            self.postprocessor.inference(
                                self.net,
                                self.dataloader_dict['csid'][dataset_name],
                                progress)
                        self.scores['csid'][dataset_name] = [
                            temp_pred, temp_conf, temp_gt
                        ]

                    csid_pred.append(self.scores['csid'][dataset_name][0])
                    csid_conf.append(self.scores['csid'][dataset_name][1])
                    csid_gt.append(self.scores['csid'][dataset_name][2])

                csid_pred = np.concatenate(csid_pred)
                csid_conf = np.concatenate(csid_conf)
                csid_gt = np.concatenate(csid_gt)

                id_pred = np.concatenate((id_pred, csid_pred))
                id_conf = np.concatenate((id_conf, csid_conf))
                id_gt = np.concatenate((id_gt, csid_gt))

            print('Shape of the ID data prediction: {}'.format(id_pred.shape))

            # load nearood data and compute ood metrics
            near_metrics = self._eval_ood([id_pred, id_conf, id_gt],
                                          ood_split='near',
                                          progress=progress)
            # load farood data and compute ood metrics
            far_metrics = self._eval_ood([id_pred, id_conf, id_gt],
                                         ood_split='far',
                                         progress=progress)

            if self.metrics[f'{id_name}_acc'] is None:
                self.eval_acc(id_name)
            # near_metrics[:, -1] = np.array([self.metrics[f'{id_name}_acc']] *
            #                               len(near_metrics))
            # far_metrics[:, -1] = np.array([self.metrics[f'{id_name}_acc']] *
            #                              len(far_metrics))

            self.metrics[task] = pd.DataFrame(
                np.concatenate([near_metrics, far_metrics], axis=0),
                index=list(self.dataloader_dict['ood']['near'].keys()) +
                      ['nearood'] + list(self.dataloader_dict['ood']['far'].keys()) +
                      ['farood'],
                columns=['FPR@95', 'AUROC', 'AUPR_IN', 'AUPR_OUT', 'M_ID_ACC', 'M_OOD_ACC', 'M_ALL_ACC',
                         'B_ID_ACC', 'B_OOD_ACC', 'B_ALL_ACC', 'OWLS_ID_ACC', 'OWLS_OOD_ACC', 'OWLS_ALL_ACC',
                         'WMSE', 'OWLS WMSE', 'GT rho_t', 'rho_t', 'rho_t0',
                         'b wmse', 'bbse wmse', 'mlls wmse', 'rlls wmse', 'mapls wmse', 'ours wmse'],
            )
        else:
            print('Evaluation has already been done!')

        with pd.option_context(
                'display.max_rows', None, 'display.max_columns', None,
                'display.expand_frame_repr', False,
                'display.float_format', '{:,.2f}'.format):  # more options can be specified also
            print(self.metrics[task])

        print('\n OOD dataset & ' + ' & '.join([k for k, v in self.metrics[task].items()]) + ' \\\\')
        print("\n".join(k + ' & ' + " & ".join(map(lambda x: "{:.2f}".format(x), xs)) + ' \\\\'
                        for k, xs in self.metrics[task].iterrows()))

        return self.metrics[task]

    def _eval_ood(self,
                  id_list: List[np.ndarray],
                  ood_split: str = 'near',
                  progress: bool = True):
        print(f'Processing {ood_split} ood...', flush=True)
        [id_pred, id_conf, id_gt] = id_list

        metrics_list = []
        for dataset_name, ood_dl in self.dataloader_dict['ood'][
            ood_split].items():
            if self.scores['ood'][ood_split][dataset_name] is None:
                print(f'Performing inference on {dataset_name} dataset...',
                      flush=True)
                ood_pred, ood_conf, ood_gt = self.postprocessor.inference(
                    self.net, ood_dl, progress)
                self.scores['ood'][ood_split][dataset_name] = [
                    ood_pred, ood_conf, ood_gt
                ]
                # print(1 + id_conf, 1 + ood_conf)
                print('Shape of the ID data prediction: {}'.format(id_conf.shape))
                print('Shape of the {} dataset OOD prediction: {}'.format(dataset_name, ood_conf.shape))

                print("Performing Open Label Shift Evaluation on OOD dataset {}...".format(dataset_name))
                self.owls.evaluation(self.net, dataset_name,
                                     torch.Tensor(1 + id_conf), torch.Tensor(1 + ood_conf),
                                     id_data_loader=self.dataloader_dict['id']['test'],
                                     ood_data_loader=ood_dl,
                                     progress=progress)
            else:
                print(
                    'Inference has been performed on '
                    f'{dataset_name} dataset...',
                    flush=True)
                [ood_pred, ood_conf,
                 ood_gt] = self.scores['ood'][ood_split][dataset_name]

            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            # ood_gt = self.num_classes * np.ones_like(ood_gt)
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])

            print(f'Computing metrics on {dataset_name} dataset...')
            ood_metrics = [x * 100 for x in compute_all_metrics(conf, label, pred)] + \
                          [self.owls.baseline_metrics['id_acc'], self.owls.baseline_metrics['ood_acc'],
                           self.owls.baseline_metrics['all_acc'],
                           self.owls.metrics['id_acc'], self.owls.metrics['ood_acc'], self.owls.metrics['all_acc'],
                           self.owls.baseline_metrics['w_mse'], self.owls.metrics['w_mse'],
                           self.owls.rho_t_gt, self.owls.rho_t, self.owls.rho_t0]

            ood_metrics.extend([self.owls.baseline_metrics['w_mse'],
                                self.owls.cs_metrics['bbse_soft']['w_mse'],
                                self.owls.cs_metrics['mlls_soft']['w_mse'],
                                self.owls.cs_metrics['rlls_soft']['w_mse'],
                                self.owls.cs_metrics['mapls_soft']['w_mse'],
                                self.owls.metrics['w_mse'],
                                ])
            metrics_list.append(ood_metrics)
            self._print_metrics(ood_metrics)



        print('Computing mean metrics...', flush=True)
        metrics_list = np.array(metrics_list)
        metrics_mean = np.mean(metrics_list, axis=0, keepdims=True)
        self._print_metrics(list(metrics_mean[0]))
        return np.concatenate([metrics_list, metrics_mean], axis=0)

    def _print_metrics(self, metrics):
        [fpr, auroc, aupr_in, aupr_out, m_id_acc, m_ood_acc, m_all_acc,
         u_id_acc, u_ood_acc, u_all_acc, owls_id_acc, owls_ood_acc, owls_all_acc,
         b_wmse, wmse, rho_t_gt, rho_t, rho_t0, _, _, _, _, _, _ ] \
            = metrics

        # print ood metric results
        print('FPR@95: {:.2f}, AUROC: {:.2f}'.format(fpr, auroc), end=' ', flush=True)
        print('AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}'.format(aupr_in, aupr_out), flush=True)
        print('Method ID ACC: {:.2f}, Method OOD ACC: {:.2f}, Method all ACC: {:.2f},'.format(
            m_id_acc, m_ood_acc, m_all_acc
        ), flush=True)
        print('Baseline ID ACC: {:.2f}, Baseline OOD ACC: {:.2f}, Baseline all ACC: {:.2f},'.format(
            u_id_acc, u_ood_acc, u_all_acc
        ), flush=True)
        print('OWLS ID ACC: {:.2f}, OWLS OOD ACC: {:.2f}, OWLS all ACC: {:.2f},'.format(
            owls_id_acc, owls_ood_acc, owls_all_acc
        ), flush=True)
        print('Baseline WMSE: {:.4f}, OWLS WMSE: {:.4f}'.format(b_wmse, wmse), flush=True)
        print('GT rho_t: {:.2f}, rho_t: {:.2f}, no correct rho_t {:.2f}'.format(rho_t_gt, rho_t, rho_t0), flush=True)
        print('-' * 70, flush=True)
        print('', flush=True)

    def hyperparam_search(self):
        print('Starting automatic parameter search...')
        max_auroc = 0
        hyperparam_names = []
        hyperparam_list = []
        count = 0

        for name in self.postprocessor.args_dict.keys():
            hyperparam_names.append(name)
            count += 1

        for name in hyperparam_names:
            hyperparam_list.append(self.postprocessor.args_dict[name])

        hyperparam_combination = self.recursive_generator(
            hyperparam_list, count)

        final_index = None
        for i, hyperparam in enumerate(hyperparam_combination):
            self.postprocessor.set_hyperparam(hyperparam)

            id_pred, id_conf, id_gt = self.postprocessor.inference(
                self.net, self.dataloader_dict['id']['val'])
            ood_pred, ood_conf, ood_gt = self.postprocessor.inference(
                self.net, self.dataloader_dict['ood']['val'])

            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            ood_metrics = compute_all_metrics(conf, label, pred)
            auroc = ood_metrics[1]

            print('Hyperparam: {}, auroc: {}'.format(hyperparam, auroc))
            if auroc > max_auroc:
                final_index = i
                max_auroc = auroc

        self.postprocessor.set_hyperparam(hyperparam_combination[final_index])
        print('Final hyperparam: {}'.format(
            self.postprocessor.get_hyperparam()))
        self.postprocessor.hyperparam_search_done = True

    def recursive_generator(self, list, n):
        if n == 1:
            results = []
            for x in list[0]:
                k = []
                k.append(x)
                results.append(k)
            return results
        else:
            results = []
            temp = self.recursive_generator(list, n - 1)
            for x in list[n - 1]:
                for y in temp:
                    k = y.copy()
                    k.append(x)
                    results.append(k)
            return results
