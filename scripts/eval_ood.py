import os, sys
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(ROOT_DIR)
DATA_DIR = '$YOUR_DATASET_PATH/'
import numpy as np
import pandas as pd
import argparse
import pickle
import yaml
import collections
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F

from openood.evaluation_api import Evaluator

from openood.networks import ResNet18_32x32, ResNet18_224x224, resnet32, ResNet50
from openood.networks.conf_branch_net import ConfBranchNet
from openood.networks.godin_net import GodinNet
from openood.networks.rot_net import RotNet
from openood.networks.csi_net import CSINet
from openood.networks.udg_net import UDGNet
from openood.networks.cider_net import CIDERNet
from openood.networks.npos_net import NPOSNet


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


parser = argparse.ArgumentParser()
parser.add_argument('--root', required=True)
parser.add_argument('--test_subset_config_path', default=None)
parser.add_argument('--train_subset_config_path', default=None)
parser.add_argument('--ood_ratio', type=float, default=None)
parser.add_argument('--ood_noise_gamma', type=float, default=0.1)
parser.add_argument('--postprocessor', default='msp')
parser.add_argument(
    '--id-data',
    type=str,
    default='cifar10',
    choices=['cifar10', 'cifar100', 'aircraft', 'cub', 'imagenet', 'imagenet_lt', 'imagenet200'])
parser.add_argument('--batch-size', type=int, default=512)
parser.add_argument('--save-csv', action='store_true')
parser.add_argument('--save-score', action='store_true')
parser.add_argument('--fsood', action='store_true')
args = parser.parse_args()

root = args.root

# Imbalance subset config path
train_subset_path = args.train_subset_config_path
if train_subset_path is not None:
    with open(train_subset_path, 'r') as f:
        train_subset_config = yaml.safe_load(f)['train_subset']
    print('Loaded train imbalance subset config: {}'.format(str(train_subset_config)))
else:
    train_subset_config = {'mode': 'Original'}

test_subset_path = args.test_subset_config_path
if test_subset_path is not None:
    with open(test_subset_path, 'r') as f:
        test_subset_config = yaml.safe_load(f)['test_subset']
    print('Loaded test imbalance subset config: {}'.format(str(test_subset_config)))
else:
    test_subset_config = {
        'id': {'mode': 'Original'},
        'csid': {'mode': 'Original'},
        'ood': {'mode': 'Original', 'ratio': 'None'},
    }

if args.ood_ratio is not None:
    test_subset_config['ood'] = {'mode': 'Reduce', 'ratio': args.ood_ratio, 'order': 'normal'}


# specify an implemented postprocessor
# 'openmax', 'msp', 'temp_scaling', 'odin'...
postprocessor_name = args.postprocessor

NUM_CLASSES = {'cifar10': 10, 'cifar100': 100, 'imagenet200': 200, 'imagenet': 1000}
MODEL = {
    'cifar10': ResNet18_32x32,
    'cifar100': ResNet18_32x32,
    'imagenet200': ResNet18_224x224,
    'imagenet': ResNet50,
    'imagenet_lt': ResNet50,
}

# MODEL = {
#     'cifar10': resnet32,
#     'cifar100': resnet32,
#     'imagenet200': ResNet18_224x224,
#     'imagenet': ResNet50,
#     'imagenet_lt': ResNet50,
# }

try:
    num_classes = NUM_CLASSES[args.id_data]
    model_arch = MODEL[args.id_data]
except KeyError:
    raise NotImplementedError(f'ID dataset {args.id_data} is not supported.')

# assume that the root folder contains subfolders each corresponding to
# a training run, e.g., s0, s1, s2
# this structure is automatically created if you use OpenOOD for train
if len(glob(os.path.join(root, 's*'))) == 0:
    raise ValueError(f'No subfolders found in {root}')

# iterate through training runs
all_metrics = []
for subfolder in sorted(glob(os.path.join(root, 's*'))):
    # Set manual seed to keep train set.
    seed = int(subfolder.split('/')[-1][1:])
    torch.manual_seed(seed)
    np.random.seed(seed)

    # load pre-setup postprocessor if exists
    if os.path.isfile(
            os.path.join(subfolder, 'postprocessors',
                         f'{postprocessor_name}.pkl')):
        with open(
                os.path.join(subfolder, 'postprocessors',
                             f'{postprocessor_name}.pkl'), 'rb') as f:
            postprocessor = pickle.load(f)
    else:
        postprocessor = None

    # load the pretrained model provided by the user
    if postprocessor_name == 'conf_branch':
        net = ConfBranchNet(backbone=model_arch(num_classes=num_classes),
                            num_classes=num_classes)
    elif postprocessor_name == 'godin':
        backbone = model_arch(num_classes=num_classes)
        net = GodinNet(backbone=backbone,
                       feature_size=backbone.feature_size,
                       num_classes=num_classes)
    elif postprocessor_name == 'rotpred':
        net = RotNet(backbone=model_arch(num_classes=num_classes),
                     num_classes=num_classes)
    elif 'csi' in root:
        backbone = model_arch(num_classes=num_classes)
        net = CSINet(backbone=backbone,
                     feature_size=backbone.feature_size,
                     num_classes=num_classes)
    elif 'udg' in root:
        backbone = model_arch(num_classes=num_classes)
        net = UDGNet(backbone=backbone,
                     num_classes=num_classes,
                     num_clusters=1000)
    elif postprocessor_name == 'cider':
        backbone = model_arch(num_classes=num_classes)
        net = CIDERNet(backbone,
                       head='mlp',
                       feat_dim=128,
                       num_classes=num_classes)
    elif postprocessor_name == 'npos':
        backbone = model_arch(num_classes=num_classes)
        net = NPOSNet(backbone,
                      head='mlp',
                      feat_dim=128,
                      num_classes=num_classes)
    else:
        net = model_arch(num_classes=num_classes)

    net.load_state_dict(
        torch.load(os.path.join(subfolder, 'best.ckpt'), map_location='cpu'))
    net.cuda()
    net.eval()

    print('***********', ROOT_DIR, DATA_DIR, os.path.join(ROOT_DIR, 'data'))
    evaluator = Evaluator(
        net,
        id_name=args.id_data,  # the target ID dataset
        # data_root=os.path.join(DATA_DIR, 'data'),
        data_root=DATA_DIR,
        config_root=os.path.join(ROOT_DIR, 'configs'),
        preprocessor=None,  # default preprocessing
        postprocessor_name=postprocessor_name,
        postprocessor=
        postprocessor,  # the user can pass his own postprocessor as well
        batch_size=args.
        batch_size,  # for certain methods the results can be slightly affected by batch size
        shuffle=False,
        num_workers=8,
        test_subset_config=test_subset_config,
        train_subset_config=train_subset_config,
        ood_noise_gamma=args.ood_noise_gamma
    )

    # load pre-computed scores if exist
    '''
    if os.path.isfile(
            os.path.join(subfolder, 'scores', f'{postprocessor_name}.pkl')):
        with open(
                os.path.join(subfolder, 'scores', f'{postprocessor_name}.pkl'),
                'rb') as f:
            scores = pickle.load(f)
        update(evaluator.scores, scores)
        print('Loaded pre-computed scores from file.')
    '''

    # save the postprocessor for future reuse
    if hasattr(evaluator.postprocessor, 'setup_flag'
               ) or evaluator.postprocessor.hyperparam_search_done is True:
        pp_save_root = os.path.join(subfolder, 'postprocessors')
        if not os.path.exists(pp_save_root):
            os.makedirs(pp_save_root)

        if not os.path.isfile(
                os.path.join(pp_save_root, f'{postprocessor_name}.pkl')):
            with open(os.path.join(pp_save_root, f'{postprocessor_name}.pkl'),
                      'wb') as f:
                pickle.dump(evaluator.postprocessor, f,
                            pickle.HIGHEST_PROTOCOL)

    metrics = evaluator.eval_ood(fsood=args.fsood, progress=False)
    all_metrics.append(metrics.to_numpy())

    # save computed scores
    if args.save_score:
        score_save_root = os.path.join(subfolder, 'scores')
        if not os.path.exists(score_save_root):
            os.makedirs(score_save_root)
        with open(os.path.join(score_save_root, f'{postprocessor_name}.pkl'),
                  'wb') as f:
            pickle.dump(evaluator.scores, f, pickle.HIGHEST_PROTOCOL)

# compute mean metrics over training runs
all_metrics = np.stack(all_metrics, axis=0)
metrics_mean = np.mean(all_metrics, axis=0)
metrics_std = np.std(all_metrics, axis=0)

final_metrics = []
for i in range(len(metrics_mean)):
    temp = []
    for j in range(metrics_mean.shape[1]):
        if j < 14:
            temp.append(u'{:.2f} \u00B1 {:.2f}'.format(metrics_mean[i, j],
                                                       metrics_std[i, j]))
        else:
            temp.append(u'{:.6f} \u00B1 {:.6f}'.format(metrics_mean[i, j],
                                                       metrics_std[i, j]))
    final_metrics.append(temp)
df = pd.DataFrame(final_metrics, index=metrics.index, columns=metrics.columns)


log_name = "test_ood-%s-%s-%s-%s-%.1f-" % (args.id_data, train_subset_config['name'], test_subset_config['name'],
                            str(test_subset_config['ood']['ratio']), args.ood_noise_gamma)
if args.save_csv:
    # saving_root = os.path.join(root, 'ood' if not args.fsood else 'fsood')
    saving_root = os.path.join("./results/", 'ood' if not args.fsood else 'fsood')
    if not os.path.exists(saving_root):
        os.makedirs(saving_root)
    save_path = os.path.join(saving_root, log_name + f'{postprocessor_name}.csv')
    df.to_csv(save_path)
    print("Experiment finished! log file has been saved in %s" % save_path)
else:
    print(df)
