import os
import torch
from numpy import load
from torch.utils.data import DataLoader

from openood.preprocessors.test_preprocessor import TestStandardPreProcessor
from openood.preprocessors.utils import get_preprocessor
from openood.utils.config import Config
from openood.label_shift.imbalance_dataset_generator import imb_subset

from .feature_dataset import FeatDataset
from .imglist_dataset import ImglistDataset
from .imglist_augmix_dataset import ImglistAugMixDataset
from .imglist_extradata_dataset import ImglistExtraDataDataset, TwoSourceSampler


def get_dataloader(config: Config):
    # prepare a dataloader dictionary
    dataset_config = config.dataset

    # print('#' * 20, hasattr(config, 'trainer'))
    test_subset_config = config.test_subset if hasattr(config, 'test_subset') else \
        {'id': {'mode': 'Original'}, 'csid': {'mode': 'Original'}, 'ood': {'mode': 'Original'}}
    train_subset_config = config.train_subset if hasattr(config, 'train_subset') else {'mode': 'Original'}

    dataloader_dict = {}
    for split in dataset_config.split_names:
        split_config = dataset_config[split]
        preprocessor = get_preprocessor(config, split)
        # weak augmentation for data_aux
        data_aux_preprocessor = TestStandardPreProcessor(config)

        if split_config.dataset_class == 'ImglistExtraDataDataset':
            dataset = ImglistExtraDataDataset(
                name=dataset_config.name + '_' + split,
                imglist_pth=split_config.imglist_pth,
                data_dir=split_config.data_dir,
                num_classes=dataset_config.num_classes,
                preprocessor=preprocessor,
                data_aux_preprocessor=data_aux_preprocessor,
                extra_data_pth=split_config.extra_data_pth,
                extra_label_pth=split_config.extra_label_pth,
                extra_percent=split_config.extra_percent)

            orig_len = dataset.__len__()
            if split == 'train':
                dataset, cls_num_list = imb_subset(dataset, split_config.imglist_pth, train_subset_config)
                print('Train Dataset {}, Original {:d}, Subsampled {:d}, Path {}'.format(
                    split, orig_len, dataset.__len__(), split_config.imglist_pth))
                dataloader_dict['train_cls_num_list'] = cls_num_list
            elif split == 'test':
                dataset, cls_num_list = imb_subset(dataset, split_config.imglist_pth, test_subset_config['id'],
                                                   random=True)
                print('Test Dataset {}, Original {:d}, Subsampled {:d}, Path {}'.format(
                    split, orig_len, dataset.__len__(), split_config.imglist_pth))
                dataloader_dict['test_cls_num_list'] = cls_num_list


            batch_sampler = TwoSourceSampler(dataset.orig_ids,
                                             dataset.extra_ids,
                                             split_config.batch_size,
                                             split_config.orig_ratio)

            dataloader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=dataset_config.num_workers,
            )
        elif split_config.dataset_class == 'ImglistAugMixDataset':
            dataset = ImglistAugMixDataset(
                name=dataset_config.name + '_' + split,
                imglist_pth=split_config.imglist_pth,
                data_dir=split_config.data_dir,
                num_classes=dataset_config.num_classes,
                preprocessor=preprocessor,
                data_aux_preprocessor=data_aux_preprocessor)

            orig_len = dataset.__len__()
            if split == 'train':
                dataset, cls_num_list = imb_subset(dataset, split_config.imglist_pth, train_subset_config)
                print('Train Dataset {}, Original {:d}, Subsampled {:d}, Path {}'.format(
                    split, orig_len, dataset.__len__(), split_config.imglist_pth))
                dataloader_dict['train_cls_num_list'] = cls_num_list
            elif split == 'test':
                dataset, cls_num_list = imb_subset(dataset, split_config.imglist_pth, test_subset_config['id'],
                                                   random=True)
                print('Test Dataset {}, Original {:d}, Subsampled {:d}, Path {}'.format(
                    split, orig_len, dataset.__len__(), split_config.imglist_pth))
                dataloader_dict['test_cls_num_list'] = cls_num_list

            sampler = None
            if dataset_config.num_gpus * dataset_config.num_machines > 1:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset)
                split_config.shuffle = False

            dataloader = DataLoader(dataset,
                                    batch_size=split_config.batch_size,
                                    shuffle=split_config.shuffle,
                                    num_workers=dataset_config.num_workers,
                                    sampler=sampler)
        else:
            CustomDataset = eval(split_config.dataset_class)
            dataset = CustomDataset(
                name=dataset_config.name + '_' + split,
                imglist_pth=split_config.imglist_pth,
                data_dir=split_config.data_dir,
                num_classes=dataset_config.num_classes,
                preprocessor=preprocessor,
                data_aux_preprocessor=data_aux_preprocessor)

            orig_len = dataset.__len__()
            if split == 'train':
                dataset, cls_num_list = imb_subset(dataset, split_config.imglist_pth, train_subset_config)
                print('Train Dataset {}, Original {:d}, Subsampled {:d}, Path {}'.format(
                    split, orig_len, dataset.__len__(), split_config.imglist_pth))
                dataloader_dict['train_cls_num_list'] = cls_num_list
            elif split == 'test':
                dataset, cls_num_list = imb_subset(dataset, split_config.imglist_pth, test_subset_config['id'],
                                                   random=True)
                print('Test Dataset {}, Original {:d}, Subsampled {:d}, Path {}'.format(
                    split, orig_len, dataset.__len__(), split_config.imglist_pth))
                dataloader_dict['test_cls_num_list'] = cls_num_list

            sampler = None
            if dataset_config.num_gpus * dataset_config.num_machines > 1:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset)
                split_config.shuffle = False

            dataloader = DataLoader(dataset,
                                    batch_size=split_config.batch_size,
                                    shuffle=split_config.shuffle,
                                    num_workers=dataset_config.num_workers,
                                    sampler=sampler)

        dataloader_dict[split] = dataloader

    print('Train dataset label distribution: {}'.format(str(dataloader_dict['train_cls_num_list'])))
    print('Test dataset label distribution: {}'.format(str(dataloader_dict['test_cls_num_list'])))

    return dataloader_dict


def get_ood_dataloader(config: Config):
    # specify custom dataset class
    subset_config = config.subset if hasattr(config, 'subset') else None
    ood_config = config.ood_dataset

    CustomDataset = eval(ood_config.dataset_class)
    dataloader_dict = {}
    for split in ood_config.split_names:
        split_config = ood_config[split]
        preprocessor = get_preprocessor(config, split)
        data_aux_preprocessor = TestStandardPreProcessor(config)
        if split == 'val':
            # validation set
            dataset = CustomDataset(
                name=ood_config.name + '_' + split,
                imglist_pth=split_config.imglist_pth,
                data_dir=split_config.data_dir,
                num_classes=ood_config.num_classes,
                preprocessor=preprocessor,
                data_aux_preprocessor=data_aux_preprocessor)
            dataloader = DataLoader(dataset,
                                    batch_size=ood_config.batch_size,
                                    shuffle=ood_config.shuffle,
                                    num_workers=ood_config.num_workers)
            dataloader_dict[split] = dataloader
        else:
            # dataloaders for csid, nearood, farood
            sub_dataloader_dict = {}
            for dataset_name in split_config.datasets:
                dataset_config = split_config[dataset_name]
                dataset = CustomDataset(
                    name=ood_config.name + '_' + split,
                    imglist_pth=dataset_config.imglist_pth,
                    data_dir=dataset_config.data_dir,
                    num_classes=ood_config.num_classes,
                    preprocessor=preprocessor,
                    data_aux_preprocessor=data_aux_preprocessor)

                orig_len = dataset.__len__()
                if split == 'csid':
                    dataset, cls_num_list = \
                        imb_subset(dataset, dataset_config.imglist_pth,
                                   {'mode': 'Original'} if subset_config is None else subset_config['id'], random=True)
                    print('CSID Dataset {}, Original {:d}, Subsampled {:d}, Path {}'.format(
                        split, orig_len, dataset.__len__(), split_config.imglist_pth))
                    # sub_dataloader_dict['{}_cls_num_list'.format(dataset_name)] = cls_num_list
                elif split in ['nearood', 'farood']:
                    dataset, cls_num_list = \
                        imb_subset(dataset, dataset_config.imglist_pth,
                                   {'mode': 'Original'} if subset_config is None else subset_config['ood'], random=True)
                    print('OOD Dataset {}, Original {:d}, Subsampled {:d}, Path {}'.format(
                        split, orig_len, dataset.__len__(), split_config.imglist_pth))
                    # sub_dataloader_dict['{}_cls_num_list'.format(dataset_name)] = cls_num_list

                dataloader = DataLoader(dataset,
                                        batch_size=ood_config.batch_size,
                                        shuffle=ood_config.shuffle,
                                        num_workers=ood_config.num_workers)
                sub_dataloader_dict[dataset_name] = dataloader
            dataloader_dict[split] = sub_dataloader_dict

    return dataloader_dict


def get_feature_dataloader(dataset_config: Config):
    # load in the cached feature
    loaded_data = load(dataset_config.feat_path, allow_pickle=True)
    total_feat = torch.from_numpy(loaded_data['feat_list'])
    del loaded_data
    # reshape the vector to fit in to the network
    total_feat.unsqueeze_(-1).unsqueeze_(-1)
    # let's see what we got here should be something like:
    # torch.Size([total_num, channel_size, 1, 1])
    print('Loaded feature size: {}'.format(total_feat.shape))

    split_config = dataset_config['train']

    dataset = FeatDataset(feat=total_feat)
    dataloader = DataLoader(dataset,
                            batch_size=split_config.batch_size,
                            shuffle=split_config.shuffle,
                            num_workers=dataset_config.num_workers)

    return dataloader


def get_feature_opengan_dataloader(dataset_config: Config):
    feat_root = dataset_config.feat_root

    dataloader_dict = {}
    for d in ['id_train', 'id_val', 'ood_val']:
        # load in the cached feature
        loaded_data = load(os.path.join(feat_root, f'{d}.npz'),
                           allow_pickle=True)
        total_feat = torch.from_numpy(loaded_data['feat_list'])
        total_labels = loaded_data['label_list']
        del loaded_data
        # reshape the vector to fit in to the network
        total_feat.unsqueeze_(-1).unsqueeze_(-1)
        # let's see what we got here should be something like:
        # torch.Size([total_num, channel_size, 1, 1])
        print('Loaded feature size: {}'.format(total_feat.shape))

        if d == 'id_train':
            split_config = dataset_config['train']
        else:
            split_config = dataset_config['val']

        dataset = FeatDataset(feat=total_feat, labels=total_labels)
        dataloader = DataLoader(dataset,
                                batch_size=split_config.batch_size,
                                shuffle=split_config.shuffle,
                                num_workers=dataset_config.num_workers)
        dataloader_dict[d] = dataloader

    return dataloader_dict

