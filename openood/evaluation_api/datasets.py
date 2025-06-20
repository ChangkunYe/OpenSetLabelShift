import os
import gdown
import zipfile

from torch.utils.data import DataLoader
import torchvision as tvs
if tvs.__version__ >= '0.13':
    tvs_new = True
else:
    tvs_new = False

from openood.datasets.imglist_dataset import ImglistDataset
from openood.preprocessors import BasePreprocessor
from openood.label_shift.imbalance_dataset_generator import imb_subset

from .preprocessor import get_default_preprocessor, ImageNetCPreProcessor

DATA_INFO = {
    'cifar10': {
        'num_classes': 10,
        'pre_size': 32,
        'id': {
            'train': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/train_cifar10.txt'
            },
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/val_cifar10.txt'
            },
            'test': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/test_cifar10.txt'
            }
        },
        'csid': {
            'datasets': ['cifar10c'],
            'cinic10': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/val_cinic10.txt'
            },
            'cifar10c': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/test_cifar10c.txt'
            }
        },
        'ood': {
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/val_tin.txt'
            },
            'near': {
                'datasets': ['cifar100', 'tin'],
                'cifar100': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar10/test_cifar100.txt'
                },
                'tin': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar10/test_tin.txt'
                }
            },
            'far': {
                'datasets': ['mnist', 'svhn', 'texture', 'places365'],
                'mnist': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar10/test_mnist.txt'
                },
                'svhn': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar10/test_svhn.txt'
                },
                'texture': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar10/test_texture.txt'
                },
                'places365': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar10/test_places365.txt'
                },
            }
        }
    },
    'cifar100': {
        'num_classes': 100,
        'pre_size': 32,
        'id': {
            'train': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/train_cifar100.txt'
            },
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/val_cifar100.txt'
            },
            'test': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/test_cifar100.txt'
            }
        },
        'csid': {
            'datasets': [],
        },
        'ood': {
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/val_tin.txt'
            },
            'near': {
                'datasets': ['cifar10', 'tin'],
                'cifar10': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar100/test_cifar10.txt'
                },
                'tin': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar100/test_tin.txt'
                }
            },
            'far': {
                'datasets': ['mnist', 'svhn', 'texture', 'places365'],
                'mnist': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar100/test_mnist.txt'
                },
                'svhn': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar100/test_svhn.txt'
                },
                'texture': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar100/test_texture.txt'
                },
                'places365': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar100/test_places365.txt'
                }
            },
        }
    },
    'imagenet200': {
        'num_classes': 200,
        'pre_size': 256,
        'id': {
            'train': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/train_imagenet200.txt'
            },
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/val_imagenet200.txt'
            },
            'test': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/test_imagenet200.txt'
            }
        },
        'csid': {
            'datasets': ['imagenet_v2', 'imagenet_c', 'imagenet_r'],
            'imagenet_v2': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/test_imagenet200_v2.txt'
            },
            'imagenet_c': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/test_imagenet200_c.txt'
            },
            'imagenet_r': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/test_imagenet200_c.txt'
            },
        },
        'ood': {
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/val_openimage_o.txt'
            },
            'near': {
                'datasets': ['ssb_hard', 'ninco'],
                'ssb_hard': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet200/test_ssb_hard.txt'
                },
                'ninco': {
                    'data_dir': 'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet200/test_ninco.txt'
                }
            },
            'far': {
                'datasets': ['inaturalist', 'textures', 'openimage_o'],
                'inaturalist': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet200/test_inaturalist.txt'
                },
                'textures': {
                    'data_dir':
                    'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/imagenet200/test_textures.txt'
                },
                'openimage_o': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet200/test_openimage_o.txt'
                },
            },
        }
    },
    'imagenet': {
        'num_classes': 1000,
        'pre_size': 256,
        'id': {
            'train': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/imagenet/train_imagenet.txt'
            },
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/imagenet/val_imagenet.txt'
            },
            'test': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/imagenet/test_imagenet.txt'
            }
        },
        'csid': {
            'datasets': ['imagenet_v2'],
            'imagenet_v2': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet/test_imagenet_v2.txt'
            },
        },
        'ood': {
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet/val_openimage_o.txt'
            },
            'near': {
                'datasets': ['ssb_hard', 'ninco'],
                'ssb_hard': {
                    'data_dir': 'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet/test_ssb_hard.txt'
                },
                'ninco': {
                    'data_dir': 'images_largescale/',
                    'imglist_path': 'benchmark_imglist/imagenet/test_ninco.txt'
                }
            },
            'far': {
                'datasets': ['inaturalist', 'textures', 'openimage_o'],
                'inaturalist': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet/test_inaturalist.txt'
                },
                'textures': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/imagenet/test_textures.txt'
                },
                'openimage_o': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet/test_openimage_o.txt'
                },
            },
        }
    },
    'imagenet_lt': {
        'num_classes': 1000,
        'pre_size': 256,
        'id': {
            'train': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'ImageNet/ImageNet_LT_train.txt'
            },
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path': '/ImageNet/ImageNet_LT_val.txt'
            },
            'test': {
                'data_dir': 'images_largescale/',
                'imglist_path': '/ImageNet/ImageNet_LT_test.txt'
            }
        },
        'csid': {
            'datasets': ['imagenet_v2'],
            'imagenet_v2': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                    'benchmark_imglist/imagenet/test_imagenet_v2.txt'
            },
        },
        'ood': {
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                    'benchmark_imglist/imagenet/val_openimage_o.txt'
            },
            'near': {
                'datasets': ['ssb_hard', 'ninco'],
                'ssb_hard': {
                    'data_dir': 'images_largescale/',
                    'imglist_path':
                        'benchmark_imglist/imagenet/test_ssb_hard.txt'
                },
                'ninco': {
                    'data_dir': 'images_largescale/',
                    'imglist_path': 'benchmark_imglist/imagenet/test_ninco.txt'
                }
            },
            'far': {
                'datasets': ['inaturalist', 'textures', 'openimage_o'],
                'inaturalist': {
                    'data_dir':
                        'images_largescale/',
                    'imglist_path':
                        'benchmark_imglist/imagenet/test_inaturalist.txt'
                },
                'textures': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                        'benchmark_imglist/imagenet/test_textures.txt'
                },
                'openimage_o': {
                    'data_dir':
                        'images_largescale/',
                    'imglist_path':
                        'benchmark_imglist/imagenet/test_openimage_o.txt'
                },
            },
        }
    },
    'places': {
        'num_classes': 365,
        'pre_size': 256,
        'id': {
            'train': {
                'data_dir': 'Places/',
                'imglist_path': 'Places/places365_train_standard.txt'
            },
            'val': {
                'data_dir': 'Places/',
                'imglist_path': 'Places/places365_val.txt'
            },
            'test': {
                'data_dir': 'Places/',
                'imglist_path': 'Places/places365_test.txt'
            },
        },
        'ood': {
            'val': {
                'data_dir': 'Places/',
                'imglist_path': 'Places/places365_val.txt'
            },
            'near': {
                'datasets': ['ssb_hard', 'ninco'],
                'ssb_hard': {
                    'data_dir': 'images_largescale/',
                    'imglist_path':
                        'benchmark_imglist/imagenet/test_ssb_hard.txt'
                },
                'ninco': {
                    'data_dir': 'images_largescale/',
                    'imglist_path': 'benchmark_imglist/imagenet/test_ninco.txt'
                }
            },
            'far': {
                'datasets': ['inaturalist', 'textures', 'openimage_o'],
                'inaturalist': {
                    'data_dir':
                        'images_largescale/',
                    'imglist_path':
                        'benchmark_imglist/imagenet/test_inaturalist.txt'
                },
                'textures': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                        'benchmark_imglist/imagenet/test_textures.txt'
                },
                'openimage_o': {
                    'data_dir':
                        'images_largescale/',
                    'imglist_path':
                        'benchmark_imglist/imagenet/test_openimage_o.txt'
                },
            },
        },
    }
}

download_id_dict = {
    'cifar10': '1Co32RiiWe16lTaiOU6JMMnyUYS41IlO1',
    'cifar100': '1PGKheHUsf29leJPPGuXqzLBMwl8qMF8_',
    'tin': '1PZ-ixyx52U989IKsMA2OT-24fToTrelC',
    'mnist': '1CCHAGWqA1KJTFFswuF9cbhmB-j98Y1Sb',
    'svhn': '1DQfc11HOtB1nEwqS4pWUFp8vtQ3DczvI',
    'texture': '1OSz1m3hHfVWbRdmMwKbUzoU8Hg9UKcam',
    'places365': '1Ec-LRSTf6u5vEctKX9vRp9OA6tqnJ0Ay',
    'imagenet_1k': '1i1ipLDFARR-JZ9argXd2-0a6DXwVhXEj',
    'species_sub': '1-JCxDx__iFMExkYRMylnGJYTPvyuX6aq',
    'ssb_hard': '1PzkA-WGG8Z18h0ooL_pDdz9cO-DCIouE',
    'ninco': '1Z82cmvIB0eghTehxOGP5VTdLt7OD3nk6',
    'inaturalist': '1zfLfMvoUD0CUlKNnkk7LgxZZBnTBipdj',
    'places': '1fZ8TbPC4JGqUCm-VtvrmkYxqRNp2PoB3',
    'sun': '1ISK0STxWzWmg-_uUr4RQ8GSLFW7TZiKp',
    'openimage_o': '1VUFXnB_z70uHfdgJG2E_pjYOcEgqM7tE',
    'imagenet_v2': '1akg2IiE22HcbvTBpwXQoD7tgfPCdkoho',
    'imagenet_r': '1EzjMN2gq-bVV7lg-MEAdeuBuz-7jbGYU',
    'imagenet_c': '1JeXL9YH4BO8gCJ631c5BHbaSsl-lekHt',
    'benchmark_imglist': '1XKzBdWCqg3vPoj-D32YixJyJJ0hL63gP'
}

dir_dict = {
    'images_classic/': [
        'cifar100', 'tin', 'tin597', 'svhn', 'cinic10', 'imagenet10', 'mnist',
        'fashionmnist', 'cifar10', 'cifar100c', 'places365', 'cifar10c',
        'fractals_and_fvis', 'usps', 'texture', 'notmnist'
    ],
    'images_largescale/': [
        'imagenet_1k',
        'ssb_hard',
        'ninco',
        'inaturalist',
        'places',
        'sun',
        'openimage_o',
        'imagenet_v2',
        'imagenet_c',
        'imagenet_r',
    ],
    'images_medical/': ['actmed', 'bimcv', 'ct', 'hannover', 'xraybone'],
}

benchmarks_dict = {
    'cifar10':
    ['cifar10', 'cifar100', 'tin', 'mnist', 'svhn', 'texture', 'places365'],
    'cifar100':
    ['cifar100', 'cifar10', 'tin', 'mnist', 'svhn', 'texture', 'places365'],
    'imagenet200': [
        'imagenet_1k', 'ssb_hard', 'ninco', 'inaturalist', 'texture',
        'openimage_o', 'imagenet_v2', 'imagenet_c', 'imagenet_r'
    ],
    'imagenet': [
        'imagenet_1k', 'ssb_hard', 'ninco', 'inaturalist', 'texture',
        'openimage_o', 'imagenet_v2', 'imagenet_c', 'imagenet_r'
    ],
}


def require_download(filename, path):
    for item in os.listdir(path):
        if item.startswith(filename) or filename.startswith(
                item) or path.endswith(filename):
            return False

    else:
        print(filename + ' needs download:')
        return True


def download_dataset(dataset, data_root):
    for key in dir_dict.keys():
        if dataset in dir_dict[key]:
            store_path = os.path.join(data_root, key, dataset)
            if not os.path.exists(store_path):
                os.makedirs(store_path)
            break
    else:
        print('Invalid dataset detected {}'.format(dataset))
        return

    if require_download(dataset, store_path):
        print(store_path)
        if not store_path.endswith('/'):
            store_path = store_path + '/'
        gdown.download(id=download_id_dict[dataset], output=store_path)

        file_path = os.path.join(store_path, dataset + '.zip')
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            zip_file.extractall(store_path)
        os.remove(file_path)


def data_setup(data_root, id_data_name):
    if not data_root.endswith('/'):
        data_root = data_root + '/'

    if not os.path.exists(os.path.join(data_root, 'benchmark_imglist')):
        gdown.download(id=download_id_dict['benchmark_imglist'],
                       output=data_root)
        file_path = os.path.join(data_root, 'benchmark_imglist.zip')
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            zip_file.extractall(data_root)
        os.remove(file_path)

    for dataset in benchmarks_dict[id_data_name]:
        download_dataset(dataset, data_root)


def get_id_ood_dataloader(id_name, data_root, preprocessor,
                          train_subset_config, test_subset_config, ood_noise_gamma,
                          **loader_kwargs):
    if 'imagenet' in id_name:
        if tvs_new:
            if isinstance(preprocessor,
                          tvs.transforms._presets.ImageClassification):
                mean, std = preprocessor.mean, preprocessor.std
            elif isinstance(preprocessor, tvs.transforms.Compose):
                temp = preprocessor.transforms[-1]
                mean, std = temp.mean, temp.std
            elif isinstance(preprocessor, BasePreprocessor):
                temp = preprocessor.transform.transforms[-1]
                mean, std = temp.mean, temp.std
            else:
                raise TypeError
        else:
            if isinstance(preprocessor, tvs.transforms.Compose):
                temp = preprocessor.transforms[-1]
                mean, std = temp.mean, temp.std
            elif isinstance(preprocessor, BasePreprocessor):
                temp = preprocessor.transform.transforms[-1]
                mean, std = temp.mean, temp.std
            else:
                raise TypeError
        imagenet_c_preprocessor = ImageNetCPreProcessor(mean, std)

    # weak augmentation for data_aux
    test_standard_preprocessor = get_default_preprocessor(id_name)

    dataloader_dict = {}
    data_info = DATA_INFO[id_name]

    # id
    sub_dataloader_dict = {}
    for split in data_info['id'].keys():
        imglist_pth = os.path.join(data_root,
                                     data_info['id'][split]['imglist_path'])
        dataset = ImglistDataset(
            name='_'.join((id_name, split)),
            imglist_pth=imglist_pth,
            data_dir=os.path.join(data_root,
                                  data_info['id'][split]['data_dir']),
            num_classes=data_info['num_classes'],
            preprocessor=preprocessor,
            data_aux_preprocessor=test_standard_preprocessor)

        # Sampling Subset according to imb_config
        orig_len = dataset.__len__()
        # if split == 'train' or split == 'fake_ood':
        if split == 'train':
            dataset, train_cls_num_list = \
                imb_subset(dataset, imglist_pth,
                           {'mode': 'Original'} if train_subset_config is None else train_subset_config
                           )
            print('Train ID Dataset {}, Original {:d}, Subsampled {:d}, Path {}'.format(
                split, orig_len, dataset.__len__(), imglist_pth))
            dataloader_dict['train_cls_num_list'] = train_cls_num_list
            print('Train dataset label distribution: ' + str(train_cls_num_list))
        elif split == 'val':
            dataset, train_cls_num_list = \
                imb_subset(dataset, imglist_pth,
                           {'mode': 'Original'} if train_subset_config is None else train_subset_config,
                           random=True
                           )
            print('Val ID Dataset {}, Original {:d}, Subsampled {:d}, Path {}'.format(
                split, orig_len, dataset.__len__(), imglist_pth))
        elif split == 'test':
            dataset, test_cls_num_list = \
                imb_subset(dataset, imglist_pth,
                           {'mode': 'Original'} if test_subset_config is None else test_subset_config['id'],
                           random=True
                           )
            print('Test ID Dataset {}, Original {:d}, Subsampled {:d}, Path {}'.format(
                split, orig_len, dataset.__len__(), imglist_pth))
            ood_num = dataset.__len__() if test_subset_config['ood']['ratio'] == 'None' else \
                int(dataset.__len__() * test_subset_config['ood']['ratio'])
            test_subset_config['ood']['data_num'] = ood_num
            test_cls_num_list.append(ood_num)
            dataloader_dict['test_cls_num_list'] = test_cls_num_list
            print('Test dataset label distribution: ', test_cls_num_list)

        dataloader = DataLoader(dataset, **loader_kwargs)
        sub_dataloader_dict[split] = dataloader
    dataloader_dict['id'] = sub_dataloader_dict

    # dataloader_dict['train_cls_num_list'] = train_cls_num_list
    # dataloader_dict['test_cls_num_list'] = test_cls_num_list

    # fake ood
    # sub_dataloader_dict = {}
    dummy_size = (3, data_info['pre_size'], data_info['pre_size'])
    # if 'fake_ood' in data_info.keys():
    imglist_pth = os.path.join(data_root, data_info['id']['val']['imglist_path'])
    pseudo_ood_dataset = ImglistDataset(
        name='fake_ood',
        imglist_pth=imglist_pth,
        data_dir=os.path.join(data_root,
                              data_info['id']['train']['data_dir']),
        num_classes=data_info['num_classes'],
        preprocessor=preprocessor,
        data_aux_preprocessor=test_standard_preprocessor,
        dummy_read=True,
        dummy_size=dummy_size,
        fake_ood_gamma=ood_noise_gamma,
    )
    dataset, _ = imb_subset(pseudo_ood_dataset, imglist_pth,
                            {'mode': 'Original'} if train_subset_config is None else train_subset_config,
                            random=True
                            )
    dataloader = DataLoader(dataset, **loader_kwargs)
    # sub_dataloader_dict[split] = dataloader
    dataloader_dict['fake_ood'] = dataloader


    # csid
    sub_dataloader_dict = {}
    for dataset_name in data_info['csid']['datasets']:
        imglist_pth = os.path.join(
                data_root, data_info['csid'][dataset_name]['imglist_path'])
        dataset = ImglistDataset(
            name='_'.join((id_name, 'csid', dataset_name)),
            imglist_pth=imglist_pth,
            data_dir=os.path.join(data_root,
                                  data_info['csid'][dataset_name]['data_dir']),
            num_classes=data_info['num_classes'],
            preprocessor=preprocessor
            if dataset_name != 'imagenet_c' else imagenet_c_preprocessor,
            data_aux_preprocessor=test_standard_preprocessor)

        # Sampling Subset according to imb_config
        orig_len = dataset.__len__()
        if test_subset_config is not None:
            dataset, csid_cls_num_list  = \
                imb_subset(dataset, imglist_pth,
                           {'mode': 'Original'} if test_subset_config is None else test_subset_config['id'],
                           random=True)
            print('CSID Dataset {}, Original {:d}, Subsampled {:d}, Path {}'.format(
                dataset_name, orig_len, dataset.__len__(), imglist_pth))

        dataloader = DataLoader(dataset, **loader_kwargs)
        sub_dataloader_dict[dataset_name] = dataloader
        dataloader_dict['{}_cls_num_list'.format(dataset_name)] = csid_cls_num_list
    dataloader_dict['csid'] = sub_dataloader_dict

    # ood
    dataloader_dict['ood'] = {}
    for split in data_info['ood'].keys():
        split_config = data_info['ood'][split]

        if split == 'val':
            # validation set
            imglist_pth = os.path.join(data_root,
                                         split_config['imglist_path'])
            dataset = ImglistDataset(
                name='_'.join((id_name, 'ood', split)),
                imglist_pth=imglist_pth,
                data_dir=os.path.join(data_root, split_config['data_dir']),
                num_classes=data_info['num_classes'],
                preprocessor=preprocessor,
                data_aux_preprocessor=test_standard_preprocessor)

            # print('Original:', split, imglist_pth, dataset.__len__())
            # if imb_config is not None:
            #    dataset, _ = imb_subset(dataset, imglist_pth, imb_config['ood'])
            #    print('Subsampled:', split, imglist_pth, dataset.__len__())

            dataloader = DataLoader(dataset, **loader_kwargs)
            dataloader_dict['ood'][split] = dataloader
        else:
            # dataloaders for nearood, farood
            sub_dataloader_dict = {}
            for dataset_name in split_config['datasets']:
                dataset_config = split_config[dataset_name]
                imglist_pth=os.path.join(data_root, dataset_config['imglist_path'])
                dataset = ImglistDataset(
                    name='_'.join((id_name, 'ood', dataset_name)),
                    imglist_pth=imglist_pth,
                    data_dir=os.path.join(data_root,
                                          dataset_config['data_dir']),
                    num_classes=data_info['num_classes'],
                    preprocessor=preprocessor,
                    data_aux_preprocessor=test_standard_preprocessor)

                orig_len = dataset.__len__()
                if test_subset_config is not None:
                    dataset, _ = \
                        imb_subset(dataset, imglist_pth,
                                   {'mode': 'Original'} if test_subset_config is None else test_subset_config['ood'])
                    print('OOD Dataset {}, Original {:d}, Subsampled {:d}, Path {}'.format(
                        dataset_name, orig_len, dataset.__len__(), imglist_pth))

                dataloader = DataLoader(dataset, **loader_kwargs)
                sub_dataloader_dict[dataset_name] = dataloader
                dataloader_dict['{}_cls_num_list'.format(dataset_name)] = [dataset.__len__()]
            dataloader_dict['ood'][split] = sub_dataloader_dict

    return dataloader_dict
