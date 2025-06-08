import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

def addlabels(x,y,l):
    for i in range(len(x)):
        plt.text(x[i],y[i],l[i])

def main():
    cls_num = 3
    ood_num = 1
    img_max = 1000
    alpha = [10.0, 10.0]
    cls_idx = np.arange(cls_num)

    ticks = ['Dog', 'Cat', 'Bird']
    ticks2 = ['Dog', 'Cat', 'Bird', 'OOD']

    train1 = np.ones(cls_num) * img_max
    test1 = dirichlet_imb_gen(img_max, cls_num, alpha[0], replace=True)
    # test1 = np.ones(cls_num) * img_max
    ood1 = np.random.uniform(np.max(test1), np.max(test1))

    train2 = exp_imb_gen(img_max, cls_num, 0.05)[::-1]
    test2 = dirichlet_imb_gen(img_max, cls_num, alpha[1], replace=True)
    ood2 = np.random.uniform(np.max(test2)/2, np.max(test2))

    train3 = exp_imb_gen(img_max, cls_num, 0.5)[::-1]
    test3 = train3.copy()[::-1]
    ood3 = np.random.uniform(np.max(test3)/2, np.max(test3) )

    train = [train1, train2, train3]
    test = [test1, test2, test3]
    ood = [ood1, ood2, ood3]

    tick_params = {"axis": 'both', "left": False, "top": False, "right": False, "bottom": False,
                   "labelleft": False, "labeltop": False,
                   "labelright": False, "labelbottom": False}
    fig_size = (6.0, 2.2)
    fig_size2 = (6.0, 2.2)

    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{amssymb}')
    params = {'text.usetex': True,
              'font.size': 20,
              'font.family': 'lmodern'
              }
    # params.update({'axes.spines.right': False,
    #                 'axes.spines.top': False,
    #                'axes.spines.left': True,
    #                'axes.spines.bottom': True,
    #                })
    plt.rcParams.update(params)


    # fig, axes = plt.subplots(3, 1, figsize=(6.4, 6.6))
    for i, x in enumerate(train):
        x = x / np.sum(x) * 100
        plt.figure(figsize=fig_size)
        # if i == 0:
        #     tick_params.update({"labelbottom": True})
        # else:
        #     tick_params.update({"labelbottom": False})
        plt.tick_params(**tick_params)
        # plt.xticks(cls_idx, ticks)
        plt.bar(cls_idx, x, alpha=0.5, color='blue')
        # plt.bar([cls_num], 0, alpha=0.5, color='yellow')
        # plt.text(cls_num, np.max(x)/2, "?")
        # addlabels(cls_idx - 0.1, x, ['%.2f' % a for a in (x/np.sum(x))])

        plt.tight_layout()
        plt.savefig('./fig1/train_' + str(i) + '.pdf')
        plt.show()
        plt.close()

    for i, x in enumerate(test):
        data_sum = (np.sum(x) + ood[i])
        x = x / data_sum * 100
        ood[i] = ood[i] / data_sum * 100

        plt.figure(figsize=fig_size2)
        # if i == 0:
        #     tick_params.update({"labelbottom": True})
        # else:
        #     tick_params.update({"labelbottom": False})
        plt.tick_params(**tick_params)
        # plt.xticks(np.arange(cls_num + 1), ticks2)
        plt.bar(cls_idx, x, alpha=0.5, color="blue")
        # plt.bar([cls_num, cls_num + 1, cls_num + 2], [ood[i], ood[i], ood[i]], alpha=0.5, color='yellow')
        # addlabels(np.arange(cls_num + 1) - 0.1, list(x) + [ood[i]],
        #           ['%.2f' % a for a in (x + [ood[i]]) / (np.sum(x) + ood[i])])
        # plt.fill_between(np.arange(cls_num - 0.2, cls_num + ood_num), 0, max(x), alpha=0.5, color='yellow')

        plt.tight_layout()
        plt.savefig('./fig1/test_' + str(i) + '.pdf')
        plt.show()
        plt.close()



# Long-Tailed Shift Test Set--------------------------------------------------#
def exp_imb_gen(img_max, cls_num, imb_factor):
    img_num_per_cls = []
    for cls_idx in range(cls_num):
        num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(math.trunc(num))
    if np.array(img_num_per_cls).min() == 0:
        raise Exception("Imbalance factor too small that 0 sample in tail class")

    return img_num_per_cls[::-1]


# Dirichlet Shift Test Set----------------------------------------------------#
def dirichlet_imb_gen(img_max, cls_num, alpha,
                      total_num: int = None, max_iter: int = 50, replace: bool = False):
    sample_num = img_max * cls_num
    if total_num is not None:
        assert total_num <= sample_num
        sample_num = total_num

    if not replace:
        for i in range(max_iter):
            probs = np.random.dirichlet([alpha] * cls_num)
            img_num_per_cls = [math.trunc(x * sample_num) for x in probs]
            if np.max(img_num_per_cls) <= img_max:
                break
            elif i == (max_iter - 1):
                raise Exception('Dirichlet shift with %i iterations still failed to generate plausible subset, '
                                'please recheck the params.' % max_iter)
    else:
        probs = np.random.dirichlet([alpha] * cls_num)
        img_num_per_cls = [math.trunc(x * sample_num) for x in probs]

    if 0 in img_num_per_cls:
        print('[warning] Some test class have 0 sample number after Dirichlet Shift.')

    return img_num_per_cls


if __name__ == "__main__":
    main()
