import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def main():

    dataset1 = {"cifar10": [0.29, 0.35, 0.40, 0.48, 0.51],
               "tin": [0.26, 0.34, 0.38, 0.48, 0.51],
               "nearood": [0.27, 0.32, 0.39, 0.48, 0.51],
               "mnist": [0.23, 0.30, 0.37, 0.48, 0.51],
               "svhn": [0.22, 0.30, 0.37, 0.48, 0.51],
               "texture": [0.28, 0.34, 0.39, 0.48, 0.51],
               "places365": [0.25, 0.32, 0.38, 0.48, 0.51],
               "farood": [0.25, 0.31, 0.37, 0.48, 0.51]}

    dataset2 = {"cifar10": [0.23, 0.26, 0.28, 0.32, 0.34],
               "tin": [0.21, 0.24, 0.27, 0.32, 0.34],
               "nearood": [0.22, 0.25, 0.28, 0.32, 0.34],
               "mnist": [0.14, 0.28, 0.24, 0.31, 0.34],
               "svhn": [0.35, 0.35, 0.34, 0.34, 0.34],
               "texture": [0.34, 0.34, 0.34, 0.33, 0.34],
               "places365": [0.19, 0.22, 0.26, 0.32, 0.34],
               "farood": [0.15, 0.27, 0.29, 0.32, 0.34]}

    gt = [0.33, 0.50, 0.67, 0.91, 0.99]


    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{amssymb}')
    params = {'text.usetex': True,
              'font.size': 14,
              'font.family': 'lmodern'
              }
    plt.rcParams.update(params)

    def f(x, fpr, tpr):
        return (x - fpr) / (tpr - fpr)

    tpr = 0.7
    fpr = 0.25

    plt.title(r"Linearity between $\rho_t$ and $\hat{\rho}'_t$")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel(r"Predicted $\hat{\rho}'_t$")
    plt.ylabel(r"Ground Truth $\hat{\rho}_t$")

    plt.xticks([0, fpr, tpr, 1], [0, r"$\hat{\sigma}_0'$ (FPR)", r"$\hat{\sigma}_1'$ (TPR)", 1])
    plt.yticks([0,1])

    x = np.linspace(fpr,tpr,101)
    gt = np.linspace(0,1,101)
    plt.plot(x, f(x, fpr, tpr), label=r'Predicted $\hat{\rho}_t$')
    plt.plot(gt, gt, color='k', label=r'GT $\hat{\rho}_t$')
    plt.vlines(tpr, 0, 1, color='r', linestyles='dashed')
    plt.vlines(fpr, 0, 1, color='g', linestyles='dashed')
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('./rho_t_linearity.pdf')
    plt.show()

    # # plt.ylim(0,1)
    # # plt.xlim(0,1)
    # plt.title(r"[CIFAR100] estimate $p_t(\text{i})=\rho_t$")
    # plt.xlabel(r"Ground Truth")
    # plt.ylabel(r"Prediction")
    # for k, v in dataset1.items():
    #     plt.plot(gt, v, label=k)
    #
    # plt.legend(loc='best')
    # plt.show()
    # plt.close()
    #
    # # plt.ylim(0,1)
    # # plt.xlim(0,1)
    # plt.title(r"[CIFAR100-100-LT] estimate $p_t(\text{i})=\rho_t$")
    # plt.xlabel(r"Ground Truth")
    # plt.ylabel(r"Prediction")
    # for k, v in dataset2.items():
    #     plt.plot(gt, v, label=k)
    #
    # plt.legend(loc='best')
    # plt.show()
    # plt.close()






















if __name__ == "__main__":
    main()