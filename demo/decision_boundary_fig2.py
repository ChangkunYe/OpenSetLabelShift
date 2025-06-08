import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Polygon, Wedge
from scipy.stats import norm


def main():
    # c1 = Wedge((3, 7), 3, 0, 360, width=0.05, color='b')
    # c2 = Wedge((6, 4), 3, 0, 360, width=0.05, color='g')
    # patches = [c1,c2]

    x_min = 0
    x_max = 10
    y_max = 0.5
    x = np.linspace(x_min, x_max, 400)

    normal_dist1 = [3, 2]
    normal_dist2 = [7, 2]
    # y1 = norm.pdf(x, loc=normal_dist1[0], scale=normal_dist1[1])
    # y2 = norm.pdf(x, loc=normal_dist2[0], scale=normal_dist2[1])

    a1, b1 = [2, 6]
    y1 = uniform_dist(x, a1, b1)

    a2, b2, cone = [3, 8, 7]
    y2 = triangle_dist(x, a2, b2, cone)
    # y2 = uniform_dist(x, a2, b2)

    # true_boundary = (b1 + a2) / 2
    # x1 = np.linspace(0, true_boundary, max(1, int(true_boundary - 0)) * 50)
    # x2 = np.linspace(true_boundary, x_max, max(1, int(x_max - true_boundary)) * 50)

    normal_dist3 = [4, 2]
    normal_dist4 = [6, 2]

    pred_boundary = (normal_dist3[0] + normal_dist4[0]) / 2
    # pred_boundary = 4
    x3 = np.linspace(0, pred_boundary, max(1, int(pred_boundary - 0)) * 50)
    x4 = np.linspace(pred_boundary, x_max, max(1, int(x_max - pred_boundary)) * 50)

    pi_1 = [0.7, 0.3]
    pi_2 = [0.2, 0.8]
    wrong1 = False
    # if true_boundary < pred_boundary:
    #     overlap = [1 if true_boundary <= z <= pred_boundary else 0 for z in x]
    # else:
    #     wrong1 = True
    #     overlap = [1 if pred_boundary <= z <= true_boundary else 0 for z in x]
    fp = [((a1 >= pred_boundary) and (a2 > 0)) for a1, a2 in zip(x, pi_1[0] * y1)]
    fn = [((a1 <= pred_boundary) and (a2 > 0)) for a1, a2 in zip(x, pi_1[1] * y2)]

    y3 = norm.pdf(x, loc=normal_dist3[0], scale=normal_dist3[1])
    y4 = norm.pdf(x, loc=normal_dist4[0], scale=normal_dist4[1])



    arrow_len = 2
    # boundary1 = get_boundary(x, y1 * pi_1[0], y2 * pi_1[1], normal_dist1[0], normal_dist2[0])
    # boundary2 = get_boundary(x, y1 * pi_2[0], y2 * pi_2[1], normal_dist1[0], normal_dist2[0])
    boundary3 = get_boundary(x, y3 * pi_1[0], y4 * pi_1[1], normal_dist3[0], normal_dist4[0])
    boundary4 = get_boundary(x, y3 * pi_2[0], y4 * pi_2[1], normal_dist3[0], normal_dist4[0])
    # print(boundary1, boundary2, boundary3, boundary4)

    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{amssymb}')
    params = {'text.usetex': True,
              'font.size': 14,
              'font.family': 'lmodern'
              }
    plt.rcParams.update(params)

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 6.4))

    ax = axes[0,0]
    ax.axis('off')
    # ax.xaxis.set_visible(False)
    # ax.yaxis.set_visible(False)
    # for spine in ['top', 'right', 'left', 'bottom']:
    #    ax.spines[spine].set_visible(False)

    # ax.set_xlim(0, 2)
    # ax.set_ylim(0, 2)
    # ax.text(0.9, 1.7, r"Original Model", wrap=True)
    # ax.text(0.9, 1.2, r"Corrected Model", wrap=True)
    # ax.plot(np.linspace(0,2,100), 2 - np.linspace(0,2,100), color='black')

    # ax.text(0.1, 0.6, r'Source $p_s(y=\cdot)=\mathbf{c}$ = %s' % pi_1, wrap=True)
    # ax.text(0.1, 0.3, r'Target $p_t(y=\cdot)=\mathbf{\pi}$ = %s' % pi_2, wrap=True)

    # p = PatchCollection(patches, alpha=0.4)

    ax = axes[1,0]
    # ax.annotate("", xy=(true_boundary - 1, y_max/2), xytext=(true_boundary - 1 - arrow_len, y_max/2),
    #            # arrowprops=dict(arrowstyle="<-", color='blue'))
    # ax.annotate("", xy=(x_max - 1, y_max/2), xytext=(x_max - 1 - arrow_len, y_max/2),
    #            arrowprops=dict(arrowstyle="->", color='green'))
    # # ax.axvline(true_boundary, color='black', ls='-')
    ax.set_title('Source Domain True pdf')
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.set_ylabel(r'$p_s(x|y=\cdot)$')
    # ax.set_ylabel('Ground Truth')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, y_max)
    # ax.add_collection(p)
    ax.plot(x, pi_1[0] * y1, color='b', label=r"Class 1", linewidth=2.0)
    ax.plot(x, pi_1[1] * y2, color='g', label=r"Class 2", linewidth=2.0)

    ax.legend(loc='upper right')

    ax = axes[2,0]
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.set_title('Target Domain True pdf')
    # ax.axvline(true_boundary, color='black', ls='-')
    ax.set_ylabel(r'$p_t(x|y=\cdot)$')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel('x')
    ax.plot(x, pi_2[0] * y1, color='b', label=r"Class 1", linewidth=2.0)
    ax.plot(x, pi_2[1] * y2, color='g', label=r"Class 2", linewidth=2.0)

    # ax.axvline(boundary2, color='black', ls='--', label='Boundary')
    # ax.legend(loc='upper right')


    ax = axes[0, 1]
    ax.xaxis.set_ticklabels([])
    # ax.yaxis.set_ticklabels([])
    # ax.axvline(true_boundary, color='black', ls='-')
    ax.axvline(pred_boundary, color='red', ls='-')
    # ax.set_title(r'Source Domain, $p_s(y=\cdot)=%s$' % pi_1)
    ax.set_title(r'Source Domain Model $f(x)$')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, y_max)
    # ax.add_collection(p)
    # ax.set_ylabel(r'$f(x)_{\cdot}$')
    ax.plot(x, y3, color='mediumslateblue', ls='--', label=r'$f(x)_1$', linewidth=2.0)
    ax.plot(x, y4, color='forestgreen', ls='--', label=r'$f(x)_2$', linewidth=2.0)
    ax.fill_between(x3, y_max * np.ones_like(x3), color='b', alpha=0.1)
    ax.fill_between(x4, y_max * np.ones_like(x4), color='g', alpha=0.1)
    ax.legend(loc='upper right')


    """
    conf3 = y3 / (y3 + y4)
    conf4 = y4 / (y3 + y4)

    ax2 = ax.twinx()
    ax2.yaxis.set_ticklabels([])
    # ax2.set_ylabel(r"Confidence ($f(x)$)")
    ax2.plot(x, conf3, color='mediumslateblue', label=r"$f(x)_1$", ls='-')
    ax2.plot(x, conf4, color='forestgreen', label=r"$f(x)_2$", ls='-')
    ax2.legend(loc='center right')
    """



    ps_x = pi_1[0] * y1 + pi_1[1] * y2
    # ps_x /= np.sum(ps_x)
    ax = axes[1,1]
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    # ax.axvline(true_boundary, color='black', ls='-')
    ax.axvline(pred_boundary, color='red', ls='-')
    # ax.set_title(r'Source Domain, $p_s(y=\cdot)=%s$' % pi_1)
    ax.set_title(r'Optimal')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, y_max)
    # ax.add_collection(p)
    ax.plot(x, pi_1[0] * y1, color='b', linewidth=2.0)
    ax.plot(x, pi_1[1] * y2, color='g', linewidth=2.0)

    ax.fill_between(x3, y_max * np.ones_like(x3), color='b', alpha=0.1)
    ax.fill_between(x4, y_max * np.ones_like(x4), color='g', alpha=0.1)
    ax.fill_between(x, pi_1[0] * y1, color='lime', alpha=0.5)
    ax.fill_between(x, pi_1[1] * y2, color='lime', alpha=0.5, label=r"$TP + TN$")
    ax.fill_between(x, pi_1[0] * y1 * fp,
                    color='r', alpha=0.5, hatch='/', label=r"$FN$")
    ax.fill_between(x,  pi_1[1] * y2 * fn,
                    color='orange', alpha=0.5, hatch='/', label=r"$FP$")
    # ax.axvline(boundary3, color='black', ls='--', label='Boundary')
    ax.legend(loc='upper right')

    # print(ps_x * y3, ps_x * y4)


    pt_x = pi_2[0] * y1 + pi_2[1] * y2
    # pt_x /= np.sum(pt_x)
    ax = axes[2,1]
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    # ax.axvline(true_boundary, color='black', ls='-')
    ax.axvline(pred_boundary, color='red', ls='-')
    ax.set_title(r'Not Optimal' % pi_2)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel('x')
    ax.plot(x, pi_2[0] * y1, color='b', linewidth=2.0)
    ax.plot(x, pi_2[1] * y2, color='g', linewidth=2.0)

    ax.fill_between(x3, y_max * np.ones_like(x3), color='b', alpha=0.1)
    ax.fill_between(x4, y_max * np.ones_like(x4), color='g', alpha=0.1)
    # ax.fill_between(x, pi_2[0] * y1, color='mediumslateblue', alpha=0.5, label=r"$Acc_1$")
    # ax.fill_between(x, pi_2[1] * y2, color='forestgreen', alpha=0.5, label=r"$Acc_2$")
    ax.fill_between(x, pi_2[0] * y1, color='lime', alpha=0.5)
    ax.fill_between(x, pi_2[1] * y2, color='lime', alpha=0.5, label=r"$TP + TN$")
    ax.fill_between(x, pi_2[0] * y1 * fp,
                    color='r', alpha=0.5, hatch='/', label=r"$FN$")
    ax.fill_between(x,  pi_2[1] * y2 * fn,
                    color='orange', alpha=0.5, hatch='/', label=r"$FP$")
    # ax.axvline(boundary4, color='black', ls='--', label='Boundary')
    # ax.legend(loc='upper right')




    y5 = pi_2[0] / pi_1[0] * y3 * 2.5
    y6 = pi_2[1] / pi_1[1] * y4
    ls_boundary = get_boundary(x, y5, y6)
    # ls_boundary = 2.5
    x5 = linspace(0, ls_boundary)
    x6 = linspace(ls_boundary, x_max)

    ax = axes[0, 2]
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    # ax.axvline(true_boundary, color='black', ls='-')
    ax.axvline(ls_boundary, color='red', ls='-')
    # ax.set_title(r'Source Domain, $p_s(y=\cdot)=%s$' % pi_1)
    ax.set_title(r'Target Domain Model $g(x)$ (Objective)')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0)
    # ax.add_collection(p)
    # ax.set_ylabel(r'$g(x)_{\cdot}$')
    ax.plot(x, y5, color='cyan', label=r"$g(x)_1$", ls='--', linewidth=2.0)
    ax.plot(x, y6, color='lime', label=r"$g(x)_2$", ls='--', linewidth=2.0)
    ax.fill_between(x5, 1 * np.ones_like(x5), color='b', alpha=0.1)
    ax.fill_between(x6, 1 * np.ones_like(x6), color='g', alpha=0.1)
    ax.legend(loc='upper right')

    conf5 = y5 / (y5 + y6)
    conf6 = y6 / (y5 + y6)




    wrong2 = False
    # if true_boundary < ls_boundary:
    #     overlap2 = [1 if true_boundary <= z <= ls_boundary else 0 for z in x]
    # else:
    #     wrong2 = True
    #     overlap2 = [1 if ls_boundary <= z <= true_boundary else 0 for z in x]
    fp2 = np.array([((a1 >= ls_boundary) and (a2 > 0)) for a1, a2 in zip(x, y1)]).astype(int)
    fn2 = np.array([((a1 <= ls_boundary) and (a2 > 0)) for a1, a2 in zip(x, y2)]).astype(int)
    # print(fp2, fn2)

    ax = axes[1,2]
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    # ax.axvline(true_boundary, color='black', ls='-')
    ax.axvline(ls_boundary, color='red', ls='-')
    # ax.set_title(r'Source Domain, $p_s(y=\cdot)=%s$' % pi_1)
    ax.set_title(r'Not Optimal')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, y_max)
    # ax.add_collection(p)
    ax.plot(x, pi_1[0] * y1, color='b', linewidth=2.0)
    ax.plot(x, pi_1[1] * y2, color='g', linewidth=2.0)

    ax.fill_between(x5, y_max * np.ones_like(x5), color='b', alpha=0.1)
    ax.fill_between(x6, y_max * np.ones_like(x6), color='g', alpha=0.1)
    # ax.fill_between(x, pi_1[0] * y1, color='cyan', alpha=0.5, label=r"$Acc^s_1$")
    # ax.fill_between(x, pi_1[1] * y2, color='lime', alpha=0.5, label=r"$Acc^s_2$")
    ax.fill_between(x, pi_1[0] * y1, color='lime', alpha=0.5)
    ax.fill_between(x, pi_1[1] * y2, color='lime', alpha=0.5, label=r"$TP + TN$")
    ax.fill_between(x, pi_1[0] * y1 * fp2,
                    color='r', alpha=0.5, hatch='/', label=r"$FN$")
    ax.fill_between(x,  pi_1[1] * y2 * fn2,
                    color='orange', alpha=0.5, hatch='/', label=r"$FP$")
    # ax.axvline(boundary3, color='black', ls='--', label='Boundary')
    ax.legend(loc='upper right')

    # print(ps_x * y3, ps_x * y4)


    pt_x = pi_2[0] * y1 + pi_2[1] * y2
    # pt_x /= np.sum(pt_x)
    ax = axes[2,2]
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    # # ax.axvline(true_boundary, color='black', ls='-')
    # ax.axvline(ls_boundary, color='red', ls='-')
    # ax.axvline(true_boundary, color='black', ls='-')
    ax.axvline(ls_boundary, color='red', ls='-')

    ax.set_title(r'Optimal' % pi_1)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, y_max)
    # ax.add_collection(p)
    ax.plot(x, pi_2[0] * y1, color='b', linewidth=2.0)
    ax.plot(x, pi_2[1] * y2, color='g', linewidth=2.0)
    ax.set_xlabel('x')
    ax.fill_between(x5, y_max * np.ones_like(x5), color='b', alpha=0.1)
    ax.fill_between(x6, y_max * np.ones_like(x6), color='g', alpha=0.1)
    # ax.fill_between(x, pi_2[0] * y1, color='cyan', alpha=0.5, label=r"$Acc^t_1$")
    # ax.fill_between(x, pi_2[1] * y2, color='lime', alpha=0.5, label=r"$Acc^t_2$")
    ax.fill_between(x, pi_2[0] * y1, color='lime', alpha=0.5)
    ax.fill_between(x, pi_2[1] * y2, color='lime', alpha=0.5, label=r"$TP + TN$")
    ax.fill_between(x, pi_2[0] * y1 * fp2,
                    color='r', alpha=0.5, hatch='/', label=r"$FN$")
    ax.fill_between(x,  pi_2[1] * y2 * fn2,
                    color='orange', alpha=0.5, hatch='/', label=r"$FP$")
    # ax.axvline(boundary3, color='black', ls='--', label='Boundary')
    # ax.legend(loc='upper right')

    # handles, labels = axes.get_legend_handles_labels()
    # lgd = fig.legend(handles, labels, bbox_to_anchor=(0.85, 0.05), ncol=6)
    # plt.tight_layout()
    plt.tight_layout()
    plt.savefig('./ls_illustration.pdf')
    plt.show()

def linspace(a, b, nums=50):
    return np.linspace(a, b, max(1, int(b - a)) * nums)


def get_boundary(x, y1, y2, mean1=None, mean2=None):
    idx = 0
    m = 100

    mean1 = 0 if mean1 == None else mean1
    mean2 = np.max(x) if mean2 == None else mean2
    mean = sorted([mean1, mean2])

    for i, (a, b, c) in enumerate(zip(y1, y2, x)):
        diff = abs(a - b)
        if mean[0] <= c <= mean[1]:
            if diff < m:
                m = diff
                idx = i

    return x[idx]


def uniform_dist(x, a ,b):
    return np.array([1 / (b - a) if (a <= loc <= b) else 0 for loc in x])

def triangle_dist(x, a, b, cone):
    assert a < cone < b
    y = []
    h = 2 / (b - a)

    for loc in x:
        if (a <= loc < cone):
            y.append((loc - a) / (cone - a) * h)
        elif (cone <= loc <= b):
            y.append((b - loc) / (b - cone) * h)
        else:
            y.append(0)

    return np.array(y)



if __name__ == '__main__':
    main()