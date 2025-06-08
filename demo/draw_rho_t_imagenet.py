import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

headings = ['\\multirow{4}{*}{OpenMax}  & \\multirow{2}{*}{Near} &  Baseline & ',
            '&  &  \\textbf{ours} & ',
            '& \\multirow{2}{*}{Far} & Baseline & ',
            '&  &  \\textbf{ours} & ',
            '\\multirow{4}{*}{MLS}  & \\multirow{2}{*}{Near} & Baseline & ',
            '&  &  \\textbf{ours} & ',
            '& \\multirow{2}{*}{Far} &  Baseline & ',
            '&  &  \\textbf{ours} & ',
            '\\multirow{4}{*}{ReAct}  & \\multirow{2}{*}{Near} & Baseline & ',
            '&  &  \\textbf{ours} & ',
            '& \\multirow{2}{*}{Far} &  Baseline & ',
            '&  &  \\textbf{ours} & ',
            '\\multirow{4}{*}{KNN}  & \\multirow{2}{*}{Near} & Baseline & ',
            '&  &  \\textbf{ours} & ',
            '& \\multirow{2}{*}{Far} &  Baseline & ',
            '&  &  \\textbf{ours} & ',
            '\\multirow{4}{*}{Ash}  & \\multirow{2}{*}{Near} & Baseline & ',
            '&  &  \\textbf{ours} & ',
            '& \\multirow{2}{*}{Far} &  Baseline & ',
            '&  &  \\textbf{ours} & '
            ]

method_list = {
    'mls':  "MLS",
    'openmax':  "OpenMax",
    'react':    "ReAct",
    'knn':      "KNN",
    'ash':      "Ash",
}

shift_name = {
    'LT10': "LT10 F",
    'LT50': "LT50 F",
    'LT100': "LT100 F",
    'LT10r': "LT10 B",
    'LT50r': "LT50 B",
    'LT100r': "LT100 B",
    'Dir1': r"Dir $\alpha=1$",
    'Dir10': r"Dir $\alpha=10$",
}

def main():
    info = [
        {
            "csv_prefix": "/Users/changkunye/Projects/LOGS/OpenOOD/ImageNet200-20240809/",
            "dataset": 'imagenet200',
            'name': "ImageNet-200",
        },
    ]


    estimate = 'MAP'
    ood_shift_types = ["LT", "LTr"]
    # ood_shift_type = ["Dir"]

    # id_shifts = ['Original', 'LT100']
    id_shifts = ['Original']

    # ood_shifts_names = [['Origainal', 'LT10', 'LT50', 'LT100'], ['Original', 'LT10r', 'LT50r', 'LT100r']]
    ood_shifts_names = [['LT10', 'LT100', 'LT10r', 'LT100r']]
    # ood_shifts_names = [['Dir1', 'Dir10']]

    # ood_shifts_names = [['LT10'], ['LT10r']]
    ood_id_ratios = ["1.0", "0.1", "0.01"]
    methods = ["openmax", "mls", "react", "knn", "ash"]

    name_list = ['imagenet200']
    data_row = [3, 7]

    for ood_shift_name in ood_shifts_names:
        plot_rows = 2
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{amssymb}')
        params = {'text.usetex': True,
                  'font.size': 18,
                  'font.family': 'lmodern'
                  }
        plt.rcParams.update(params)

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']


        n_row, n_col = (plot_rows, 4)
        fig, axes = plt.subplots(n_row, n_col, sharey='row', figsize=(19.2, 4.0 * plot_rows))
        for l in range(n_row):
            axes[l, 0].set_ylim(0.3, 1)
            axes[l, 0].set_ylabel(r'Predicted $\hat{\rho}_t$')


        i = -1
        for id_shift in id_shifts:
            for o_shift in ood_shift_name:
                for p, name in zip(info, name_list):
                    i += 1
                    k = -1
                    print(i)
                    for method in methods:
                        k += 1

                        rho_t_gt = []
                        rho_t1_near, rho_t1_std_near = [], []
                        rho_t0_near, rho_t0_std_near = [], []
                        rho_t1_far, rho_t1_std_far = [], []
                        rho_t0_far, rho_t0_std_far = [], []
                        for oi_ratio in ood_id_ratios:
                            file_stamp = "test_ood-%s-%s-%s-%s-%s" % (p['dataset'], id_shift, o_shift, oi_ratio, method)
                            file_path = p["csv_prefix"] + file_stamp + ".csv"
                            # print(file_path)
                            with open(file_path) as f:
                                reader = csv.reader(f, delimiter=',')
                                j = 0
                                for row in reader:
                                    # print(j)
                                    if j == data_row[0]:

                                        rho_t_gt.append(float(row[16].split(' ')[0]))
                                        d1 = row[17].split(' ')
                                        rho_t1_near.append(float(d1[0]))
                                        rho_t1_std_near.append(float(d1[-1]))
                                        d2 = row[18].split(' ')
                                        rho_t0_near.append(float(d2[0]))
                                        rho_t0_std_near.append(float(d2[-1]))

                                    elif j == data_row[1]:
                                        d1 = row[17].split(' ')
                                        rho_t1_far.append(float(d1[0]))
                                        rho_t1_std_far.append(float(d1[-1]))
                                        d2 = row[18].split(' ')
                                        rho_t0_far.append(float(d2[0]))
                                        rho_t0_std_far.append(float(d2[-1]))

                                    j += 1


                        print(method_list[method])
                        axes[0,i].grid(True)
                        axes[0,i].set_xlabel(r'Ground truth $\rho_t$')
                        axes[0,i].plot(rho_t_gt, rho_t1_near, label=method_list[method] + r' $\hat{\rho}_t^*$',
                                     color=colors[k])
                        axes[0,i].fill_between(rho_t_gt, np.array(rho_t1_near) - np.array(rho_t1_std_near),
                                             np.array(rho_t1_near) + np.array(rho_t1_std_near), alpha=0.2,
                                             color=colors[k])

                        axes[0,i].plot(rho_t_gt, rho_t0_near, '--', label=method_list[method] + r' $\hat{\rho}_t$',
                                     color=colors[k])
                        axes[0,i].fill_between(rho_t_gt, np.array(rho_t0_near) - np.array(rho_t0_std_near),
                                             np.array(rho_t0_near) + np.array(rho_t0_std_near), alpha=0.2,
                                             color=colors[k])

                        axes[1,i].grid(True)
                        axes[1,i].set_xlabel(r'Ground truth $\rho_t$')
                        axes[1,i].plot(rho_t_gt, rho_t1_far, label=method_list[method] + r' $\hat{\rho}_t^*$',
                                     color=colors[k])
                        axes[1,i].fill_between(rho_t_gt, np.array(rho_t1_far) - np.array(rho_t1_std_far),
                                             np.array(rho_t1_far) + np.array(rho_t1_std_far), alpha=0.2,
                                             color=colors[k])

                        axes[1,i].plot(rho_t_gt, rho_t0_far, '--', label=method_list[method] + r' $\hat{\rho}_t$',
                                     color=colors[k])
                        axes[1,i].fill_between(rho_t_gt, np.array(rho_t0_far) - np.array(rho_t0_std_far),
                                             np.array(rho_t0_far) + np.array(rho_t0_std_far), alpha=0.2,
                                             color=colors[k])

                    axes[0,i].set_title("%s, %s, Near" % (p['name'], shift_name[o_shift]))
                    axes[0,i].plot(rho_t_gt, rho_t_gt, 'k', label='Ground Truth', linewidth=2.0)

                    axes[1,i].set_title("%s, %s, Far" % (p['name'], shift_name[o_shift]))
                    axes[1,i].plot(rho_t_gt, rho_t_gt, 'k', label='Ground Truth', linewidth=2.0)
                    # axes[i].plot(rho_t_gt, np.ones_like(rho_t_gt) / 2, 'r', label='Baseline', linewidth=1.5)
             # plt.legend(loc='best')

        handles, labels = axes[0,0].get_legend_handles_labels()
        lgd = fig.legend(handles, labels, bbox_to_anchor=(0.85, 0.05), ncol=6)
        plt.tight_layout()
        plt.savefig('./rho_t-imagenet-%s.pdf' % ood_shift_name[0], bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.show()
        plt.close()



def main2():
    info = [
        {
            "csv_prefix": "/Users/changkunye/Projects/LOGS/OpenOOD/ImageNet200-20240809/",
            "dataset": 'imagenet200',
            'name': "ImageNet-200",
        },
    ]

    estimate = 'MAP'
    ood_shift_types = ["LT", "LTr"]
    # ood_shift_type = "LTr"

    # id_shifts = ['Original', 'LT100']
    id_shifts = ['Original']

    # ood_shifts_names = [['Original', 'LT10', 'LT100'], ['Original', 'LT10r', 'LT100r']]
    # ood_shifts_names = [['LT10', 'LT50', 'LT100'], ['LT10r', 'LT50r', 'LT100r']]
    ood_shifts_names = [['LT10', 'LT100'], ['LT10r', 'LT100r']]
    ood_id_ratios = ["1.0", "0.1", "0.01"]
    methods = ["openmax", "mls", "react", "knn", "ash"]

    name_list = ['imagenet200']
    data_row = [3, 8]

    for ood_shift_name in ood_shifts_names:
        plot_rows = 1
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{amssymb}')
        params = {'text.usetex': True,
                  'font.size': 18,
                  'font.family': 'lmodern'
                  }
        plt.rcParams.update(params)
        n_row, n_col = (plot_rows, 4)
        fig, axes = plt.subplots(n_row, n_col, sharey='row', figsize=(19.2, 3.2 * plot_rows))
        axes[0].set_ylim(0.3, 1)
        axes[0].set_ylabel(r'Predicted $\hat{\rho}_t$')
        i = -1
        for id_shift in id_shifts:
            for o_shift in ood_shift_name:
                for p, name in zip(info, name_list):
                    i += 1
                    for method in methods:

                        rho_t_gt = []
                        rho_t1_near, rho_t1_std_near = [], []
                        rho_t0_near, rho_t0_std_near = [], []
                        rho_t1_far, rho_t1_std_far = [], []
                        rho_t0_far, rho_t0_std_far = [], []
                        for oi_ratio in ood_id_ratios:
                            file_stamp = "test_ood-%s-%s-%s-%s-%s" % (p['dataset'], id_shift, o_shift, oi_ratio, method)
                            file_path = p["csv_prefix"] + file_stamp + ".csv"
                            print(file_path)
                            with open(file_path) as f:
                                reader = csv.reader(f, delimiter=',')
                                j = 0
                                for row in reader:
                                    # print(j)
                                    if j == data_row[0]:

                                        rho_t_gt.append(float(row[16].split(' ')[0]))
                                        d1 = row[17].split(' ')
                                        rho_t1_near.append(float(d1[0]))
                                        rho_t1_std_near.append(float(d1[-1]))
                                        d2 = row[18].split(' ')
                                        rho_t0_near.append(float(d2[0]))
                                        rho_t0_std_near.append(float(d2[-1]))

                                    elif j == data_row[1]:
                                        d1 = row[17].split(' ')
                                        rho_t1_far.append(float(d1[0]))
                                        rho_t1_std_far.append(float(d1[-1]))
                                        d2 = row[18].split(' ')
                                        rho_t0_far.append(float(d2[0]))
                                        rho_t0_std_far.append(float(d2[-1]))

                                    j += 1


                        print(method)
                        axes[i].set_xlabel(r'Ground truth $\rho_t$')
                        axes[i].plot(rho_t_gt, rho_t1_near, label=method_list[method] + r' $\hat{\rho}_t^*$')
                        axes[i].fill_between(rho_t_gt, np.array(rho_t1_near) - np.array(rho_t1_std_near),
                                         np.array(rho_t1_near) + np.array(rho_t1_std_near), alpha=0.2)

                        axes[i].plot(rho_t_gt, rho_t0_near, '--', label=method_list[method] + r' $\hat{\rho}_t$')
                        axes[i].fill_between(rho_t_gt, np.array(rho_t0_near) - np.array(rho_t0_std_near),
                                         np.array(rho_t0_near) + np.array(rho_t0_std_near), alpha=0.2)

                    axes[i].set_title("%s, %s" % (p['name'], shift_name[o_shift]))
                    axes[i].plot(rho_t_gt, rho_t_gt, 'k', label='GT', linewidth=2.0)
                    # axes[i].plot(rho_t_gt, np.ones_like(rho_t_gt) / 2, 'r', label='Baseline', linewidth=1.5)
             # plt.legend(loc='best')

        handles, labels = axes[0].get_legend_handles_labels()
        lgd = fig.legend(handles, labels, bbox_to_anchor=(0.8, 0.05), ncol=6)
        plt.tight_layout()
        plt.savefig('./rho_t-cifar-%s.pdf' % ood_shift_name[0], bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.show()
        plt.close()

if __name__ == "__main__":
    main()
    # main2()