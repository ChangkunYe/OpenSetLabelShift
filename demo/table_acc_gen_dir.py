import csv

headings = ['\\multirow{6}{*}{OpenMax}  & \\multirow{3}{*}{Near} &  Original & ',
            ' & & Baseline & ',
            '&  &  \\textbf{ours} & ',
            '& \\multirow{3}{*}{Far} & Original & ',
            ' & & Baseline & ',
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


def main():
    info = [
        {
            "csv_prefix": "/Users/changkunye/Projects/LOGS/OpenOOD/CIFAR10-20240811-Dir/",
            "dataset": 'cifar10',
        },
        {
            "csv_prefix": "/Users/changkunye/Projects/LOGS/OpenOOD/CIFAR100-20240811-Dir/",
            "dataset": 'cifar100',
        }
    ]

    estimate = 'MAP'
    ood_shift_types = ["Dir"]
    # ood_shift_type = "LTr"

    id_shifts = ['Original', 'LT100']

    # ood_shifts_names = [['Original', 'LT10', 'LT50', 'LT100'], ['Original', 'LT10r', 'LT50r', 'LT100r']]
    # ood_shifts_names = [['LT10', 'LT50', 'LT100'], ['LT10r', 'LT50r', 'LT100r']]
    # ood_shifts_names2 = [['LT10', 'LT100'], ['LT10r', 'LT100r']]
    ood_shifts_names = [['Dir1', 'Dir10']]
    ood_id_ratios = ["1.0", "0.1", "0.01"]
    methods = ["openmax", "mls", "react", "knn", "ash"]

    info_list = [[info[0]], [info[1]]]
    name_list = ['cifar10', 'cifar100']

    for data_info, name in zip(info_list, name_list):
        for id_shift in id_shifts:
            for ood_shift_type, ood_shift_name in zip(ood_shift_types, ood_shifts_names):
                write_path = "/Users/changkunye/Projects/LOGS/OpenOOD/all_tables/table_acc_%s-%s-%s-%s.txt" % \
                             (name, estimate, id_shift, ood_shift_type)

                print_table(data_info, methods, ood_shift_name, ood_id_ratios, id_shift, write_path)

    for id_shift in id_shifts:
        for ood_shift_type, ood_shift_name in zip(ood_shift_types, ood_shifts_names):
            write_path = "/Users/changkunye/Projects/LOGS/OpenOOD/all_tables/table_acc_%s-%s-%s-%s.txt" % \
                         ('cifar10100', estimate, id_shift, ood_shift_type)

            print_table(info, methods, ood_shift_name, ood_id_ratios, id_shift, write_path)

def main2():
    info = [
        {
            "csv_prefix": "/Users/changkunye/Projects/LOGS/OpenOOD/ImageNet200-20240809/",
            "dataset": 'imagenet200',
        },
    ]

    estimate = 'MAP'
    ood_shift_types = ["LT", "LTr"]
    # ood_shift_type = "LTr"

    # id_shifts = ['Original', 'LT100']
    id_shifts = ['Original']

    ood_shifts_names = [['Original', 'LT10', 'LT100'], ['Original', 'LT10r', 'LT100r']]
    # ood_shifts_names = [['LT10', 'LT50', 'LT100'], ['LT10r', 'LT50r', 'LT100r']]
    # ood_shifts_names2 = [['LT10', 'LT100'], ['LT10r', 'LT100r']]
    ood_id_ratios = ["1.0", "0.1", "0.01"]
    methods = ["openmax", "mls", "react", "knn", "ash"]

    for id_shift in id_shifts:
        for ood_shift_type, ood_shift_name in zip(ood_shift_types, ood_shifts_names):
            write_path = "/Users/changkunye/Projects/LOGS/OpenOOD/all_tables/table_acc_%s-%s-%s-%s.txt" % \
                         ('imagenet200', estimate, id_shift, ood_shift_type)

            print_table(info, methods, ood_shift_name, ood_id_ratios, id_shift, write_path, data_row=[3,7])

def format_data(x):
    # return x
    return '%.2f' % float(x)


def print_table(info, methods, ood_shifts, ood_id_ratios, id_shift, write_path, data_row=[3,8]):
    tab = []
    prec = 2

    better_count = 0
    count = 0
    for method in methods:
        if method == 'openmax':
            tab += [[], []]
        tab += [[], [], [], []]

        for p in info:
            for o_shift in ood_shifts:
                for oi_ratio in ood_id_ratios:
                    file_stamp = "test_ood-%s-%s-%s-%s-%s" % (p['dataset'], id_shift, o_shift, oi_ratio, method)
                    file_path = p["csv_prefix"] + file_stamp + ".csv"
                    print(file_path)
                    with open(file_path) as f:
                        reader = csv.reader(f, delimiter=',')
                        j = 0
                        for row in reader:
                            if j == data_row[0]:
                                if method != 'openmax':
                                    # print(row[10].split(' '), row[13].split(' '))
                                    d0 = row[10].split(' ')
                                    # print("$%s_{\\scaleto{\\pm %s}{3pt}}$" % (format_data(d0[0]), format_data(d0[-1])))
                                    tab[count].append("$%s_{\\scaleto{\\pm %s}{3pt}}$" %
                                                      (format_data(d0[0]), format_data(d0[-1])))

                                    d1 = row[13].split(' ')
                                    if round(float(d0[0]), prec) <= round(float(d1[0]), prec):
                                        better_count += 1
                                        tab[count + 1].append("\\cc $%s_{\\scaleto{\\pm %s}{3pt}}$" %
                                                              (format_data(d1[0]), format_data(d1[-1])))
                                    else:
                                        tab[count + 1].append("$%s_{\\scaleto{\\pm %s}{3pt}}$" %
                                                              (format_data(d1[0]), format_data(d1[-1])))
                                else:
                                    d = row[7].split(' ')
                                    # print("$%s_{\\scaleto{\\pm %s}{3pt}}$" % (format_data(d0[0]), format_data(d0[-1])))
                                    tab[count].append("$%s_{\\scaleto{\\pm %s}{3pt}}$" %
                                                      (format_data(d[0]), format_data(d[-1])))

                                    # print(row[10].split(' '), row[13].split(' '))
                                    d0 = row[10].split(' ')
                                    # print("$%s_{\\scaleto{\\pm %s}{3pt}}$" % (format_data(d0[0]), format_data(d0[-1])))
                                    tab[count + 1].append("$%s_{\\scaleto{\\pm %s}{3pt}}$" %
                                                          (format_data(d0[0]), format_data(d0[-1])))

                                    d1 = row[13].split(' ')
                                    if round(max(float(d0[0]), float(d[0])), prec) <= round(float(d1[0]), prec):
                                        better_count += 1
                                        tab[count + 2].append("\\cc $%s_{\\scaleto{\\pm %s}{3pt}}$" %
                                                              (format_data(d1[0]), format_data(d1[-1])))
                                    else:
                                        tab[count + 2].append("$%s_{\\scaleto{\\pm %s}{3pt}}$" %
                                                              (format_data(d1[0]), format_data(d1[-1])))
                            elif j == data_row[1]:
                                if method != 'openmax':
                                    # print(row[10].split(' '), row[13].split(' '))
                                    d0 = row[10].split(' ')
                                    # print("$%s_{\\scaleto{\\pm %s}{3pt}}$" % (format_data(d0[0]), format_data(d0[-1])))
                                    tab[count + 2].append("$%s_{\\scaleto{\\pm %s}{3pt}}$" %
                                                          (format_data(d0[0]), format_data(d0[-1])))

                                    d1 = row[13].split(' ')
                                    if round(float(d0[0]), prec) <= round(float(d1[0]), prec):
                                        better_count += 1
                                        tab[count + 3].append("\\cc $%s_{\\scaleto{\\pm %s}{3pt}}$" %
                                                              (format_data(d1[0]), format_data(d1[-1])))
                                    else:
                                        tab[count + 3].append("$%s_{\\scaleto{\\pm %s}{3pt}}$" %
                                                              (format_data(d1[0]), format_data(d1[-1])))
                                else:
                                    d = row[7].split(' ')
                                    # print("$%s_{\\scaleto{\\pm %s}{3pt}}$" % (format_data(d0[0]), format_data(d0[-1])))
                                    tab[count + 3].append("$%s_{\\scaleto{\\pm %s}{3pt}}$" %
                                                          (format_data(d[0]), format_data(d[-1])))

                                    # print(row[10].split(' '), row[13].split(' '))
                                    d0 = row[10].split(' ')
                                    # print("$%s_{\\scaleto{\\pm %s}{3pt}}$" % (format_data(d0[0]), format_data(d0[-1])))
                                    tab[count + 4].append("$%s_{\\scaleto{\\pm %s}{3pt}}$" %
                                                          (format_data(d0[0]), format_data(d0[-1])))

                                    d1 = row[13].split(' ')
                                    if round(max(float(d0[0]), float(d[0])), prec) <= round(float(d1[0]), prec):
                                        better_count += 1
                                        tab[count + 5].append("\\cc $%s_{\\scaleto{\\pm %s}{3pt}}$" %
                                                              (format_data(d1[0]), format_data(d1[-1])))
                                    else:
                                        tab[count + 5].append("$%s_{\\scaleto{\\pm %s}{3pt}}$" %
                                                              (format_data(d1[0]), format_data(d1[-1])))

                            j += 1

        if method == 'openmax':
            count += 6
        else:
            count += 4

        print("SOTA num %i" % better_count)

    with open(write_path, 'w') as f:
        for i, line in enumerate(tab):
            # print(line)
            print(len(line))
            if i in [5, 9, 13, 17, 21, 25]:
                f.write(headings[i] + " & ".join(line) + "\\\\ \\hline \n")
            elif i in [2, 7, 11, 15, 19, 23]:
                f.write(headings[i] + " & ".join(line) + "\\\\ \\cline{2-%i} \n" % (len(line) + 3))
            else:
                f.write(headings[i] + " & ".join(line) + "\\\\ \n")


if __name__ == '__main__':
    main()
    # main2()