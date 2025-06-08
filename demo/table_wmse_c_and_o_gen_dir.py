import csv

headings = ['\\multicolumn{2}{c|}{\multirow{2}{*}{BBSE}}  & Near & ',
            '\\multicolumn{2}{c|}{} & Far &  ',
            '\\multicolumn{2}{c|}{\multirow{2}{*}{MLLS}}  & Near & ',
            '\\multicolumn{2}{c|}{} & Far &  ',
            '\\multicolumn{2}{c|}{\multirow{2}{*}{RLLS}}  & Near & ',
            '\\multicolumn{2}{c|}{} & Far &  ',
            '\\multicolumn{2}{c|}{\multirow{2}{*}{MAPLS}}  & Near & ',
            '\\multicolumn{2}{c|}{} & Far &  ',
            '\\multicolumn{3}{c||}{Baseline} & ',
            '\\multirow{10}{*}{\\textbf{ours}} & \\multirow{2}{*}{OpenMax}  &  Near &  ',
            '& & Far & ',
            '& \\multirow{2}{*}{MLS}  & Near & ',
            '& & Far & ',
            '& \\multirow{2}{*}{ReAct}  &  Near & ',
            '& & Far & ',
            '& \\multirow{2}{*}{KNN}  &  Near &  ',
            '& & Far & ',
            '& \\multirow{2}{*}{Ash}  &  Near & ',
            '& & Far & ',
            ]


def main():
    info = [
        {
            "csv_prefix": "/Users/changkunye/Projects/LOGS/OpenOOD/CIFAR10-20241113-dir/",
            "dataset": 'cifar10',
        },
        {
            "csv_prefix": "/Users/changkunye/Projects/LOGS/OpenOOD/CIFAR100-20241113-dir/",
            "dataset": 'cifar100',
        }
    ]

    estimate = 'MAP'
    ood_shift_types = ["Dir"]
    # ood_shift_type = "LTr"

    # id_shifts = ['Original', 'LT100']
    id_shifts = ['Original']

    # ood_shifts_names = [['Original', 'LT10', 'LT50', 'LT100'], ['Original', 'LT10r', 'LT50r', 'LT100r']]
    # ood_shifts_names = [['LT10', 'LT50', 'LT100'], ['LT10r', 'LT50r', 'LT100r']]
    # ood_shifts_names2 = [['LT10', 'LT100'], ['LT10r', 'LT100r']]
    ood_shifts_names = [['Dir1', 'Dir10']]
    ood_shifts_names2 = ood_shifts_names
    ood_id_ratios = ["1.0", "0.1", "0.01"]
    methods = ["openmax", "mls", "react", "knn", "ash"]

    info_list = [[info[0]], [info[1]]]
    name_list = ['cifar10', 'cifar100']

    for data_info, name in zip(info_list, name_list):
        for id_shift in id_shifts:
            for ood_shift_type, ood_shift_name in zip(ood_shift_types, ood_shifts_names):
                write_path = "/Users/changkunye/Projects/LOGS/OpenOOD/all_tables/table_co_wmse_%s-%s-%s-%s.txt" % \
                             (name, estimate, id_shift, ood_shift_type)

                print_table(data_info, methods, ood_shift_name, ood_id_ratios, id_shift, write_path)

    for id_shift in id_shifts:
        for ood_shift_type, ood_shift_name in zip(ood_shift_types, ood_shifts_names2):
            write_path = "/Users/changkunye/Projects/LOGS/OpenOOD/all_tables/table_co_wmse_%s-%s-%s-%s.txt" % \
                         ('cifar10100', estimate, id_shift, ood_shift_type)

            print_table(info, methods, ood_shift_name, ood_id_ratios, id_shift, write_path)



def main2():
    info = [
        {
            "csv_prefix": "/Users/changkunye/Projects/LOGS/OpenOOD/ImageNet200-20241113-dir/",
            "dataset": 'imagenet200',
        },
    ]

    estimate = 'MAP'
    # ood_shift_type = "LTr"
    ood_shift_types = ["Dir"]

    # id_shifts = ['Original', 'LT100']
    id_shifts = ['Original']

    # ood_shifts_names = [['Original', 'LT10', 'LT100'], ['Original', 'LT10r', 'LT100r']]
    # ood_shifts_names = [['LT10', 'LT50', 'LT100'], ['LT10r', 'LT50r', 'LT100r']]
    ood_shifts_names = [['Dir1', 'Dir10']]
    ood_shifts_names2 = ood_shifts_names
    ood_id_ratios = ["1.0", "0.1", "0.01"]
    methods = ["openmax", "mls", "react", "knn", "ash"]

    for id_shift in id_shifts:
        for ood_shift_type, ood_shift_name in zip(ood_shift_types, ood_shifts_names):
            write_path = "/Users/changkunye/Projects/LOGS/OpenOOD/all_tables/table_co_wmse_%s-%s-%s-%s.txt" % \
                         ('imagenet200', estimate, id_shift, ood_shift_type)

            print_table(info, methods, ood_shift_name, ood_id_ratios, id_shift, write_path, data_row=[3,7])

def format_data(x):
    # return x
    return '%.3f' % float(x)


def print_table(info, methods, ood_shifts, ood_id_ratios, id_shift, write_path, data_row=[3,8], all_rows=[19,24]):
    tab = []
    prec = 4

    better_count = 0
    best_count = 0
    count = 8
    tab = [[] for _ in headings]
    for method in methods:
        for i in range(9):
            tab[i] = []
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
                                # print(row[10].split(' '), row[13].split(' '))
                                d0 = row[all_rows[0]].split(' ')
                                tab[8].append("$%s_{\\scaleto{\\pm %s}{3pt}}$" %
                                                  (format_data(d0[0]), format_data(d0[-1])))

                                for m,n in enumerate(range(all_rows[0] + 1, all_rows[1])):
                                    d = row[n].split(' ')
                                    # print("$%s_{\\scaleto{\\pm %s}{3pt}}$" % (format_data(d[0]), format_data(d[-1])))
                                    tab[2 * m].append("$%s_{\\scaleto{\\pm %s}{3pt}}$" %
                                                      (format_data(d[0]), format_data(d[-1])))
                                    if m == 0:
                                        d_min = round(float(d[0]), prec)
                                    else:
                                        tmp = round(float(d[0]), prec)
                                        d_min = tmp if tmp < d_min else d_min

                                d1 = row[all_rows[1]].split(' ')
                                if round(float(d0[0]), prec) > round(float(d1[0]), prec):
                                    better_count += 1
                                    if round(float(d1[0]), prec) < d_min:
                                        best_count += 1
                                        tab[count + 1].append("\\cc $\\mathbf{%s}_{\\scaleto{\\pm %s}{3pt}}$" %
                                                              (format_data(d1[0]), format_data(d1[-1])))
                                    else:
                                        tab[count + 1].append("\\cc $%s_{\\scaleto{\\pm %s}{3pt}}$" %
                                                              (format_data(d1[0]), format_data(d1[-1])))
                                    # tab[count + 1].append("$%s$" % (format_data(d1[0])))
                                else:
                                    tab[count + 1].append("$%s_{\\scaleto{\\pm %s}{3pt}}$" %
                                                          (format_data(d1[0]), format_data(d1[-1])))
                                    # tab[count + 1].append("$%s$" % (format_data(d1[0])))

                            elif j == data_row[1]:
                                # print(row[10].split(' '), row[13].split(' '))
                                d0 = row[all_rows[0]].split(' ')

                                for m,n in enumerate(range(all_rows[0] + 1, all_rows[1])):
                                    d = row[n].split(' ')
                                    # print("$%s_{\\scaleto{\\pm %s}{3pt}}$" % (format_data(d[0]), format_data(d[-1])))
                                    tab[2 * m + 1].append("$%s_{\\scaleto{\\pm %s}{3pt}}$" %
                                                      (format_data(d[0]), format_data(d[-1])))
                                    if m == 0:
                                        d_min = round(float(d[0]), prec)
                                    else:
                                        tmp = round(float(d[0]), prec)
                                        d_min = tmp if tmp < d_min else d_min

                                d1 = row[all_rows[1]].split(' ')
                                if round(float(d0[0]), prec) > round(float(d1[0]), prec):
                                    better_count += 1
                                    if round(float(d1[0]), prec) < d_min:
                                        best_count += 1
                                        tab[count + 2].append("\\cc $\\mathbf{%s}_{\\scaleto{\\pm %s}{3pt}}$" %
                                                              (format_data(d1[0]), format_data(d1[-1])))
                                    else:
                                        tab[count + 2].append("\\cc $%s_{\\scaleto{\\pm %s}{3pt}}$" %
                                                              (format_data(d1[0]), format_data(d1[-1])))
                                    # tab[count + 3].append("$%s$" % (format_data(d1[0])))
                                else:
                                    tab[count + 2].append("$%s_{\\scaleto{\\pm %s}{3pt}}$" %
                                                          (format_data(d1[0]), format_data(d1[-1])))
                                    # tab[count + 3].append("$%s$" % (format_data(d1[0])))


                            j += 1

        count += 2

    print("SOTA num %i, Best num %i" % (better_count, best_count))

    with open(write_path, 'w') as f:
        f.write("\\multicolumn{9}{c}{Closed Set Label Shift estimation models} \\\\ \\hline\\hline \n")
        for i, line in enumerate(tab):
            print(len(line))
            if i == 8:
                f.write("\\hline \n")
                f.write("\\multicolumn{9}{c}{Open Set Label Shift estimation models} \\\\ \\hline\\hline \n")

            if (i < 8) and (i % 2 == 1):
                f.write(headings[i] + " & ".join(line) + "\\\\ \\hline \n")
            elif (8 < i < (len(tab) - 1)) and (i % 2 == 0):
                f.write(headings[i] + " & ".join(line) + "\\\\ \\cline{2-9}\n")
            elif i == 8:
                f.write(headings[i] + " & ".join(line) + "\\\\ \\hline \n")
            else:
                f.write(headings[i] + " & ".join(line) + "\\\\ \n")

if __name__ == "__main__":
    main()
    main2()