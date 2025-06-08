import csv

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

method_names = {
    'b wmse':       "Baseline",
    'bbse wmse':    "BBSE",
    'mlls wmse':    "MLLS",
    'rlls wmse':    "RLLS",
    'mapls wmse':   "MAPLS",
    'ours wmse':    r"\textbf{ours}",
}


def main():
    info = [
        {
            "csv_prefix": "/Users/changkunye/Projects/LOGS/OpenOOD/CIFAR100-20240814-closed_set_ablation2/",
            "dataset": 'cifar100',
        },
        {
            "csv_prefix": "/Users/changkunye/Projects/LOGS/OpenOOD/ImageNet200-20240814-closed_set_ablation/",
            "dataset": 'imagenet200',
        }
    ]

    estimate = 'MAP'
    # ood_shift_types = ["LT", "LTr"]
    # ood_shift_type = "LTr"
    ood_shift_types = ["LT"]

    id_shifts = ['Original']

    # ood_shifts_names = [['Origainal', 'LT10', 'LT50', 'LT100'], ['Original', 'LT10r', 'LT50r', 'LT100r']]
    ood_shifts_names = [['LT10', 'LT50', 'LT100'], ['LT10r', 'LT50r', 'LT100r']]
    ood_shifts_names2 = [['LT10', 'LT100'], ['LT10r', 'LT100r']]
    ood_id_ratios = ["1.0", "0.1", "0.01"]
    methods = ["openmax"]

    for id_shift in id_shifts:
        for ood_shift_type, ood_shift_name in zip(ood_shift_types, ood_shifts_names2):
            write_path = "/Users/changkunye/Projects/LOGS/OpenOOD/all_tables/table_wmse_%s-%s-%s-%s.txt" % \
                         ('compare', estimate, id_shift, ood_shift_type)

            print_table(info, methods, ood_shift_name, ood_id_ratios, id_shift, write_path)



def format_data(x):
    # return x
    return '%.3f' % float(x)


def print_table(info, methods, ood_shifts, ood_id_ratios, id_shift, write_path,
                data_rows=([3,8],[3,7]), data_cols=([19,25], [19,25])):
    tab = []
    prec = 4
    previous_near = {}
    previous_far = {}
    for k,v in method_names.items():
        previous_near[v] = []
        previous_far[v] = []

    better_count = 0
    count = 0
    for method in methods:
        tab += [[], [], [], []]

        for data_row, data_col, p in zip(data_rows, data_cols, info):
            for o_shift in ood_shifts:
                for oi_ratio in ood_id_ratios:
                    file_stamp = "test_ood-%s-%s-%s-%s-%s" % (p['dataset'], id_shift, o_shift, oi_ratio, method)
                    file_path = p["csv_prefix"] + file_stamp + ".csv"
                    print(file_path)
                    with open(file_path) as f:
                        reader = csv.reader(f, delimiter=',')
                        j = 0
                        for row in reader:
                            # print(len(row), row[0])
                            if j == 0:
                                m_names = row
                            if row[0] == 'nearood':
                                for k in range(data_col[0],data_col[1]):
                                    print(m_names[k])
                                    d0 = row[k].split(' ')
                                    print("$%s_{\\scaleto{\\pm %s}{3pt}}$" % (format_data(d0[0]), format_data(d0[-1])))
                                    previous_near[method_names[m_names[k]]].append("$%s_{\\scaleto{\\pm %s}{3pt}}$" %
                                                  (format_data(d0[0]), format_data(d0[-1])))

                            elif row[0] == 'farood':
                                for k in range(data_col[0],data_col[1]):
                                    d0 = row[k].split(' ')
                                    print("$%s_{\\scaleto{\\pm %s}{3pt}}$" % (format_data(d0[0]), format_data(d0[-1])))
                                    previous_far[method_names[m_names[k]]].append("$%s_{\\scaleto{\\pm %s}{3pt}}$" %
                                                  (format_data(d0[0]), format_data(d0[-1])))


                            j += 1

        count += 4

    print("SOTA num %i" % better_count)

    with open(write_path, 'w') as f:
        for k, v in method_names.items():
            # print(line)
            print(" \\multicolumn{2}{c|}{\\multirow{2}{*}{%s}} & Near & " % v)
            print(' & '.join(previous_near[v]) + '\\\\ \n')
            print(" \\multicolumn{2}{c|}{} & Far & ")
            print(' & '.join(previous_far[v]) + "\\\\ \\cline{2-15} \n")
            # if (i+1) % 4 == 0:
            #     f.write(headings[i] + " & ".join(line) + "\\\\ \\hline \n")
            # elif (i+1) % 2 == 0:
            #     f.write(headings[i] + " & ".join(line) + "\\\\ \\cline{2-%i} \n" % (len(line) + 3))
            # else:
            #     f.write(headings[i] + " & ".join(line) + "\\\\ \n")


if __name__ == "__main__":
    main()
    # main2()