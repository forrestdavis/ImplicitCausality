with open("IC_match_tf_results.csv", 'r') as f:
    heading = []
    t_idxs = []

    header = f.readline().strip().split(',')
    for x in range(len(header)):
        if 'pval' not in header[x]:
            heading.append(header[x])
            t_idxs.append(x)

    with open("IC_match_tf_rho_results.csv", 'w') as o:

        o.write(','.join(heading)+'\n')

        for line in f:
            line = line.strip().split(',')
            out_str = ''
            for x in range(len(line)):
                if x in t_idxs:
                    out_str += line[x] + ','
            out_str = out_str[:-1] + '\n'
            o.write(out_str)

'''
with open('IC_match_tf_results.csv', 'r') as f:
    with open('IC_match_tf_rho_results.csv', 'w') as o:
        for line in f:
            line = line.strip().split(',')
            out_str = ''
            for x in range(len(line)):
                if x % 2 == 0:
                    out_str += line[x]+','
            out_str = out_str[:-1]+'\n'
            o.write(out_str):
'''
