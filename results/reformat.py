#RT LSTM
with open("IC_match_gpt_SIM.csv", 'r') as f:

    header = f.readline().strip().split(',')
    print(header)

    heading = header[:7]
    heading.append('layer')
    heading.append('NP')
    heading.append('sim')
    out_str = ','.join(heading)+'\n'

    for line in f:
        line = line.strip().split(',')
        repeat = line[:7]
        simHigh=0
        for x in range(7, len(line)):
            '''
            if 'avg' not in header[x]:
                continue
            layer = str(int(header[x].split('_')[3])+1)
            '''
            layer = str(int(header[x].split('_')[2])+1)
            if 'HIGH' in header[x]:
                out = repeat + [layer, '1', line[x]]
                out_str += ','.join(out)+'\n'
            else:
                out = repeat + [layer, '2', line[x]]
                out_str += ','.join(out)+'\n'
            
    with open("gpt_pronoun_flat_SIM.csv", 'w') as o:
        o.write(out_str)
'''
#RT LSTM
with open("Reading_Time_LSTM_were_SIM.csv", 'r') as f:

    header = f.readline().strip().split(',')
    print(header)

    heading = header[:6]
    heading.append('layer')
    heading.append('NP')
    heading.append('sim')
    out_str = ','.join(heading)+'\n'

    for line in f:
        line = line.strip().split(',')
        repeat = line[:6]
        simHigh=0
        for x in range(6, len(line)):
            if 'avg' not in header[x]:
                continue
            layer = str(int(header[x].split('_')[3])+1)
            if 'HIGH' in header[x]:
                out = repeat + [layer, '1', line[x]]
                out_str += ','.join(out)+'\n'
            else:
                out = repeat + [layer, '2', line[x]]
                out_str += ','.join(out)+'\n'
            
    with open("LSTM_were_flat_SIM.csv", 'w') as o:
        o.write(out_str)
'''
'''
#RT tf
with open("Reading_Time_gpt_who_SIM.csv", 'r') as f:

    header = f.readline().strip().split(',')
    print(header)

    heading = header[:6]
    heading.append('layer')
    heading.append('NP')
    heading.append('sim')
    out_str = ','.join(heading)+'\n'

    for line in f:
        line = line.strip().split(',')
        repeat = line[:6]
        simHigh=0
        for x in range(6, len(line)):
            layer = str(int(header[x].split('_')[2])+1)
            if x % 2 == 0:
                out = repeat + [layer, '1', line[x]]
                out_str += ','.join(out)+'\n'
            else:
                out = repeat + [layer, '2', line[x]]
                out_str += ','.join(out)+'\n'
            
    with open("gpt_who_flat_SIM.csv", 'w') as o:
        o.write(out_str)
'''
'''
#read in rt lstm surps
with open("Reading_Time_surp.csv", 'r') as f:
    header = f.readline().strip().split(',')
    heading = header[:6]

    heading.append('model')
    heading.append('lstm_surp')
    out_str = ','.join(heading)+'\n'

    for line in f:
        line = line.strip().split(',')
        repeat = line[:6]

        for x in range(6, len(line)):
            label = header[x]
            if "LSTM" in label and 'avg' not in label:
                model = '_'.join(label.split('_')[:-1])
                out = repeat + [model]
                out.append(line[x])
                out_str += ','.join(out) + '\n'

    with open("Reading_Time_LSTM_surp.csv", 'w') as o:
        o.write(out_str)

'''
'''
#read in gpt-2
with open("IC_match_gpt_SIM.csv", 'r') as f:

    header = f.readline().strip().split(',')
    print(header)

    heading = header[:7]
    heading.append('layer')
    heading.append('simHIGH')
    heading.append('simLOW')
    out_str = ','.join(heading)+'\n'

    for line in f:
        line = line.strip().split(',')
        repeat = line[:7]
        simHigh=0
        for x in range(7, len(line)):
            layer = header[x].split('_')[2]
            if x % 2 == 1:
                simHigh=line[x]
            else:
                out = repeat + [layer]
                out.append(simHigh)
                out.append(line[x])
                out_str += ','.join(out)+'\n'
            
    with open("gpt_pronoun_SIM.csv", 'w') as o:
        o.write(out_str)
'''
'''
with open("IC_match_tf_SIM.csv", 'r') as f:

    header = f.readline().strip().split(',')
    print(header)

    heading = header[:7]
    heading.append('layer')
    heading.append('simHigh')
    heading.append('simLOW')
    out_str = ','.join(heading)+'\n'

    for line in f:
        line = line.strip().split(',')
        repeat = line[:7]
        simHigh=0
        for x in range(7, len(line)):
            layer = header[x].split('_')[2]
            if x % 2 == 1:
                simHigh=line[x]
            else:
                out = repeat + [layer]
                out.append(simHigh)
                out.append(line[x])
                out_str += ','.join(out)+'\n'
            
    with open("tf_pronoun_SIM.csv", 'w') as o:
        o.write(out_str)
'''


'''
with open("IC_match_LSTM_SIM.csv", 'r') as f:

    header = f.readline().strip().split(',')

    heading = header[:7]
    heading.append('layer')
    heading.append('simHIGH')
    heading.append('simLOW')
    out_str = ','.join(heading)+'\n'

    print(heading)
    for line in f:
        line = line.strip().split(',')
        repeat = line[:7]
        simHigh=0
        for x in range(7, 11):
            if x == 7 or x == 8:
                layer = '0'
            else:
                layer = '1'
            if x % 2 == 1:
                simHigh=line[x]
            else:
                out = repeat + [layer]
                out.append(simHigh)
                out.append(line[x])
                out_str += ','.join(out)+'\n'

    with open("LSTM_pronoun_SIM.csv", 'w') as o:
        o.write(out_str)
'''
