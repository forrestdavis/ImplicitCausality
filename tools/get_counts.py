
data = '../results/Reading_Time_surp.csv'

with open(data) as f:

    header = f.readline().strip().split(',')
    print(header)


    surps = []
    count = 0
    for line in f:

        line = line.strip().split(',')

        #has objectIC
        if line[3] != '0':

            print(line[2], line[5])
            surps.append(float(line[5]))

        count += 1


sg_high = 0
sg_low = 0
pl_low = 0
pl_high = 0

isHIGH = 0
isLOW = 0

for i, surp in enumerate(surps):

    if i%4 == 0:
        sg_high = surp
    elif i%4 == 1:
        sg_low = surp
    elif i%4 == 2:
        pl_low = surp
    else:
        pl_high = surp
        if sg_high - sg_low > 0:
            isLOW += 1
        else:
            isHIGH += 1
        if pl_high - pl_low > 0:
            isLOW += 1
        else:
            isHIGH += 1
        sg_high = 0
        sg_low = 0
        pl_low = 0
        pl_high = 0

print(isHIGH, isLOW, isHIGH/(isHIGH+isLOW))

