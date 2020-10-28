## Story completion experiment
names = {
        1: 'The man', 
        #2: 'The man', 
        3: 'The man', 
        4: 'The man', 
        5: 'The man', 
        6: 'The woman', 
        #7: 'The woman', 
        #8: 'The woman', 
        9: 'The woman', 
        #10: 'The woman', 
        11: 'The man', 
        #12: 'The man', 
        13: 'The man', 
        14: 'The man', 
        #15: 'The woman', 
        16: 'The man', 
        17: 'The man', 
        18: 'The man', 
        19: 'The man', 
        20: 'The woman', 
        #21: 'The man'
        }

verbs = {
    1: ['admires', 'works with'],
        #2: ['adores', 'smiles at'],
        3: ['blamed', 'noticed'],
        4: ['complimented', 'met'],
        5: ['congratulated', 'visited'],
        6: ['criticized', 'talked to'],
        #7: ['despises', 'babysits'],
        #8: ['detests', 'looks like'],
        9: ['dislikes', 'watches'],
        #10: ['insulted', 'chatted with'],
        11: ['likes', 'resembles'],
        #12: ['pities', 'hires'],
        13: ['praised', 'videotaped'],
        14: ['punished', 'saw'],
        #15: ['resents', 'knows'],
        16: ['scolded', 'recognized'],
        17: ['rewarded', 'inspected'],
        18: ['ridiculed', 'counted'],
        19: ['thanked', 'interviewed'],
        20: ['values', 'lives next to'],
        #21: ['worships', 'listens to']
        }

NP1 = {
    1: ['agent', 'agents'], 
        #2: ['secretary', 'secretaries'],
        3: ['friend', 'friends'],
        4: ['guest', 'guests'],
        5: ['teacher', 'teachers'],
        6: ['leader', 'leaders'],
        #7: ['child', 'children'],
        #8: ['father', 'fathers'],
        9: ['little girl', 'little girls'],
        #10: ['gardener', 'gardeners'],
        11: ['captain', 'captains'],
        #12: ['bodyguard', 'bodyguards'],
        13: ['assistant', 'assistant'],
        14: ['accountant', 'accountants'],
        #15: ['doctor', 'doctors'],
        16: ['landlady', 'landladies'],
        17: ['servant', 'servants'],
        18: ['fan', 'fans'],
        19: ['representative', 'representatives'],
        20: ['surgeon', 'surgeons'],
        #21: ['coach', 'coaches']
        }

NP2 = {
    #1: ['rockstar', 'rockstars'], 
    1: ['rocker', 'rockers'], 
        #2: ['lawyer', 'lawyers'],
        3: ['athlete', 'athletes'],
        4: ['bride', 'brides'],
        5: ['second grader', 'second graders'],
        6: ['activist', 'activists'],
        #7: ['jazz musician', 'jazz musicians'],
        #8: ['student', 'students'],
        9: ['neighbor', 'neighbors'],
        #10: ['millionaire', 'millionaires'],
        11: ['old sailor', 'old sailors'],
        #12: ['celebrity', 'celebrities'],
        #13: ['ceo', 'ceos'],
        13: ['boss', 'bosses'],
        14: ['businessman', 'businessman'],
        #15: ['supermodel', 'supermodels'],
        16: ['actor', 'actors'],
        17: ['dictator', 'dictator'],
        18: ['singer', 'singers'],
        19: ['employee', 'employees'],
        20: ['soldier', 'soldiers'],
        #21: ['cheerleader', 'cheerleaders']
        }

for key in names:
    name = names[key]
    IC = verbs[key][0]
    non_IC = verbs[key][1]
    NP1_sg = NP1[key][0]
    NP1_pl = NP1[key][1]
    NP2_sg = NP2[key][0]
    NP2_pl = NP2[key][1]

    sent1 = ' '.join([name, IC, 'the', NP1_sg, 'of the', NP2_pl, 'who was'])
    sent2 = ' '.join([name, IC, 'the', NP1_pl, 'of the', NP2_sg, 'who was'])
    sent3 = ' '.join([name, IC, 'the', NP1_sg, 'of the', NP2_pl, 'who were'])
    sent4 = ' '.join([name, IC, 'the', NP1_pl, 'of the', NP2_sg, 'who were'])

    sent5 = ' '.join([name, non_IC, 'the', NP1_sg, 'of the', NP2_pl, 'who was'])
    sent6 = ' '.join([name, non_IC, 'the', NP1_pl, 'of the', NP2_sg, 'who was'])
    sent7 = ' '.join([name, non_IC, 'the', NP1_sg, 'of the', NP2_pl, 'who were'])
    sent8 = ' '.join([name, non_IC, 'the', NP1_pl, 'of the', NP2_sg, 'who were'])

    sent9 = ' '.join([name, IC, 'the', NP2_sg, 'of the', NP1_pl, 'who was'])
    sent10 = ' '.join([name, IC, 'the', NP2_pl, 'of the', NP1_sg, 'who was'])
    sent11 = ' '.join([name, IC, 'the', NP2_sg, 'of the', NP1_pl, 'who were'])
    sent12 = ' '.join([name, IC, 'the', NP2_pl, 'of the', NP1_sg, 'who were'])

    sent13 = ' '.join([name, non_IC, 'the', NP2_sg, 'of the', NP1_pl, 'who was'])
    sent14 = ' '.join([name, non_IC, 'the', NP2_pl, 'of the', NP1_sg, 'who was'])
    sent15 = ' '.join([name, non_IC, 'the', NP2_sg, 'of the', NP1_pl, 'who were'])
    sent16 = ' '.join([name, non_IC, 'the', NP2_pl, 'of the', NP1_sg, 'who were'])

    '''
    print(','.join(['story', str(key), sent1, '1', 'sg', '1']))
    print(','.join(['story', str(key), sent2, '1', 'sg', '0']))
    #print(','.join(['story', str(key), sent3, '1', 'pl', '0']))
    #print(','.join(['story', str(key), sent4, '1', 'pl', '1']))
    print(','.join(['story', str(key), sent5, '0', 'sg', '1']))
    print(','.join(['story', str(key), sent6, '0', 'sg', '0']))
    #print(','.join(['story', str(key), sent7, '0', 'pl', '0']))
    #print(','.join(['story', str(key), sent8, '0', 'pl', '1']))

    print(','.join(['story', str(key), sent9, '1', 'sg', '1']))
    print(','.join(['story', str(key), sent10, '1', 'sg', '0']))
    #print(','.join(['story', str(key), sent11, '1', 'pl', '0']))
    #print(','.join(['story', str(key), sent12, '1', 'pl', '1']))
    print(','.join(['story', str(key), sent13, '0', 'sg', '1']))
    print(','.join(['story', str(key), sent14, '0', 'sg', '0']))
    #print(','.join(['story', str(key), sent15, '0', 'pl', '0']))
    #print(','.join(['story', str(key), sent16, '0', 'pl', '1']))
    '''

#Reading time
names = {
        1: 'The woman', 
        #2: 'The man', 
        3: 'The woman', 
        4: 'The man', 
        #5: 'The woman', 
        6: 'The man', 
        #7: 'The woman', 
        8: 'The man', 
        #9: 'The woman', 
        10: 'The man', 
        11: 'The woman', 
        #12: 'The man', 
        13: 'The woman', 
        14: 'The man', 
        #15: 'The woman', 
        16: 'The man', 
        #17: 'The woman', 
        18: 'The man', 
        #19: 'The woman', 
        20: 'The man'
        }

verbs = {
        1: ['scolded', 'studied with'],
        #2: ['stared at', 'lived next to'],
        3: ['assisted', 'joked with'],
        4: ['trusted', 'stood near'],
        #5: ['corrected', 'gossiped with'],
        6: ['comforted', 'greeted'],
        #7: ['envies', 'knows'],
        8: ['valued', 'recognized'],
        #9: ['fears', 'jogs with'],
        10: ['noticed', 'resembled'],
        11: ['praised', 'met'],
        #12: ['hates', 'carpools with'],
        13: ['blamed', 'waited with'],
        14: ['helped', 'ran into'],
        #15: ['reproached', 'worked with'],
        16: ['pacified', 'visited'],
        #17: ['detests', 'babysits'],
        18: ['thanked', 'talked to'],
        #19: ['congratulated', 'chatted with'],
        20: ['mocked', 'counted']
        }

NP1 = {
        1: ['chef', 'chefs'], 
        #2: ['teacher', 'teachers'],
        3: ['maid', 'maids'],
        4: ['captain', 'captains'],
        #5: ['secretary', 'secretaries'],
        6: ['leader', 'leaders'],
        #7: ['manager', 'managers'],
        8: ['daughter', 'daughters'],
        #9: ['uncle', 'uncles'],
        10: ['representative', 'representatives'],
        11: ['gardener', 'gardeners'],
        #12: ['cousin', 'cousins'],
        13: ['niece', 'nieces'],
        14: ['brother', 'brothers'],
        #15: ['doctors', 'doctors'],
        16: ['associate', 'associates'],
        #17: ['child', 'children'],
        18: ['servant', 'servants'],
        #19: ['bodyguard', 'bodyguards'],
        20: ['fan', 'fans']
        }

NP2 = {1: ['aristocrat', 'aristocrats'], 
        #2: ['second grader', 'second graders'],
        3: ['executive', 'executives'],
        4: ['sailor', 'sailors'],
        #5: ['lawyer', 'lawyers'],
        6: ['activist', 'activists'],
        #7: ['cashier', 'cashiers'],
        8: ['shopkeeper', 'shopkeepers'],
        #9: ['toddler', 'toddlers'],
        10: ['employee', 'employees'],
        11: ['millionaire', 'millionaires'],
        #12: ['accountant', 'accountants'],
        #13: ['florist', 'florists'],
        13: ['clerk', 'clerks'],
        14: ['athlete', 'athletes'],
        #15: ['supermodel', 'supermodels'],
        16: ['businessman', 'businessmen'],
        #17: ['musician', 'musicians'],
        18: ['dictator', 'dictators'],
        #19: ['celebrity', 'celebrities'],
        20: ['singer', 'singers']
        }

for key in names:
    name = names[key]
    IC = verbs[key][0]
    non_IC = verbs[key][1]
    NP1_sg = NP1[key][0]
    NP1_pl = NP1[key][1]
    NP2_sg = NP2[key][0]
    NP2_pl = NP2[key][1]

    sent1 = ' '.join([name, IC, 'the', NP1_sg, 'of the', NP2_pl, 'who was'])
    sent2 = ' '.join([name, IC, 'the', NP1_pl, 'of the', NP2_sg, 'who was'])
    sent3 = ' '.join([name, IC, 'the', NP1_sg, 'of the', NP2_pl, 'who were'])
    sent4 = ' '.join([name, IC, 'the', NP1_pl, 'of the', NP2_sg, 'who were'])

    sent5 = ' '.join([name, non_IC, 'the', NP1_sg, 'of the', NP2_pl, 'who was'])
    sent6 = ' '.join([name, non_IC, 'the', NP1_pl, 'of the', NP2_sg, 'who was'])
    sent7 = ' '.join([name, non_IC, 'the', NP1_sg, 'of the', NP2_pl, 'who were'])
    sent8 = ' '.join([name, non_IC, 'the', NP1_pl, 'of the', NP2_sg, 'who were'])

    sent9 = ' '.join([name, IC, 'the', NP2_sg, 'of the', NP1_pl, 'who was'])
    sent10 = ' '.join([name, IC, 'the', NP2_pl, 'of the', NP1_sg, 'who was'])
    sent11 = ' '.join([name, IC, 'the', NP2_sg, 'of the', NP1_pl, 'who were'])
    sent12 = ' '.join([name, IC, 'the', NP2_pl, 'of the', NP1_sg, 'who were'])

    sent13 = ' '.join([name, non_IC, 'the', NP2_sg, 'of the', NP1_pl, 'who was'])
    sent14 = ' '.join([name, non_IC, 'the', NP2_pl, 'of the', NP1_sg, 'who was'])
    sent15 = ' '.join([name, non_IC, 'the', NP2_sg, 'of the', NP1_pl, 'who were'])
    sent16 = ' '.join([name, non_IC, 'the', NP2_pl, 'of the', NP1_sg, 'who were'])

    '''
    print(','.join(['reading', str(key), sent1, '1', 'sg', '1']))
    print(','.join(['reading', str(key), sent2, '1', 'sg', '0']))
    print(','.join(['reading', str(key), sent3, '1', 'pl', '0']))
    print(','.join(['reading', str(key), sent4, '1', 'pl', '1']))
    print(','.join(['reading', str(key), sent5, '0', 'sg', '1']))
    print(','.join(['reading', str(key), sent6, '0', 'sg', '0']))
    print(','.join(['reading', str(key), sent7, '0', 'pl', '0']))
    print(','.join(['reading', str(key), sent8, '0', 'pl', '1']))

    print(','.join(['reading', str(key), sent9, '1', 'sg', '1']))
    print(','.join(['reading', str(key), sent10, '1', 'sg', '0']))
    print(','.join(['reading', str(key), sent11, '1', 'pl', '0']))
    print(','.join(['reading', str(key), sent12, '1', 'pl', '1']))
    print(','.join(['reading', str(key), sent13, '0', 'sg', '1']))
    print(','.join(['reading', str(key), sent14, '0', 'sg', '0']))
    print(','.join(['reading', str(key), sent15, '0', 'pl', '0']))
    print(','.join(['reading', str(key), sent16, '0', 'pl', '1']))
    '''

verbs = open('IC_verbs', 'r')
bias = open('IC_bias', 'r')

missing = set([])
with open("missing_IC", 'r') as f:
    for line in f:
        line = line.strip()
        missing.add(line)

data = {}
for line in verbs:
    line = line.strip()
    b = bias.readline().strip()
    #skip over out of vocab IC
    if line in missing or line.split(' ')[0] in missing:
        continue
    data[line] = b

verbs.close()
bias.close()

#gender mismatch
pairs = []
with open('gender_pairs', 'r') as f:
    for line in f:
        line = line.strip().split()
        pairs.append(line)

count = 0
for pair in pairs:
    for verb in data:
        bias = data[verb]
        sent0 = ' '.join(['The', pair[0], verb, 'the', pair[1], 'because he'])
        sent1 = ' '.join(['The', pair[0], verb, 'the', pair[1], 'because she'])
        sent2 = ' '.join(['The', pair[1], verb, 'the', pair[0], 'because he'])
        sent3 = ' '.join(['The', pair[1], verb, 'the', pair[0], 'because she'])
        '''
        print(','.join(['ic_mismatch', verb, str(count), sent0, bias, '1', 'm']) )
        print(','.join(['ic_mismatch', verb, str(count), sent1, bias, '0', 'f']) )
        print(','.join(['ic_mismatch', verb, str(count), sent2, bias, '0', 'm']) )
        print(','.join(['ic_mismatch', verb, str(count), sent3, bias, '1', 'f']) )
        '''
    count += 1

#gender match
pairs = []
with open('same_gender_pairs', 'r') as f:
    for line in f:
        line = line.strip().split()
        pairs.append(line)

count = 0
for x in range(len(pairs)):
    pair = pairs[x]
    if x %2 == 0:
        gender = 'm'
    else:
        gender = 'f'

    for verb in data:
        bias = data[verb]
        if gender == 'm':

            sent0 = ' '.join(['The', pair[0], verb, 'the', pair[1], 'because he'])
            sent1 = ' '.join(['The', pair[1], verb, 'the', pair[0], 'because he'])
        else:
            sent0 = ' '.join(['The', pair[0], verb, 'the', pair[1], 'because she'])
            sent1 = ' '.join(['The', pair[1], verb, 'the', pair[0], 'because she'])
        '''
        print(','.join(['ic_match', verb, str(count), sent0, bias, '0', gender]) )
        print(','.join(['ic_match', verb, str(count), sent1, bias, '0', gender]) )
        '''
    count += 1

#For BERT
verbs = open('es_IC_verbs', 'r')
#verbs = open('IC_verbs', 'r')
bias = open('es_IC_bias', 'r')
#bias = open('IC_bias', 'r')

missing = set([])
with open("missing_IC", 'r') as f:
    for line in f:
        line = line.strip()
        missing.add(line)

data = {}
for line in verbs:
    line = line.strip()
    b = bias.readline().strip()
    #skip over out of vocab IC
    if line in missing or line.split(' ')[0] in missing:
        continue
    data[line] = b

verbs.close()
bias.close()

#gender mismatch
pairs = []
with open('es_gender_pairs', 'r') as f:
#with open('gender_pairs', 'r') as f:
    for line in f:
        line = line.strip().split()
        pairs.append(line)

count = 0
'''
for pair in pairs:
    for verb in data:
        bias = data[verb]

        sent0 = ' '.join(['The', pair[0], verb, 'the', pair[1], 'because he was there.'])
        #sent0 = ' '.join(['el', pair[0], verb, 'a la', pair[1], 'porque [MASK] estaba allí.'])
        sent1 = ' '.join(['The', pair[0], verb, 'the', pair[1], 'because she was there.'])
        sent2 = ' '.join(['The', pair[1], verb, 'the', pair[0], 'because he was there.'])
        #sent2 = ' '.join(['la', pair[1], verb, 'al', pair[0], 'porque [MASK] estaba allí.'])
        sent3 = ' '.join(['The', pair[1], verb, 'the', pair[0], 'because she was there.'])

        print(','.join(['ic_mismatch', verb, str(count), sent0, bias, '1', 'm']) )
        print(','.join(['ic_mismatch', verb, str(count), sent1, bias, '0', 'f']) )
        print(','.join(['ic_mismatch', verb, str(count), sent2, bias, '0', 'm']) )
        print(','.join(['ic_mismatch', verb, str(count), sent3, bias, '1', 'f']) )
    count += 1
'''
'''
for pair in pairs:
    for verb in data:
        bias = data[verb]

        if verb.split(' ')[-1] == "de":
            sent0 = ' '.join(['el', pair[0], verb, 'la', pair[1], 'porque él estaba allí.'])
            #sent0 = ' '.join(['el', pair[0], verb, 'la', pair[1], 'porque [MASK] estaba allí.'])
            sent1 = ' '.join(['el', pair[0], verb, 'la', pair[1], 'porque ella estaba allí.'])
            verb_del = verb.replace(' de', ' del')
            sent2 = ' '.join(['la', pair[1], verb_del, pair[0], 'porque él estaba allí.'])
            #sent2 = ' '.join(['la', pair[1], verb_del, pair[0], 'porque [MASK] estaba allí.'])
            sent3 = ' '.join(['la', pair[1], verb_del, pair[0], 'porque ella estaba allí.'])

        elif verb.split(' ')[-1] == "en" or verb.split(' ')[-1] == "por":

            sent0 = ' '.join(['el', pair[0], verb, 'la', pair[1], 'porque él estaba allí.'])
            #sent0 = ' '.join(['el', pair[0], verb, 'la', pair[1], 'porque [MASK] estaba allí.'])
            sent1 = ' '.join(['el', pair[0], verb, 'la', pair[1], 'porque ella estaba allí.'])
            sent2 = ' '.join(['la', pair[1], verb, 'el', pair[0], 'porque él estaba allí.'])
            #sent2 = ' '.join(['la', pair[1], verb, 'el', pair[0], 'porque [MASK] estaba allí.'])
            sent3 = ' '.join(['la', pair[1], verb, 'el', pair[0], 'porque ella estaba allí.'])

        else:
            sent0 = ' '.join(['el', pair[0], verb, 'a la', pair[1], 'porque él estaba allí.'])
            #sent0 = ' '.join(['el', pair[0], verb, 'a la', pair[1], 'porque [MASK] estaba allí.'])
            sent1 = ' '.join(['el', pair[0], verb, 'a la', pair[1], 'porque ella estaba allí.'])
            sent2 = ' '.join(['la', pair[1], verb, 'al', pair[0], 'porque él estaba allí.'])
            #sent2 = ' '.join(['la', pair[1], verb, 'al', pair[0], 'porque [MASK] estaba allí.'])
            sent3 = ' '.join(['la', pair[1], verb, 'al', pair[0], 'porque ella estaba allí.'])

        print(','.join(['ic_mismatch', verb, str(count), sent0, bias, '1', 'm']) )
        print(','.join(['ic_mismatch', verb, str(count), sent1, bias, '0', 'f']) )
        print(','.join(['ic_mismatch', verb, str(count), sent2, bias, '0', 'm']) )
        print(','.join(['ic_mismatch', verb, str(count), sent3, bias, '1', 'f']) )
    count += 1
for pair in pairs:
    for verb in data:
        bias = data[verb]

        if verb.split(' ')[-1] == "de":
            sent0 = ' '.join(['el', pair[0], verb, 'la', pair[1], 'porque estaba [+male].'])
            #sent0 = ' '.join(['el', pair[0], verb, 'la', pair[1], 'porque estaba [MASK].'])
            sent1 = ' '.join(['el', pair[0], verb, 'la', pair[1], 'porque estaba [+female].'])
            verb_del = verb.replace(' de', ' del')
            sent2 = ' '.join(['la', pair[1], verb_del, pair[0], 'porque estaba [+male].'])
            #sent2 = ' '.join(['la', pair[1], verb_del, pair[0], 'porque estaba [MASK].'])
            sent3 = ' '.join(['la', pair[1], verb_del, pair[0], 'porque estaba [+female].'])

        elif verb.split(' ')[-1] == "en" or verb.split(' ')[-1] == "por":

            sent0 = ' '.join(['el', pair[0], verb, 'la', pair[1], 'porque estaba [+male].'])
            #sent0 = ' '.join(['el', pair[0], verb, 'la', pair[1], 'porque estaba [MASK].'])
            sent1 = ' '.join(['el', pair[0], verb, 'la', pair[1], 'porque estaba [+female].'])
            sent2 = ' '.join(['la', pair[1], verb, 'el', pair[0], 'porque estaba [+male].'])
            #sent2 = ' '.join(['la', pair[1], verb, 'el', pair[0], 'porque estaba [MASK].'])
            sent3 = ' '.join(['la', pair[1], verb, 'el', pair[0], 'porque estaba [+female].'])

        else:
            sent0 = ' '.join(['el', pair[0], verb, 'a la', pair[1], 'porque estaba [+male].'])
            #sent0 = ' '.join(['el', pair[0], verb, 'a la', pair[1], 'porque estaba [MASK].'])
            sent1 = ' '.join(['el', pair[0], verb, 'a la', pair[1], 'porque estaba [+female].'])
            sent2 = ' '.join(['la', pair[1], verb, 'al', pair[0], 'porque estaba [+male].'])
            #sent2 = ' '.join(['la', pair[1], verb, 'al', pair[0], 'porque estaba [MASK].'])
            sent3 = ' '.join(['la', pair[1], verb, 'al', pair[0], 'porque estaba [+female].'])

        print(','.join(['ic_mismatch', verb, str(count), sent0, bias, '1', 'm']) )
        print(','.join(['ic_mismatch', verb, str(count), sent1, bias, '0', 'f']) )
        print(','.join(['ic_mismatch', verb, str(count), sent2, bias, '0', 'm']) )
        print(','.join(['ic_mismatch', verb, str(count), sent3, bias, '1', 'f']) )
    count += 1
'''

'''
#gender match
pairs = []
with open('same_gender_pairs', 'r') as f:
    for line in f:
        line = line.strip().split()
        pairs.append(line)

count = 0
for x in range(len(pairs)):
    pair = pairs[x]
    if x %2 == 0:
        gender = 'm'
    else:
        gender = 'f'

    for verb in data:
        bias = data[verb]
        if gender == 'm':

            sent0 = ' '.join(['The', pair[0], verb, 'the', pair[1], 'because he'])
            sent1 = ' '.join(['The', pair[1], verb, 'the', pair[0], 'because he'])
        else:
            sent0 = ' '.join(['The', pair[0], verb, 'the', pair[1], 'because she'])
            sent1 = ' '.join(['The', pair[1], verb, 'the', pair[0], 'because she'])
        print(','.join(['ic_match', verb, str(count), sent0, bias, '0', gender]) )
        print(','.join(['ic_match', verb, str(count), sent1, bias, '0', gender]) )
    count += 1
'''

#For BERT Italian
verbs = open('it_IC_verbs', 'r')
bias = open('it_IC_bias', 'r')

data = {}
for line in verbs:
    line = line.strip()
    b = bias.readline().strip()
    #skip over out of vocab IC
    data[line] = b

verbs.close()
bias.close()

#gender mismatch
pairs = []
with open('it_gender_pairs', 'r') as f:
#with open('gender_pairs', 'r') as f:
    for line in f:
        line = line.strip().split()
        pairs.append(line)

count = 0
for pair in pairs:
    for verb in data:
        bias = data[verb]

        det_one = pair[0]
        noun_one = pair[1]
        det_two = pair[2]
        noun_two = pair[3]
        #context_l = 'a causa del tipo di persona che'
        #context_r = 'è.'
        context_l = 'perché è'
        context_r = '.'

        if verb.split(' ')[-1] == 'di':
            verb = ' '.join(verb.split(' ')[:-1])

            if "'" in det_one:
                subj = ''.join([det_one, noun_one])
            else:
                subj = ' '.join([det_one, noun_one])

            if det_two == 'la':
                prep = 'della'
            if det_two == "l'":
                prep = "dell'"

            if "'" in prep:
                obj = ''.join([prep, noun_two])
            else:
                obj = ' '.join([prep, noun_two])

            #sent0 = ' '.join([subj, verb, obj, context_l, 'egli', context_r])
            #sent0 = ' '.join([subj, verb, obj, context_l, '[MASK]', context_r])
            #sent1 = ' '.join([subj, verb, obj, context_l, 'ella', context_r])

            #sent0 = ' '.join([subj, verb, obj, context_l, '[+male]'+context_r])
            sent0 = ' '.join([subj, verb, obj, context_l, '[MASK]'+context_r])
            sent1 = ' '.join([subj, verb, obj, context_l, '[+female]'+context_r])

            if "'" in det_two:
                subj = ''.join([det_two, noun_two])
            else:
                subj = ' '.join([det_two, noun_two])

            if det_one == 'il':
                prep = 'del'
            if det_one == "l'":
                prep = "dell'"
            if det_one == 'lo':
                prep = "dello"

            if "'" in prep:
                obj = ''.join([prep, noun_one])
            else:
                obj = ' '.join([prep, noun_one])

            #sent2 = ' '.join([subj, verb, obj, context_l, 'egli', context_r])
            #sent2 = ' '.join([subj, verb, obj, context_l, '[MASK]', context_r])
            #sent3 = ' '.join([subj, verb, obj, context_l, 'ella', context_r])

            #sent2 = ' '.join([subj, verb, obj, context_l, '[+male]'+context_r])
            sent2 = ' '.join([subj, verb, obj, context_l, '[MASK]'+context_r])
            sent3 = ' '.join([subj, verb, obj, context_l, '[+female]'+context_r])
                
        elif verb.split(' ')[-1] == 'a':
            verb = ' '.join(verb.split(' ')[:-1])

            if "'" in det_one:
                subj = ''.join([det_one, noun_one])
            else:
                subj = ' '.join([det_one, noun_one])

            if det_two == 'la':
                prep = 'alla'
            if det_two == "l'":
                prep = "all''"

            if "'" in prep:
                obj = ''.join([prep, noun_two])
            else:
                obj = ' '.join([prep, noun_two])

            #sent0 = ' '.join([subj, verb, obj, context_l, 'egli', context_r])
            #sent0 = ' '.join([subj, verb, obj, context_l, '[MASK]', context_r])
            #sent1 = ' '.join([subj, verb, obj, context_l, 'ella', context_r])

            #sent0 = ' '.join([subj, verb, obj, context_l, '[+male]'+context_r])
            sent0 = ' '.join([subj, verb, obj, context_l, '[MASK]'+context_r])
            sent1 = ' '.join([subj, verb, obj, context_l, '[+female]'+context_r])

            if "'" in det_two:
                subj = ''.join([det_two, noun_two])
            else:
                subj = ' '.join([det_two, noun_two])

            if det_one == 'il':
                prep = 'al'
            if det_one == "l'":
                prep = "all'"
            if det_one == 'lo':
                prep = "allo"

            if "'" in prep:
                obj = ''.join([prep, noun_one])
            else:
                obj = ' '.join([prep, noun_one])

            #sent2 = ' '.join([subj, verb, obj, context_l, 'egli', context_r])
            #sent2 = ' '.join([subj, verb, obj, context_l, '[MASK]', context_r])
            #sent3 = ' '.join([subj, verb, obj, context_l, 'ella', context_r])

            #sent2 = ' '.join([subj, verb, obj, context_l, '[+male]'+context_r])
            sent2 = ' '.join([subj, verb, obj, context_l, '[MASK]'+context_r])
            sent3 = ' '.join([subj, verb, obj, context_l, '[+female]'+context_r])

        else:
            if "'" in det_one:
                subj = ''.join([det_one, noun_one])
            else:
                subj = ' '.join([det_one, noun_one])

            if "'" in det_two:
                obj = ''.join([det_two, noun_two])
            else:
                obj = ' '.join([det_two, noun_two])

            #sent0 = ' '.join([subj, verb, obj, context_l, 'egli', context_r])
            #sent0 = ' '.join([subj, verb, obj, context_l, '[MASK]', context_r])
            #sent1 = ' '.join([subj, verb, obj, context_l, 'ella', context_r])

            #sent0 = ' '.join([subj, verb, obj, context_l, '[+male]'+context_r])
            sent0 = ' '.join([subj, verb, obj, context_l, '[MASK]'+context_r])
            sent1 = ' '.join([subj, verb, obj, context_l, '[+female]'+context_r])

            if "'" in det_two:
                subj = ''.join([det_two, noun_two])
            else:
                subj = ' '.join([det_two, noun_two])

            if "'" in det_one:
                obj = ''.join([det_one, noun_one])
            else:
                obj = ' '.join([det_one, noun_one])


            #sent2 = ' '.join([subj, verb, obj, context_l, 'egli', context_r])
            #sent2 = ' '.join([subj, verb, obj, context_l, '[MASK]', context_r])
            #sent3 = ' '.join([subj, verb, obj, context_l, 'ella', context_r])

            #sent2 = ' '.join([subj, verb, obj, context_l, '[+male]'+context_r])
            sent2 = ' '.join([subj, verb, obj, context_l, '[MASK]'+context_r])
            sent3 = ' '.join([subj, verb, obj, context_l, '[+female]'+context_r])
                
        #print(','.join(['ic_mismatch', verb, str(count), sent0, bias, '1', 'm']) )
        #print(','.join(['ic_mismatch', verb, str(count), sent1, bias, '0', 'f']) )
        #print(','.join(['ic_mismatch', verb, str(count), sent2, bias, '0', 'm']) )
        #print(','.join(['ic_mismatch', verb, str(count), sent3, bias, '1', 'f']) )

    count += 1

#For BERT Dutch
verbs = open('nl_IC_verbs', 'r')
bias = open('nl_IC_bias', 'r')

data = {}
for line in verbs:
    line = line.strip()
    b = bias.readline().strip()
    #skip over out of vocab IC
    data[line] = b

verbs.close()
bias.close()

#gender mismatch
pairs = []
with open('nl_gender_pairs', 'r') as f:
#with open('gender_pairs', 'r') as f:
    for line in f:
        line = line.strip().split()
        pairs.append(line)

frames = []
with open('nl-stim.csv', 'r') as f:
    f.readline()
    for line in f:
        line = line.strip().split(',')
        frames.append(line)

count = 0
for pair in pairs:
    for index, verb in enumerate(data):
        bias = data[verb]

        det_one = pair[0]
        noun_one = pair[1]
        det_two = pair[2]
        noun_two = pair[3]

        frame = frames[index]
        assert 'SUBJ' in frame[0]
        assert 'OBJ' in frame[0]
        sent0 = ' '.join([det_one.capitalize(), frame[0].replace('SUBJ', noun_one).replace('OBJ', det_two + ' '+noun_two), frame[1]])
        #sent0 = ' '.join([det_one.capitalize(), frame[0].replace('SUBJ', noun_one).replace('OBJ', det_two + ' '+noun_two), frame[1].replace(' hij ', ' [MASK] ')])

        sent1 = ' '.join([det_two.capitalize(), frame[0].replace('SUBJ', noun_two).replace('OBJ', det_one + ' '+noun_one), frame[2]])
        #sent1 = ' '.join([det_two.capitalize(), frame[0].replace('SUBJ', noun_two).replace('OBJ', det_one + ' '+noun_one), frame[2].replace(' hij ', ' [MASK] ')])

        '''
        if int(bias) > 50:
            #sent0 = ' '.join([det_one.capitalize(), frame[0].replace('SUBJ', noun_one).replace('OBJ', det_two + ' '+noun_two), frame[1]])
            sent0 = ' '.join([det_one.capitalize(), frame[0].replace('SUBJ', noun_one).replace('OBJ', det_two + ' '+noun_two), frame[1].replace(' hij ', ' [MASK] ')])

            #sent1 = ' '.join([det_two.capitalize(), frame[0].replace('SUBJ', noun_two).replace('OBJ', det_one + ' '+noun_one), frame[2]])
            sent1 = ' '.join([det_two.capitalize(), frame[0].replace('SUBJ', noun_two).replace('OBJ', det_one + ' '+noun_one), frame[2].replace(' hij ', ' [MASK] ')])
        else:
            #sent0 = ' '.join([det_two.capitalize(), frame[0].replace('SUBJ', noun_two).replace('OBJ', det_one + ' '+noun_one), frame[1]])
            sent0 = ' '.join([det_two.capitalize(), frame[0].replace('SUBJ', noun_two).replace('OBJ', det_one + ' '+noun_one), frame[1].replace(' hij ', ' [MASK] ')])

            #sent1 = ' '.join([det_one.capitalize(), frame[0].replace('SUBJ', noun_one).replace('OBJ', det_two + ' '+noun_two), frame[2]])
            sent1 = ' '.join([det_one.capitalize(), frame[0].replace('SUBJ', noun_one).replace('OBJ', det_two + ' '+noun_two), frame[2].replace(' hij ', ' [MASK] ')])
        '''

                
        #print(','.join(['ic_mismatch', verb, str(count), sent0, bias, '1', 'm']) )
        #print(','.join(['ic_mismatch', verb, str(count), sent1, bias, '0', 'm']) )
        
    count += 1

#For BERT Chinese
verbs = open('zh_IC_verbs', 'r')
bias = open('zh_IC_bias', 'r')

data = {}
for line in verbs:
    line = line.strip()
    b = bias.readline().strip()
    #skip over out of vocab IC
    data[line] = b

verbs.close()
bias.close()

#gender mismatch
pairs = []
with open('zh_gender_pairs', 'r') as f:
#with open('gender_pairs', 'r') as f:
    for line in f:
        if '#' in line:
            continue
        line = line.strip().split()
        pairs.append(line)

count = 0
for pair in pairs:
    for index, verb in enumerate(data):
        bias = data[verb]

        sent0 = ''.join([pair[0], verb, pair[1],'因为他在那里。'])
        #sent0 = ''.join([pair[0], verb, pair[1],'因为[MASK]在那里。'])
        sent1 = ''.join([pair[0], verb, pair[1],'因为她在那里。'])
        sent2 = ''.join([pair[1], verb, pair[0],'因为他在那里。'])
        #sent2 = ''.join([pair[1], verb, pair[0],'因为[MASK]在那里。'])
        sent3 = ''.join([pair[1], verb, pair[0],'因为她在那里。'])

        print(','.join(['ic_mismatch', verb, str(count), sent0, bias, '1', 'm']) )
        print(','.join(['ic_mismatch', verb, str(count), sent1, bias, '0', 'f']) )
        print(','.join(['ic_mismatch', verb, str(count), sent2, bias, '0', 'm']) )
        print(','.join(['ic_mismatch', verb, str(count), sent3, bias, '1', 'f']) )

    count += 1
