## Story completion experiment
names = {1: 'The man', 
        2: 'The man', 
        3: 'The man', 
        4: 'The man', 
        5: 'The man', 
        6: 'The woman', 
        7: 'The woman', 
        8: 'The woman', 
        9: 'The woman', 
        10: 'The woman', 
        11: 'The man', 
        12: 'The man', 
        13: 'The man', 
        14: 'The man', 
        15: 'The woman', 
        16: 'The man', 
        17: 'The man', 
        18: 'The man', 
        19: 'The man', 
        20: 'The woman', 
        21: 'The man'}

verbs = {1: ['admires', 'works with'],
        2: ['adores', 'smiles at'],
        3: ['blamed', 'noticed'],
        4: ['complimented', 'met'],
        5: ['congratulated', 'visited'],
        6: ['criticized', 'talked to'],
        7: ['despises', 'babysits'],
        8: ['detests', 'looks like'],
        9: ['dislikes', 'watches'],
        10: ['insulted', 'chatted with'],
        11: ['likes', 'resembles'],
        12: ['pities', 'hires'],
        13: ['praised', 'videotaped'],
        14: ['punished', 'saw'],
        15: ['resents', 'knows'],
        16: ['scolded', 'recognized'],
        17: ['rewarded', 'inspected'],
        18: ['ridiculed', 'counted'],
        19: ['thanked', 'interviewed'],
        20: ['values', 'lives next to'],
        21: ['worships', 'listens to']}

NP1 = {1: ['agent', 'agents'], 
        2: ['secretary', 'secretaries'],
        3: ['friend', 'friends'],
        4: ['guest', 'guests'],
        5: ['teacher', 'teachers'],
        6: ['leader', 'leaders'],
        7: ['child', 'children'],
        8: ['father', 'fathers'],
        9: ['little girl', 'little girls'],
        10: ['gardener', 'gardeners'],
        11: ['captain', 'captains'],
        12: ['bodyguard', 'bodyguards'],
        13: ['assistant', 'assistant'],
        14: ['accountant', 'accountants'],
        15: ['doctor', 'doctors'],
        16: ['landlady', 'landladies'],
        17: ['servant', 'servants'],
        18: ['fan', 'fans'],
        19: ['representative', 'representatives'],
        20: ['surgeon', 'surgeons'],
        21: ['coach', 'coaches']}

NP2 = {1: ['rockstar', 'rockstars'], 
        2: ['lawyer', 'lawyers'],
        3: ['athlete', 'athletes'],
        4: ['bride', 'brides'],
        5: ['second grader', 'second graders'],
        6: ['activists', 'activists'],
        7: ['jazz musician', 'jazz musicians'],
        8: ['student', 'students'],
        9: ['neighbor', 'neighbors'],
        10: ['millionaire', 'millionaires'],
        11: ['old sailor', 'old sailors'],
        12: ['celebrity', 'celebrities'],
        13: ['ceo', 'ceos'],
        14: ['businessman', 'businessman'],
        15: ['supermodel', 'supermodels'],
        16: ['actor', 'actors'],
        17: ['dictator', 'dictator'],
        18: ['singer', 'singers'],
        19: ['employee', 'employees'],
        20: ['soldier', 'soldiers'],
        21: ['cheerleader', 'cheerleaders']}

for key in names:
    name = names[key]
    IC = verbs[key][0]
    non_IC = verbs[key][1]
    NP1_sg = NP1[key][0]
    NP1_pl = NP1[key][1]
    NP2_sg = NP2[key][0]
    NP2_pl = NP2[key][1]

    '''
    print(' '.join([name, IC, 'the', NP1_sg, 'of the', NP2_pl, 'who was']))
    print(' '.join([name, IC, 'the', NP1_pl, 'of the', NP2_sg, 'who was']))
    print(' '.join([name, IC, 'the', NP1_sg, 'of the', NP2_pl, 'who were']))
    print(' '.join([name, IC, 'the', NP1_pl, 'of the', NP2_sg, 'who were']))

    print(' '.join([name, non_IC, 'the', NP1_sg, 'of the', NP2_pl, 'who was']))
    print(' '.join([name, non_IC, 'the', NP1_pl, 'of the', NP2_sg, 'who was']))
    print(' '.join([name, non_IC, 'the', NP1_sg, 'of the', NP2_pl, 'who were']))
    print(' '.join([name, non_IC, 'the', NP1_pl, 'of the', NP2_sg, 'who were']))
    '''

    
#Reading time
names = {1: 'The woman', 
        2: 'The man', 
        3: 'The woman', 
        4: 'The man', 
        5: 'The woman', 
        6: 'The man', 
        7: 'The woman', 
        8: 'The man', 
        9: 'The woman', 
        10: 'The man', 
        11: 'The woman', 
        12: 'The man', 
        13: 'The woman', 
        14: 'The man', 
        15: 'The woman', 
        16: 'The man', 
        17: 'The woman', 
        18: 'The man', 
        19: 'The woman', 
        20: 'The man'}

verbs = {1: ['scolded', 'studied with'],
        2: ['stared at', 'lived next to'],
        3: ['assisted', 'joked with'],
        4: ['trusted', 'stood near'],
        5: ['corrected', 'gossiped with'],
        6: ['comforted', 'greeted'],
        7: ['envies', 'knows'],
        8: ['valued', 'recognized'],
        9: ['fears', 'jogs with'],
        10: ['noticed', 'resembled'],
        11: ['praised', 'met'],
        12: ['hates', 'carpools with'],
        13: ['blamed', 'waited with'],
        14: ['helped', 'ran into'],
        15: ['reproached', 'worked with'],
        16: ['pacified', 'visited'],
        17: ['detests', 'babysits'],
        18: ['thanked', 'talked to'],
        19: ['congratulated', 'chatted with'],
        20: ['mocked', 'counted']}

NP1 = {1: ['chef', 'chefs'], 
        2: ['teacher', 'teachers'],
        3: ['maid', 'maids'],
        4: ['captain', 'captains'],
        5: ['secretary', 'secretaries'],
        6: ['leader', 'leaders'],
        7: ['manager', 'managers'],
        8: ['daughter', 'daughters'],
        9: ['uncle', 'uncles'],
        10: ['representative', 'representatives'],
        11: ['gardener', 'gardeners'],
        12: ['cousin', 'cousins'],
        13: ['niece', 'nieces'],
        14: ['brother', 'brothers'],
        15: ['doctors', 'doctors'],
        16: ['associate', 'associates'],
        17: ['child', 'children'],
        18: ['servant', 'servants'],
        19: ['bodyguard', 'bodyguards'],
        20: ['fan', 'fans']}

NP2 = {1: ['aristocrat', 'aristocrats'], 
        2: ['second grader', 'second graders'],
        3: ['executive', 'executives'],
        4: ['sailor', 'sailors'],
        5: ['lawyer', 'lawyers'],
        6: ['activist', 'activists'],
        7: ['cashier', 'cashiers'],
        8: ['shopkeeper', 'shopkeepers'],
        9: ['toddler', 'toddlers'],
        10: ['employee', 'employees'],
        11: ['millionaire', 'millionaires'],
        12: ['accountant', 'accountants'],
        13: ['florist', 'florists'],
        14: ['athlete', 'athletes'],
        15: ['supermodel', 'supermodels'],
        16: ['businessman', 'businessmen'],
        17: ['musician', 'musicians'],
        18: ['dictator', 'dictators'],
        19: ['celebrity', 'celebrities'],
        20: ['singer', 'singers']}

for key in names:
    name = names[key]
    IC = verbs[key][0]
    non_IC = verbs[key][1]
    NP1_sg = NP1[key][0]
    NP1_pl = NP1[key][1]
    NP2_sg = NP2[key][0]
    NP2_pl = NP2[key][1]

    '''
    print(' '.join([name, IC, 'the', NP1_sg, 'of the', NP2_pl, 'who was']))
    print(' '.join([name, IC, 'the', NP1_pl, 'of the', NP2_sg, 'who was']))
    print(' '.join([name, IC, 'the', NP1_sg, 'of the', NP2_pl, 'who were']))
    print(' '.join([name, IC, 'the', NP1_pl, 'of the', NP2_sg, 'who were']))

    print(' '.join([name, non_IC, 'the', NP1_sg, 'of the', NP2_pl, 'who was']))
    print(' '.join([name, non_IC, 'the', NP1_pl, 'of the', NP2_sg, 'who was']))
    print(' '.join([name, non_IC, 'the', NP1_sg, 'of the', NP2_pl, 'who were']))
    print(' '.join([name, non_IC, 'the', NP1_pl, 'of the', NP2_sg, 'who were']))
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
        print(','.join(['ic_match', verb, str(count), sent0, bias, '0', gender]) )
        print(','.join(['ic_match', verb, str(count), sent1, bias, '0', gender]) )
    count += 1
