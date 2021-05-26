import re

COUNTS = {'1sg': 0, '1pl': 0, '2sg': 0, '2pl': 0, '3sg':0, '3pl':0, 
        '1': 0, '2': 0, '3':0}

ES_PRONOUNS = {'1sg': ['yo'], 
        '1pl': ['nosotros', 'nosotras'], 
        '2sg': ['tú'],
        '2pl': ['vosotros','vosotras'],
        '3sg': ['él', 'ella'],
        '3pl': ['ellos', 'ellas']}

IT_PRONOUNS = {'1sg': ['io'], 
        '1pl': ['noi'],
        '2sg': ['tú'],
        '2pl': ['voi'],
        '3sg': ['lui', 'lei'],
        '3pl': ['loro']}

def remove_pros(filename, IC_verbs=None, isBaseline=False, clip_len=None):

    PROS = []
    remainder = []
    with open(filename, 'r') as f:

        sent = []
        text = ''

        for line in f:

            line = line.strip().split()
            #end of sent
            if len(line) < 2:

                if clip_len:
                    if len(PROS) > clip_len-1:
                        return PROS, remainder

                hasIC = 0
                if IC_verbs:
                    hasIC = check_for_IC_verb(sent, IC_verbs)

                if not hasIC:
                    old_text = text
                    hadPronoun, text = check_for_pro_nsubj(sent, text)

                    if isBaseline:
                        text = old_text.lower()
                    else:
                        text = text.lower()

                    if hadPronoun:
                        PROS.append(text)
                    else:
                        remainder.append(text)

                text = ''
                sent = []
                continue

            if line[0] == '#':
                if 'text' in line and 'text_en' not in line:
                    text = ' '.join(line[3:])
                continue

            sent.append(line)

    return PROS, remainder

def create_text(sent):

    text = ''
    for word in sent:
        text += word[1]
        if word[-1] == 'SpaceAfter=No':
            continue
        else:
            text += ' '

    return text

def check_for_pro_nsubj(sent, text):

    hadPronoun = 0
    replace = []

    mod_sent = []
    for idx, word in enumerate(sent):
        #Skip over those multiword entries like dont
        if '-' in word[0]:
            continue
        word_rel = word[-3]
        if 'nsubj' in word_rel or 'expl' in word_rel:
            if word[3] == 'PRON' and word[4] == 'PRP':

                person = ''
                if  'Person=1' in word[5]:
                    person = '1'
                elif  'Person=2' in word[5]:
                    person = '2'
                elif  'Person=3' in word[5]:
                    person = '3'
                number = ''
                if 'Number=Plur' in word[5]:
                    number='pl'
                elif "Number=Sing" in word[5]:
                    number = 'sg'

                #must be inflected for person
                if person == '':
                    continue

                hadPronoun = 1
                express = person+number
                COUNTS[express] += 1

                #Move over space info
                if len(mod_sent) > 0:
                    mod_sent[-1][-1] = word[-1]

                replace.append(word[1])
            else:
                mod_sent.append(word)
        else:
            mod_sent.append(word)

    if hadPronoun:
        #Make sent
        text = create_text(mod_sent)

    return hadPronoun, text

def add_pros(filename, PRONOUNS, IC_verbs=None, isBaseline=False, clip_len=None):

    PROS = []
    remainder = []
    with open(filename, 'r') as f:

        sent = []
        text = ''

        for line in f:

            line = line.strip().split()
            #end of sent
            if len(line) < 2:

                if clip_len:
                    if len(PROS) > clip_len-1:
                        return PROS, remainder

                hasIC = 0
                if IC_verbs:
                    hasIC = check_for_IC_verb(sent, IC_verbs)

                if not hasIC:

                    old_text = text
                    isMissing, text = check_for_pro_verb(sent, text, PRONOUNS)

                    if isBaseline:
                        text = old_text.lower()
                    else:
                        text = text.lower()

                    if isMissing:
                        PROS.append(text)
                    else:
                        remainder.append(text)

                text = ''
                sent = []
                continue

            if line[0] == '#':
                if 'text' in line:
                    text = ' '.join(line[3:])
                continue

            sent.append(line)

    return PROS, remainder

def check_for_IC_verb(sent, IC_verbs):

    hasIC = 0
    for word in sent:
        word = word[1].lower()
        if word in IC_verbs:
            hasIC = 1

    return hasIC

def check_for_pro_verb(sent, text, PRONOUNS):

    VERBS = {}
    isMissing = 0
    pronoun_expressions = set([])

    mod_sent = []
    #Find verbs locs
    for word in sent:
        if '-' in word[0]:
            continue
        mod_sent.append(word)
        if "'" in word[1]: 
            word[-1] = "SpaceAfter=No"
        if word[3] == 'VERB':
            #Check if finite
            if 'VerbForm=Fin' in word[5] and 'Person' in word[5] and 'Number' in word[5]:
                VERBS[word[0]] = {}
                VERBS[word[0]]['info'] = word
                VERBS[word[0]]['hasSubj'] = 0


                person = ''
                if  'Person=1' in word[5]:
                    person = '1'
                elif  'Person=2' in word[5]:
                    person = '2'
                elif  'Person=3' in word[5]:
                    person = '3'
                assert person != ''
                number = ''
                if 'Number=Plur' in word[5]:
                    number='pl'
                elif "Number=Sing" in word[5]:
                    number = 'sg'
                assert number != ''

                express = person+number
                VERBS[word[0]]['type']=express
                pronoun_expressions.add(express)

    #Find nsubj
    for word in sent:
        word_rel = word[-3]
        word_head = word[-4]
        if word_head in VERBS:
            if 'nsubj' in word_rel or 'expl' in word_rel:
                VERBS[word_head]['hasSubj'] = 1

    count_holder = 0
    added_length = 0
    y = 0
    for verb in VERBS:
        if not VERBS[verb]['hasSubj']:
            #Mark that a replacement has been made
            isMissing = 1

            word = VERBS[verb]['info'][1]
            p_type = VERBS[verb]['type']

            #if all verbs are inflected for the same 
            #person and number assume they are all 
            #co-indexed so assign them a uniform gender
            if len(pronoun_expressions) == 1:
                count_holder += 1
            else:
                COUNTS[p_type] += 1

            #Has gender
            if len(PRONOUNS[p_type]) == 2:
                pronoun = PRONOUNS[p_type][COUNTS[p_type]%2]
            else:
                pronoun = PRONOUNS[p_type][0]

            prior_len = len(text)
            #text = text.replace(word, pronoun + ' ' + word)
            entry = ['']*10
            entry[1] = pronoun
            added_length += len(pronoun + ' ')
            mod_sent.insert(y+int(verb)-1, entry)
            y += 1

    if isMissing:
        #Make pro added sent
        text = create_text(mod_sent)
    if len(pronoun_expressions) == 1:
        COUNTS[VERBS[verb]['type']] += count_holder

    return isMissing, text


def get_pro_drop_sents(filename):

    SENT_COUNT = 0
    PRO_DROP = 0

    PROs = []
    other = []
    with open(filename, 'r') as f:

        sent = []
        text = ''
        hasNSUBJ = 0
        for line in f:

            line = line.strip().split()
            #end of sent
            if len(line) < 2:
                SENT_COUNT += 1
                if not hasNSUBJ:
                    PRO_DROP += 1
                    PROs.append((text, sent))
                else:
                    other.append((text.lower(), sent))

                text = ''
                sent = []
                hasNSUBJ = 0
                continue

            if line[0] == '#':
                if 'text' in line:
                    text = ' '.join(line[3:])
                continue

            sent.append(line)

            if line[-3] == 'nsubj':
                hasNSUBJ = 1


    return PROs, other

def save_sents(sents, filename='es_train_finetuning'):
    
    with open(filename, 'w') as o:

        for sent in sents:
            sent = sent.strip()
            o.write(sent+'\n')

def load_IC_verbs(filename='../tools/es_IC_verbs'):

    IC_verbs = set([])

    with open(filename, 'r') as f:

        for line in f:
            line = line.strip()
            IC_verbs.add(line)

    return IC_verbs

###English###
##Baseline
print('English:')
COUNTS = {'1sg': 0, '1pl': 0, '2sg': 0, '2pl': 0, '3sg':0, '3pl':0, 
        '1': 0, '2': 0, '3':0}
filename = 'data/en_ewt_data.conllu'
IC_verbs = load_IC_verbs('../tools/IC_verbs')
sents, remainder = remove_pros(filename, IC_verbs, True, clip_len=4000)
save_sents(sents, 'data/en_train_finetuning_base')
save_sents(remainder[:500], 'data/en_valid_finetuning_base')

##Without Pro
COUNTS = {'1sg': 0, '1pl': 0, '2sg': 0, '2pl': 0, '3sg':0, '3pl':0, 
        '1': 0, '2': 0, '3':0}
filename = 'data/en_ewt_data.conllu'
IC_verbs = load_IC_verbs('../tools/IC_verbs')
sents, remainder = remove_pros(filename, IC_verbs, False, clip_len=4000)
print('Number of modified sentences:', len(sents))
print('Total Pronouns Removed:', sum(COUNTS.values()))
print(COUNTS)
save_sents(sents, 'data/en_train_finetuning')
save_sents(remainder[:500], 'data/en_valid_finetuning')

###Chinese###
##Baseline
print('Chinese:')
COUNTS = {'1sg': 0, '1pl': 0, '2sg': 0, '2pl': 0, '3sg':0, '3pl':0, 
        '1': 0, '2': 0, '3':0}
filename = 'data/zh_combined_data.conllu'
IC_verbs = load_IC_verbs('../tools/zh_IC_verbs')
sents, remainder = remove_pros(filename, IC_verbs, True)
save_sents(sents, 'data/zh_train_finetuning_base')
save_sents(remainder[:500], 'data/zh_valid_finetuning_base')

##Without Pro
COUNTS = {'1sg': 0, '1pl': 0, '2sg': 0, '2pl': 0, '3sg':0, '3pl':0, 
        '1': 0, '2': 0, '3':0}
filename = 'data/zh_combined_data.conllu'
IC_verbs = load_IC_verbs('../tools/zh_IC_verbs')
sents, remainder = remove_pros(filename, IC_verbs, False)
print('Number of modified sentences:', len(sents))
print('Total Pronouns Removed:', sum(COUNTS.values()))
print(COUNTS)
save_sents(sents, 'data/zh_train_finetuning')
save_sents(remainder[:500], 'data/zh_valid_finetuning')

###Italian###
##Baseline
print('Italian:')
COUNTS = {'1sg': 0, '1pl': 0, '2sg': 0, '2pl': 0, '3sg':0, '3pl':0, 
        '1': 0, '2': 0, '3':0}
filename = 'data/it_combined_data.conllu'
IC_verbs = load_IC_verbs('../tools/it_IC_verbs')
sents, remainder = add_pros(filename, IT_PRONOUNS, IC_verbs, True)
save_sents(sents, 'data/it_train_finetuning_base')
save_sents(remainder[:500], 'data/it_valid_finetuning_base')

##Without Pro
COUNTS = {'1sg': 0, '1pl': 0, '2sg': 0, '2pl': 0, '3sg':0, '3pl':0, 
        '1': 0, '2': 0, '3':0}
filename = 'data/it_combined_data.conllu'
IC_verbs = load_IC_verbs('../tools/it_IC_verbs')
sents, remainder = add_pros(filename, IT_PRONOUNS, IC_verbs, False)
print('Number of modified sentences:', len(sents))
print('Total Pronouns Added:', sum(COUNTS.values()))
print(COUNTS)
save_sents(sents, 'data/it_train_finetuning')
save_sents(remainder[:500], 'data/it_valid_finetuning')

###Spanish###
##Baseline
print('Spanish:')
COUNTS = {'1sg': 0, '1pl': 0, '2sg': 0, '2pl': 0, '3sg':0, '3pl':0, 
        '1': 0, '2': 0, '3':0}
filename = 'data/es_combined_data.conllu'
IC_verbs = load_IC_verbs('../tools/es_IC_verbs')
sents, remainder = add_pros(filename, ES_PRONOUNS, IC_verbs, True, 4000)
save_sents(sents, 'data/es_train_finetuning_base')
save_sents(remainder[:500], 'data/es_valid_finetuning_base')

##Without Pro
COUNTS = {'1sg': 0, '1pl': 0, '2sg': 0, '2pl': 0, '3sg':0, '3pl':0, 
        '1': 0, '2': 0, '3':0}
filename = 'data/es_combined_data.conllu'
IC_verbs = load_IC_verbs('../tools/es_IC_verbs')
sents, remainder = add_pros(filename, ES_PRONOUNS, IC_verbs, False, 4000)
print('Number of modified sentences:', len(sents))
print('Total Pronouns Added:', sum(COUNTS.values()))
print(COUNTS)
save_sents(sents, 'data/es_train_finetuning')
save_sents(remainder[:500], 'data/es_valid_finetuning')
