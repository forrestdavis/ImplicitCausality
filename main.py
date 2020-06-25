import torch
import numpy as np
from transformers import TransfoXLTokenizer, TransfoXLLMHeadModel, TransfoXLModel

#device = torch.device("cuda:0")

def check_unks(fname, vocabf):

    vocab = set([])
    with open(vocabf, 'r') as d:
        for line in d:
            line = line.strip()
            vocab.add(line)

    with open(fname, 'r') as f:
        header = f.readline().split(',')

        idx = 0
        for x in range(len(header)):
            if header[x] == 'sent':
                idx = x

        OOV = set([])
        for line in f:
            sent = line.strip().split(',')[idx].lower().split(' ')
            for word in sent:
                if word not in vocab:
                    OOV.add(word)
        return OOV

def load_data(fname):

    sents = []

    with open(fname, 'r') as f:

        header = f.readline().split(',')

        idx = 0
        for x in range(len(header)):
            if header[x] == 'sent':
                idx = x

        for line in f:
            sent = line.strip().split(',')[idx].lower()
            sents.append(sent)

    return sents

def apply(func, apply_dimension):
    ''' Applies a function along a given dimension '''
    output_list = [func(m) for m in torch.unbind(apply_dimension, dim=0)]
    return torch.stack(output_list, dim=0)

def test_get_batch(source):
    ''' Creates an input/target pair for evaluation '''
    seq_len = len(source) - 1
    #Get all words except last one (EOS)
    data = source[:seq_len]
    #Get all targets
    target = source[1:1+seq_len].view(-1)
    
    return data, target

def tfxl_tokenize(tf_tokenizer, sents):
    ''' Takes a list of sents, tokenizes them,
        Returns input_ids'''

    input_ids = []
    for sent in sents:

        encoded = tf_tokenizer.encode(sent, add_special_tokens=True)

        sent_id = torch.tensor(encoded)
        decoded = tf_tokenizer.decode(encoded)

        #assert decoded == sent

        input_ids.append(sent_id)

    return input_ids

def tfxl_IT(sents, tokenizer, model):
    
    values = []

    for sent in sents:

        sent_values = []

        #tokenize sentence
        input_ids = torch.tensor(tokenizer.encode(sent, add_special_tokens=True)).unsqueeze(0)

        #verify no unks
        decoded = tokenizer.decode(input_ids)
        assert decoded == sent

        #Get model outputs
        outputs = tf_model(input_ids)
        #get predictions
        predictions, mems = outputs[:2]
        #Get surprisal
        surps = torch.log2(torch.exp(-1*torch.nn.functional.log_softmax(predictions, -1)))

        #Get surprisal by target word (i.e. dog given the)
        count = 0 
        for y in range(len(input_ids[0])-1):
            target_id = input_ids[0][y+1]
            input_id = input_ids[0][y]

            target_word = tokenizer.decode([target_id]).replace(' ','')
            surp = float(surps[0, y, int(target_id)].data)
            sent_values.append((target_word, surp))
        values.append(sent_values)

    return values

'''
fname = "stimuli/IC_match.csv"
vocabf = 'wikitext103_vocab'
check_unks(fname, vocabf)
'''

fname = "stimuli/IC_mismatch.csv"

sents = load_data(fname)

tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
tf_model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')

#turn off learning
tf_model.zero_grad()

measures = tfxl_IT(sents, tokenizer, tf_model)

for measure in measures:
    target_word, surp = measure[-1]
    assert target_word == 'was' or target_word == 'were'
    print(surp)

'''
sent = 'scolded the chefs of the aristocrat who were'
sents = ['The woman scolded the chef of the aristocrats who was', 'The woman scolded the chefs of the aristocrat who was', 'The woman scolded the chef of the aristocrats who eat', 
        'The woman scolded the chefs of the aristocrat who eat', 'The woman who the man despises eats', 'The woman who the man despises eat', 
        'The monkey potato that these applesauce flower aristocrat banana were',
        'The monkey potato that these applesauce flower president fruit were',
        'Monkey potato that these appllesauce flowers were'] 
sents = ['scolded the chefs of the aristocrat who were', 
        'scolded the chefs of the president who were', 
        'scolded the chefs of the banana who were', 
        'scolded the chefs of the aristocrats who were', 
        'scolded the chefs of the presidents who were', 
        'scolded the chefs of the bananas who were', 
        'scolded the chef of the aristocrat who were', 
        'scolded the chef of the president who were', 
        'scolded the chef of the banana who were', 
        'scolded the chef of the aristocrats who were', 
        'scolded the chef of the presidents who were', 
        'scolded the chef of the bananas who were', 
        'scolded the chefs of the devour who were', 
        'scolded the chefs of the who were', 
        'scolded the were',
        'the were'
        'blank who were']
'''
