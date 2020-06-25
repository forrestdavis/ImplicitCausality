import glob
import torch
import warnings
import lm_data
warnings.filterwarnings('ignore') #spicy
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
        encoded = tokenizer.encode(sent, add_special_tokens=True)
        input_ids = torch.tensor(encoded).unsqueeze(0)

        #verify no unks
        decoded = tokenizer.decode(encoded)
        assert decoded == sent

        #Get model outputs
        outputs = model(input_ids)
        #get predictions
        predictions, mems, hidden_states = outputs
        #Get surprisal
        surps = torch.log2(torch.exp(-1*torch.nn.functional.log_softmax(predictions, -1)))

        #Get surprisal by target word (i.e. dog given the)
        count = 0 
        for y in range(len(input_ids[0])-1):
            target_id = input_ids[0][y+1]
            input_id = input_ids[0][y]

            target_word = tokenizer.decode([target_id]).replace(' ','')
            surp = float(surps[0, y, int(target_id)].data)
            #sent_values.append((target_word, surp))
        sent_values.append((target_word, surp))
        values.append(sent_values)

    return values

def run_lms(sents, vocab_file, model_files):

    data_path = './'

    for model_file in model_files:
        print('testing LSTM LM:', model_file)

        with open(model_file, 'rb') as f:
            model = torch.load(f, map_location='cpu')

            if isinstance(model, torch.nn.DataParallel):
                model = model.module

        torch.save(model, 'please.pt')

        #set to single sentence reading
        multisent_flag = False
        for sent in sents:

            #Create corpus wrapper (this is for one hoting data)
            corpus = lm_data.TestSent(data_path, vocab_file, [sent], 
                                        multisent_flag)

            #Get one hots
            sent_ids = corpus.get_data()[0].to(torch.device('cpu'))
            hidden = model.init_hidden(1)

            data, targets = test_get_batch(sent_ids)

            data = data.unsqueeze(1)

            #output, hidden = model(data, hidden)
            #print(output)



            '''
            output, hidden = model(torch.tensor([sent_ids]), hidden)
            print(output)
            '''

            '''
            for i in range(len(sent_ids)-1):
                input_word_id = sent_ids[i]
                target_word_id = sent_ids[i+1]
                target_word = corpus.dictionary.idx2word[int(target_word_id)]

                print(target_word)
            '''
                
            break
        break


'''
fname = "stimuli/IC_match.csv"
vocabf = 'wikitext103_vocab'
check_unks(fname, vocabf)
'''

#fname = "stimuli/IC_mismatch.csv"
fname = "stimuli/Reading_Time.csv"

sents = load_data(fname)

##############
#LSTM LMs
##############
vocabf = 'wikitext103_vocab'
lm_models = glob.glob('models/*.pt')
lm_models.sort()
print(len(lm_models),lm_models)
run_lms(sents, vocabf, lm_models)

##############
#Transformers
##############
'''
tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
tf_model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103', 

#turn off learning
tf_model.zero_grad()

measures = tfxl_IT(sents, tokenizer, tf_model)

for measure in measures:
    target_word, surp = measure[-1]
    assert target_word == 'was' or target_word == 'were'
    print(surp)
'''
