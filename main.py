from scipy import stats
import spacy
from tools.utils import *
import glob
import torch
import warnings
import lm_data
warnings.filterwarnings('ignore') #spicy
import numpy as np
import model as m
from transformers import TransfoXLTokenizer, TransfoXLLMHeadModel, TransfoXLModel, GPT2LMHeadModel, GPT2Tokenizer

device = torch.device("cuda:0")

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

def repackage_hidden(in_state):
    if isinstance(in_state, torch.Tensor):
        return in_state.detach()
    else:
        return tuple(repackage_hidden(value) for value in in_state)

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

def tfxl_completions(sents, tokenizer, model, K=100):
    
    values = []
    completions = {}
    completions['tfxl'] = []

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
        #Get probs
        probs = torch.exp(torch.nn.functional.log_softmax(predictions, -1))

        probs = None
        for y in range(len(input_ids[0])):
            input_id = input_ids[0][y]
            input_word = tokenizer.decode([input_id]).replace(' ', '')
            
            if input_word == 'who':
                probs = torch.exp(torch.nn.functional.log_softmax(predictions[0, y, :], -1))

        assert probs is not None

        #Get guesses
        guess_probs, guess_ids = torch.topk(probs, K, 0)
        sent = ' '.join(sent.split(' ')[:-1])

        completes = []
        #Append guess words to input sent
        for x, guess_id in enumerate(guess_ids):
            guess_word = tokenizer.decode(torch.tensor([[guess_id]]))
            completes.append((sent + ' ' + guess_word, float(guess_probs[x].to(device).data)))
        completions['tfxl'].append(completes)

    return completions

def get_tf_hidden(sents, tokenizer, model, num_layers=18):

    #model X layers X sents
    hidden_reps = {}
    hidden_reps['tf'] = {}
    for i in range(num_layers):
        hidden_reps['tf'][i] = []
    
    for sent in sents: 

        encoded = tokenizer.encode(sent, add_special_tokens=True)
        input_ids = torch.tensor(encoded).unsqueeze(0)

        #verify no unks
        decoded = tokenizer.decode(encoded)
        assert decoded == sent

        #Get model outputs
        output = model(input_ids)
        predictions, mems, hidden_states = output

        #ignore embedding
        hidden_states = hidden_states[1:]
        for i in range(len(hidden_states)):
            reps = []
            for x in range(len(encoded)):
                idx = encoded[x]
                input_word = tokenizer.decode(torch.tensor([idx]))
                h = hidden_states[i][0][x].unsqueeze(0).data
                rep = (input_word, h)
                reps.append(rep)
            hidden_reps['tf'][i].append(reps)

    return hidden_reps

def get_gpt_hidden(sents, tokenizer, model, num_layers=18):

    #model X layers X sents
    hidden_reps = {}
    hidden_reps['tf'] = {}
    for i in range(num_layers):
        hidden_reps['tf'][i] = []
    
    for sent in sents: 

        encoded = tokenizer.encode(sent, add_special_tokens=True, 
                add_prefix_space=True)
        input_ids = torch.tensor(encoded).unsqueeze(0)

        #Get model outputs
        output = model(input_ids)
        predictions, mems, hidden_states = output

        #ignore embedding
        hidden_states = hidden_states[1:]

        sent_words = sent.split(' ')
        for i in range(len(hidden_states)):
            reps = []
            idx = 0

            #offset if word is broken down
            for y in range(len(sent_words)):
                sent_word = sent_words[y]
                h_idx = encoded[idx]
                input_word = tokenizer.decode(torch.tensor([h_idx])).strip()
                if input_word != sent_word:
                    while not(sent_word.index(input_word)+len(input_word) == len(sent_word)):
                        idx += 1
                        h_idx = encoded[idx]
                        input_word = tokenizer.decode(torch.tensor([h_idx])).strip()
                h = hidden_states[i][0][idx].unsqueeze(0).data
                rep = (input_word, h)
                reps.append(rep)
                idx += 1

            hidden_reps['tf'][i].append(reps)

    return hidden_reps

def run_lms(sents, vocab_file, model_files):

    data_path = './'
    criterion = torch.nn.CrossEntropyLoss()

    measures = {}
    for model_file in model_files:
        measures[model_file] = []
        print('testing LSTM LM:', model_file)

        #Problem with diff versions of torch
        #model = m.RNNModel('LSTM', 50002, 400, 400, 2, None, 0.2, tie_weights=True)
        model = m.RNNModel('LSTM', 50002, 400, 400, 2, None, 0.2, tie_weights=True).to(device)
        with open(model_file, 'rb') as f:
            loaded_model = torch.load(f, map_location='cpu')
        model.load_state_dict(loaded_model.state_dict())

        #Cancel dropout :)
        model.eval()

        #set to single sentence reading
        multisent_flag = False
        for sent in sents:

            #Create corpus wrapper (this is for one hoting data)
            corpus = lm_data.TestSent(data_path, vocab_file, [sent], 
                                        multisent_flag)

            #Get one hots
            #sent_ids = corpus.get_data()[0].to(torch.device('cpu'))
            sent_ids = corpus.get_data()[0].to(device)
            hidden = model.init_hidden(1)

            data, targets = test_get_batch(sent_ids)

            data = data.unsqueeze(1)

            output, hidden = model(data, hidden)
            #Flatten 
            output = output.view(-1, len(corpus.dictionary))
            surps = torch.log2(torch.exp(-1*torch.nn.functional.log_softmax(output, -1)))

            for idx, target in enumerate(targets):
                #get target word
                target_word = corpus.dictionary.idx2word[int(target)]

                #skip over EOS
                if target_word == "<eos>":
                    continue

                surp = float(surps[idx][int(target)].data)
                metric = (target_word, surp)
            measures[model_file].append(metric)

    return measures

def run_lms_completion(sents, vocab_file, model_files, K=100):

    data_path = './'
    criterion = torch.nn.CrossEntropyLoss()

    completions = {}
    for model_file in model_files:
        completions[model_file] = []
        print('testing LSTM LM:', model_file)

        #Problem with diff versions of torch
        #model = m.RNNModel('LSTM', 50002, 400, 400, 2, None, 0.2, tie_weights=True)
        model = m.RNNModel('LSTM', 50002, 400, 400, 2, None, 0.2, tie_weights=True).to(device)
        with open(model_file, 'rb') as f:
            loaded_model = torch.load(f, map_location='cpu')
        model.load_state_dict(loaded_model.state_dict())


        #Cancel dropout :)
        model.eval()

        #set to single sentence reading
        multisent_flag = False
        for sent in sents:

            #Create corpus wrapper (this is for one hoting data)
            corpus = lm_data.TestSent(data_path, vocab_file, [sent], 
                                        multisent_flag)

            #Get one hots
            #sent_ids = corpus.get_data()[0].to(torch.device('cpu'))
            sent_ids = corpus.get_data()[0].to(device)
            hidden = model.init_hidden(1)

            data, targets = test_get_batch(sent_ids)

            data = data.unsqueeze(1)

            output, hidden = model(data, hidden)
            #Flatten 
            output = output.view(-1, len(corpus.dictionary))

            probs = None
            for idx, input_id in enumerate(data):
                #get input word
                input_word = corpus.dictionary.idx2word[int(input_id)]
                #if relative clause break out
                if input_word == 'who':
                    probs = torch.exp(torch.nn.functional.log_softmax(output[idx], -1))
                    break

            assert probs is not None

            #Get guesses
            guess_probs, guess_ids = torch.topk(probs, K, 0)
            sent = ' '.join(sent.split(' ')[:-1])

            completes = []
            #Append guess words to input sent
            for x, guess_id in enumerate(guess_ids):
                guess_word = corpus.dictionary.idx2word[int(guess_id)]
                completes.append((sent + ' ' + guess_word, float(guess_probs[x].to(device).data)))
            completions[model_file].append(completes)

    return completions


def get_lms_hidden(sents, vocab_file, model_files):

    data_path = './'
    criterion = torch.nn.CrossEntropyLoss()

    #model X layers X sents
    hidden_reps = {}
    for model_file in model_files:
        hidden_reps[model_file] = {}
        #add layers
        for i in range(2):
            hidden_reps[model_file][i] = []

        print('testing LSTM LM:', model_file)

        #Problem with diff versions of torch
        #model = m.RNNModel('LSTM', 50002, 400, 400, 2, None, 0.2, tie_weights=True)
        model = m.RNNModel('LSTM', 50002, 400, 400, 2, None, 0.2, tie_weights=True).to(device)
        with open(model_file, 'rb') as f:
            loaded_model = torch.load(f, map_location='cpu')
        model.load_state_dict(loaded_model.state_dict())

        #Cancel dropout :)
        model.eval()

        #set to single sentence reading
        multisent_flag = False
        for sent in sents:

            #Create corpus wrapper (this is for one hoting data)
            corpus = lm_data.TestSent(data_path, vocab_file, [sent], 
                                        multisent_flag)

            #Get one hots
            #sent_ids = corpus.get_data()[0].to(torch.device('cpu'))
            sent_ids = corpus.get_data()[0].to(device)
            hidden = model.init_hidden(1)

            data, targets = test_get_batch(sent_ids)

            data = data.unsqueeze(1)

            layer_0 = []
            layer_1 = []
            #Go through word by word and get activations
            for word_index in range(data.size(0)):
                hidden = repackage_hidden(hidden)
                #don't learn
                model.zero_grad()

                word_input = data[word_index]

                #What's going in 
                input_word = corpus.dictionary.idx2word[int(word_input.data)]
                output, hidden = model(torch.tensor([[word_input]]).to(device), hidden)
                if input_word == "<eos>":
                    continue

                #hidden[0] is hidden; hidden[1] is cell state
                h = hidden[0].data
                layer_0.append((input_word, h[0]))
                layer_1.append((input_word, h[1]))

            hidden_reps[model_file][0].append(layer_0)
            hidden_reps[model_file][1].append(layer_1)

    return hidden_reps

def parse(completions):

    nlp = spacy.load("en_core_web_sm")

    scores = {}
    for model in completions:
        scores[model] = []
        for sents in completions[model]:
            num_sg = 0
            total_vb = 0
            for sent, prob in sents:
                cont_word = nlp(sent)[-1]
                pos = cont_word.tag_
                if pos == "VBP":
                    total_vb += prob
                if pos == "VBZ":
                    num_sg += prob
                    total_vb += prob
                if pos == "VBD":
                    if cont_word.text == 'was' or cont_word.text == 'has':
                        num_sg += prob
                        total_vb += prob
                    if cont_word.text == 'were' or cont_word.text == 'have':
                        total_vb += prob
            score = num_sg/total_vb
            scores[model].append(score)
    return scores

def get_RSM(hidden):

    #model X layer X sents -> [(RSM_who, RSM_were)_0, ...]
    RSMS = {}

    for model in hidden:
        RSMS[model] = {}
        for layer in hidden[model]:
            RSMS[model][layer] = []
            for sent in hidden[model][layer]:
                table = get_cossim(sent)
                #if contains who break into two
                if len(table) == 6:
                    who = table[:-1, :-1]
                    was = np.delete(np.delete(table, -2, 0), -2, 1)
                    #extract upper triangle
                    who = who[np.triu_indices_from(who, k=1)]
                    was = was[np.triu_indices_from(was, k=1)]
                    rsms = (who, was)
                else:
                    table = table[np.triu_indices_from(table, k=1)]
                    rsms = (table)
                RSMS[model][layer].append(rsms)
    return RSMS

def return_sims(hidden, target_type='who'):

    SIMS = {}
    for model in hidden:
        SIMS[model] = {}
        for layer in hidden[model]:
            SIMS[model][layer] = []
            for sent in hidden[model][layer]:
                if target_type == 'he':
                    high_word, high_emb = sent[1]
                    low_word, low_emb = sent[-3]
                    t_word, t_emb = sent[-1]
                elif target_type == 'who':
                    if sent[3][0] == 'the':
                        high_word, high_emb = sent[4]
                    else:
                        high_word, high_emb == sent[5]
                    low_word, low_emb = sent[-3]
                    t_word, t_emb = sent[-2]
                else:
                    if sent[3][0] == 'the':
                        high_word, high_emb = sent[4]
                    else:
                        high_word, high_emb == sent[5]
                    low_word, low_emb = sent[-3]
                    t_word, t_emb = sent[-1]

                high_emb = high_emb.cpu().squeeze()
                low_emb = low_emb.cpu().squeeze()
                t_emb = t_emb.cpu().squeeze()
                if high_emb.shape != t_emb.shape:
                    print(high_emb)
                    print(t_emb)

                high_sim = np.corrcoef(high_emb, t_emb)[0, 1]
                low_sim = np.corrcoef(low_emb, t_emb)[0, 1]

                SIMS[model][layer].append((high_sim, low_sim))

    return SIMS

def get_cossim(sent):


    #Get targets
    #Filter out useless stuff
    target_idxs = [1, 2]
    if sent[3][0] == 'the':
        target_idxs.append(4)
    else:
        target_idxs.append(5)

    end = len(sent)-1

    if end-2 not in target_idxs:
        target_idxs.append(end-2)
    if sent[end-1][0] == 'who':
        target_idxs.append(end-1)
    target_idxs.append(end)

    #initalize RSM table
    table = np.full((len(target_idxs), len(target_idxs)), 0.)

    for i in range(len(target_idxs)):
        word_1_idx = target_idxs[i]
        word_1, emb_1 = sent[word_1_idx]
        for j in range(len(target_idxs)):
            word_2_idx = target_idxs[j]
            word_2, emb_2 = sent[word_2_idx]
            sim = float(torch.cosine_similarity(emb_1, emb_2).data)
            table[i, j] = sim

    return table

def get_dummies(fname, gradient=False):

    sent_idx = 0
    IC_idx = 0
    #(high, low, human)_0 ...
    dummies = []
    with open(fname, 'r') as f:

        header = f.readline().strip().split(',')
        for x in range(len(header)):
            head = header[x]
            if head == 'hasIC' or head == 'bias':
                IC_idx = x
            if head == 'sent':
                sent_idx = x

        for line in f:
            line = line.strip().split(',')
            sent = line[sent_idx]
            ic = int(line[IC_idx])

            if line[0] == 'ic_match' or line[0] == 'ic_mismatch':
                human = np.full((4, 4), 0)
                if ic > 0:
                    if not gradient:
                        ic = 1
                    human[1, 3] = ic
                    human[3, 1] = ic
                else:
                    if not gradient:
                        ic = 1
                    human[2, 3] = ic
                    human[3, 2] = ic

                high = np.full((4, 4), 0)
                high[1, 3] = 1
                high[3, 1] = 1

                low = np.full((4, 4), 0)
                low[2, 3] = 1
                low[3, 2] = 1

            else:
                human = np.full((5, 5), 0)

                if ic > 0:
                    if not gradient:
                        ic = 1
                    human[2, 4] = ic
                    human[4, 2] = ic
                else:
                    if not gradient:
                        ic = 1
                    human[3, 4] = ic
                    human[4, 3] = ic

                high = np.full((5, 5), 0)
                high[2, 4] = 1
                high[4, 2] = 1

                low = np.full((5, 5), 0)
                low[3, 4] = 1
                low[4, 3] = 1

            human = human[np.triu_indices_from(human, k=1)]
            high = high[np.triu_indices_from(high, k=1)]
            low = low[np.triu_indices_from(low, k=1)]
            dummies.append((high, low, human))

    return dummies

def run_RSA(RSMS, dummies):

    results = {}
    for model in RSMS:
        results[model] = {}
        for layer in RSMS[model]:
            results[model][layer] = []
            for x in range(len(RSMS[model][layer])):
                embed = RSMS[model][layer][x]
                high, low, human = dummies[x]
                if len(embed) == 2:
                    high_who_rho, high_who_pval = stats.spearmanr(embed[0], high)
                    low_who_rho, low_who_pval = stats.spearmanr(embed[0], low)
                    human_who_rho, human_who_pval = stats.spearmanr(embed[0], human)

                    high_were_rho, high_were_pval = stats.spearmanr(embed[1], high)
                    low_were_rho, low_were_pval = stats.spearmanr(embed[1], low)
                    human_were_rho, human_were_pval = stats.spearmanr(embed[1], human)

                    result = ((high_who_rho, high_who_pval, 
                            low_who_rho, low_who_pval,
                            human_who_rho, human_who_pval), 
                            (high_were_rho, high_were_pval, 
                                low_were_rho, low_were_pval, 
                                human_were_rho, human_were_pval))
                else:
                    high_rho, high_pval = stats.spearmanr(embed, high)
                    low_rho, low_pval = stats.spearmanr(embed, low)
                    human_rho, human_pval = stats.spearmanr(embed, human)
                    result = ((high_rho, high_pval, low_rho, low_pval, human_rho, human_pval),)
                results[model][layer].append(result)
    return results

###################
# CHECK FOR UNKs  #
###################
'''
fname = "stimuli/IC_match.csv"
vocabf = 'wikitext103_vocab'
check_unks(fname, vocabf)
'''

fname = "stimuli/IC_mismatch.csv"
#fname = "stimuli/Reading_Time.csv"
#fname = "stimuli/Story_Completion.csv"
#fname = "stimuli/IC_match.csv"

sents = load_data(fname)

###################
# LSTM LMs Compl  #
###################
'''
vocabf = 'wikitext103_vocab'
lm_models = glob.glob('models/*.pt')#[:1]
lm_models.sort()

sentences = run_lms_completion(sents, vocabf, lm_models)
scores = parse(sentences)

out_str = []
for model in lm_models:
    model = 'LSTM_'+model.split('-')[1].split('_')[-1]+'_score'
    out_str.append(model)

out_str.append('LSTM_avg_score')
out_str = ','.join(out_str)+'\n'

for x in range(len(scores[lm_models[0]])):
    all_scores = []
    for model in lm_models:
        score = scores[model][x]
        all_scores.append(score)

    avg = sum(all_scores)/len(all_scores)
    all_scores.append(avg)
    all_scores = list(map(lambda x: str(x), all_scores))
    out_str += ','.join(all_scores) + '\n'

print(out_str)
'''

###################
#  LSTM LMs Surp  #
###################
'''
vocabf = 'wikitext103_vocab'
lm_models = glob.glob('models/*.pt')[:2]
lm_models.sort()
measures = run_lms(sents, vocabf, lm_models)

out_str = []
for model in lm_models:
    model = 'LSTM_'+model.split('-')[1].split('_')[-1]+'_surp'
    out_str.append(model)

out_str.append('LSTM_avg_surp')
out_str = ','.join(out_str)+'\n'
for x in range(len(measures[lm_models[0]])):
    surps = []
    for model in lm_models:
        word = measures[model][x][0]
        assert word == 'were' or word == 'was' or word == 'he' or word == 'she'
        surp = measures[model][x][1]
        surps.append(surp)
    avg = sum(surps)/len(surps)
    surps.append(avg)
    surps = list(map(lambda x: str(x), surps))
    out_str += ','.join(surps) + '\n'

print(out_str)
'''

###################
#  LSTM LMs RSA   #
###################
'''
vocabf = 'wikitext103_vocab'
lm_models = glob.glob('models/*.pt')#[:1]
lm_models.sort()
hidden = get_lms_hidden(sents, vocabf, lm_models)

if 'IC' in fname:
    SIMS = return_sims(hidden, 'he')

    outname = fname.split('/')[-1].split('.csv')[0]+'_LSTM_SIMS_results.csv'
    save_sims(outname, SIMS, lm_models)

else:
    who_SIMS = return_sims(hidden, 'who')
    were_SIMS = return_sims(hidden, 'was')

    outname = fname.split('/')[-1].split('.csv')[0]+'_LSTM_who_SIMS_results.csv'
    save_sims(outname, who_SIMS, lm_models)
    outname = fname.split('/')[-1].split('.csv')[0]+'_LSTM_were_SIMS_results.csv'
    save_sims(outname, were_SIMS, lm_models)

RSMS = get_RSM(hidden)
dummies = get_dummies(fname)

results = run_RSA(RSMS, dummies)

#save pronoun
outname = fname.split('/')[-1].split('.csv')[0]+'_LSTM_results.csv'
save_results(outname, results, lm_models)

#save who
outname = fname.split('/')[-1].split('.csv')[0]+'_who_results.csv'
save_who_results(outname, results, lm_models)

#save where
outname = fname.split('/')[-1].split('.csv')[0]+'_were_results.csv'
save_were_results(outname, results, lm_models)
'''

###################
#Transformers Compl
###################
'''
tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
tf_model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103', 
                                    output_hidden_states = True)

#turn off learning
tf_model.zero_grad()

sentences = tfxl_completions(sents, tokenizer, tf_model)
scores = parse(sentences)

for model in scores:
    print(model)
    for score in scores[model]:
        print(score)
'''

###################
#Transformers Surps
###################
'''
tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
tf_model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103', 
                                    output_hidden_states = True)

#turn off learning
tf_model.zero_grad()

measures = tfxl_IT(sents, tokenizer, tf_model)

for measure in measures:
    target_word, surp = measure[-1]
    assert target_word == 'was' or target_word == 'were'
    print(surp)
'''
###################
#Transformers RSA #
###################
'''
tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
tf_model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103', 
                                    output_hidden_states = True)

#turn off learning
tf_model.zero_grad()

hidden = get_tf_hidden(sents, tokenizer, tf_model)

if 'IC' in fname:
    SIMS = return_sims(hidden, 'he')

    outname = fname.split('/')[-1].split('.csv')[0]+'_tf_SIMS_results.csv'
    save_sims(outname, SIMS, ['tf'], 'tf')

else:
    who_SIMS = return_sims(hidden, 'who')
    were_SIMS = return_sims(hidden, 'was')

    outname = fname.split('/')[-1].split('.csv')[0]+'_tf_who_SIMS_results.csv'
    save_sims(outname, who_SIMS, ['tf'], 'tf')
    outname = fname.split('/')[-1].split('.csv')[0]+'_tf_were_SIMS_results.csv'
    save_sims(outname, were_SIMS, ['tf'], 'tf')

RSMS = get_RSM(hidden)
dummies = get_dummies(fname)

results = run_RSA(RSMS, dummies)

#save pronoun
outname = fname.split('/')[-1].split('.csv')[0]+'_tf_results.csv'
save_results(outname, results, ['tf'], 'tf')

#save who
outname = fname.split('/')[-1].split('.csv')[0]+'_tf_who_results.csv'
save_who_results(outname, results, ['tf'], 'tf')

#save where
outname = fname.split('/')[-1].split('.csv')[0]+'_tf_were_results.csv'
save_were_results(outname, results, ['tf'], 'tf')
'''

###################
#  GPT-2 XL Compl # 
###################
'''
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
tf_model = GPT2LMHeadModel.from_pretrained('gpt2-xl', 
                                    output_hidden_states = True)

#turn off learning
tf_model.zero_grad()

sentences = tfxl_completions(sents, tokenizer, tf_model)
scores = parse(sentences)

for model in scores:
    print(model)
    for score in scores[model]:
        print(score)
'''

###################
#  GPT-2 XL Surps #
###################
'''
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
tf_model = GPT2LMHeadModel.from_pretrained('gpt2-xl', 
                                    output_hidden_states = True)

#turn off learning
tf_model.zero_grad()

measures = tfxl_IT(sents, tokenizer, tf_model)

with open('gpt_mismatch_surp', 'w') as f:
    for measure in measures:
        target_word, surp = measure[-1]
        #assert target_word == 'was' or target_word == 'were'
        assert target_word == 'he' or target_word == 'she'
        f.write(str(surp)+'\n')
        #print(surp)
'''

###################
#   GPT-2 XL RSA  #
###################
'''
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
tf_model = GPT2LMHeadModel.from_pretrained('gpt2-xl', 
                                    output_hidden_states = True)

#turn off learning
tf_model.zero_grad()

hidden = get_gpt_hidden(sents, tokenizer, tf_model, 48)

if 'IC' in fname:
    SIMS = return_sims(hidden, 'he')

    outname = fname.split('/')[-1].split('.csv')[0]+'_gpt_SIMS_results.csv'
    save_sims(outname, SIMS, ['tf'], 'gpt2')

else:
    who_SIMS = return_sims(hidden, 'who')
    were_SIMS = return_sims(hidden, 'was')

    outname = fname.split('/')[-1].split('.csv')[0]+'_gpt_who_SIMS_results.csv'
    save_sims(outname, who_SIMS, ['tf'], 'gpt2')
    outname = fname.split('/')[-1].split('.csv')[0]+'_gpt_were_SIMS_results.csv'
    save_sims(outname, were_SIMS, ['tf'], 'gpt2')

'''
'''
RSMS = get_RSM(hidden)
dummies = get_dummies(fname)

results = run_RSA(RSMS, dummies)

#save pronoun
outname = fname.split('/')[-1].split('.csv')[0]+'_tf_results.csv'
save_results(outname, results, ['tf'], 'tf')

#save who
outname = fname.split('/')[-1].split('.csv')[0]+'_tf_who_results.csv'
save_who_results(outname, results, ['tf'], 'tf')

#save where
outname = fname.split('/')[-1].split('.csv')[0]+'_tf_were_results.csv'
save_were_results(outname, results, ['tf'], 'tf')
'''
