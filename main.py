import torch
import numpy as np
from transformers import TransfoXLTokenizer, TransfoXLLMHeadModel

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

def get_surps(state):
    logprobs = torch.log2(torch.nn.functional.softmax(state, dim=0))
    return -1 * logprobs

def get_IT(state, obs, tokenizer):

    metrics = []

    surps = get_surps(state)

    for sentpos, targ in enumerate(obs):

        word = tokenizer.decode(int(targ))
        print(sentpos, targ, word)

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
        if "<eos>" not in sent:
            sent = "<eos> "+sent+ " <eos>"

        encoded = tf_tokenizer.encode(sent)

        sent_id = torch.tensor(encoded)
        decoded = tf_tokenizer.decode(encoded)

        assert decoded == sent

        input_ids.append(sent_id)

    return input_ids

def tfxl_IT(data_source, tokenizer, model):
    
    values = []
    for i in range(len(data_source)):
        sent_ids = data_source[i]
        
        data, target = test_get_batch(sent_ids)

        data = data.unsqueeze(1)

        output = model(data)

        metrics = get_IT(output, target, tokenizer)


fname = "stimuli/IC_match.csv"
vocabf = 'wikitext103_vocab'
check_unks(fname, vocabf)

sents = ['The man admires the agent of the rockers who was']

tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
tf_model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')

#turn off learning
tf_model.zero_grad()

input_ids = tfxl_tokenize(tokenizer, sents)

tfxl_IT(input_ids, tokenizer, tf_model)



'''


sg = torch.tensor([tokenizer.encode('is')])
pl = torch.tensor([tokenizer.encode('are')])

sent = "<eos> The author met the agent of the rockers who is happy"
sent_idx = torch.tensor([tokenizer.encode(sent, add_special_tokens=True)])
print(tokenizer.decode(tokenizer.encode(sent)))

print(sent_idx)
outputs = tf_model(sent_idx)
prediction_scores, mems = outputs[:2]
surps = get_surps(prediction_scores[0][-2])
print(surps[int(sg)])
print(surps[int(pl)])

sent = "The author met the agents of the rocker who are"
sent_idx = torch.tensor([tokenizer.encode(sent, add_special_tokens=True)])

print(sent_idx)
outputs = tf_model(sent_idx)
prediction_scores, mems = outputs[:2]
surps = get_surps(prediction_scores[0][-2])
print(surps[int(sg)])
print(surps[int(pl)])

sent = "The author aldfkjadlfj food"
sent_idx = torch.tensor([tokenizer.encode(sent, add_special_tokens=True)])

print(sent_idx)
outputs = tf_model(sent_idx)
prediction_scores, mems = outputs[:2]
surps = get_surps(prediction_scores[0][0])
print(surps)

sent = "The author is"
sent_idx = torch.tensor([tokenizer.encode(sent, add_special_tokens=True)])

print(sent_idx)
'''
