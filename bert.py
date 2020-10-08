import math
import sys
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, BertTokenizer, BertModel
from transformers import pipeline
import spacy 

device = torch.device("cuda:1")

def load_data(fname, mask=None, lower=True):

    sents = []

    with open(fname, 'r') as f:

        header = f.readline().split(',')

        idx = 0
        for x in range(len(header)):
            if header[x] == 'sent':
                idx = x

        for line in f:
            sent = line.strip().split(',')[idx]
            if lower:
                sent = sent.lower()

            if mask:
                sent = sent.replace('[mask]', mask)
            else:
                sent = sent.replace('[mask]', '[MASK]')
            #sent = sent.replace('there', "outside")
            sents.append(sent)

    return sents

def get_BERT_cont(sents, model, lang='es', topk=100):
    tokenizer = AutoTokenizer.from_pretrained(model)
    unmasker = pipeline('fill-mask', model=model, tokenizer=model, framework='pt', topk=topk)
    scores = []
    if lang == 'es':
        nlp = spacy.load('es_core_news_lg')
    else:
        nlp = spacy.load('it_core_news_sm')
    for sent in sents:
        result = unmasker(sent)
        male_score, female_score = get_genders(result, nlp, lang)
        scores.append(male_score)
        scores.append(female_score)
    return scores

def get_genders(results, nlp, lang='es'):

    f_prob = 0
    m_prob = 0
    for result in results:
        score = result['score']
        if lang == 'es':
            sent = ' '.join(result['sequence'].split(' ')[1:-1])
        else:
            sent = ' '.join(result['sequence'].replace('.', '. ').split(' ')[1:-1])
        parsed = nlp(sent)
        if "Gender=Fem" in parsed[-2].tag_:
            f_prob += score
        elif "Gender=Masc" in parsed[-2].tag_:
            m_prob += score

    #t_prob = f_prob+m_prob
    #f_prob = f_prob/t_prob
    #m_prob = m_prob/t_prob
    return m_prob, f_prob

def get_BERT_scores(sents, model, he, she, topk=10):

    tokenizer = AutoTokenizer.from_pretrained(model)
    he_token = tokenizer.encode(he)[-2]
    she_token = tokenizer.encode(she)[-2]

    unmasker = pipeline('fill-mask', model=model, tokenizer=model, framework='pt', topk=topk)
    scores = []
    for sent in sents:
        result = unmasker(sent)
        #for r in result:
            #print(r['sequence'])
        #print(result)
        try:
            he_result = list(filter(lambda x: x['token'] == he_token, result))[0]['score']
        except:
            he_result = 0
        try:
            she_result = list(filter(lambda x: x['token'] == she_token, result))[0]['score']
        except:
            she_result = 0
        scores.append(he_result)
        scores.append(she_result)

    return scores


def run_BERT_RSA(stim_file, layer, header=False, filter_file=None):

    EXP = data.Stim(stim_file, header, filter_file)

    #Load BERT uncased 
    pretrained_weights = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

    model = BertModel.from_pretrained(pretrained_weights, 
                                        output_hidden_states=True)
    model.eval()
    model.zero_grad()

    for x in range(len(EXP.SENTS)):
        sentences = list(EXP.SENTS[x])

        target = sentences[0]
        sentences = sentences[1:]

        #GET BASELINE
        target_encoded = tokenizer.encode(target)
        target_ids = torch.tensor(target_encoded).unsqueeze(0)

        hidden_states = model(target_ids)[-1]
        embed, hidden_states = hidden_states[:1], hidden_states[1:]

        hidden_states = hidden_states[layer][0]
        
        baseline_word = tokenizer.decode(torch.tensor([target_encoded[-2]])).strip()

        baseline = hidden_states[-2].data.cpu().squeeze()

        sims = get_BERT_sims(sentences[0], layer, baseline, tokenizer, model)
        values = get_dummy_values(sentences[0])

        EXP.load_IT('bert-uncased', x, values, False, sims)

    return EXP

def get_BERT_sims(sent, layer, baseline, tokenizer, model):

    model.zero_grad()

    encoded = tokenizer.encode(sent)
    input_ids = torch.tensor(encoded).unsqueeze(0)

    hidden_states = model(input_ids)[-1]
    embed, hidden_states = hidden_states[:1], hidden_states[1:]

    hidden_states = hidden_states[layer][0]

    #skip over [CLS] [SEP]
    hidden_states = hidden_states[1:-1]

    encoded = encoded[1:-1]

    sent_words = sent.split(' ')

    SIMS = []
    idx = 0

    sims = []
    #offset if word is broken down
    for y in range(len(sent_words)):

        sent_word = sent_words[y]
        h_idx = encoded[idx]
        input_word = tokenizer.decode(torch.tensor([h_idx])).strip()
        #replace tokenizer flag
        input_word = input_word.replace("##", '')

        if input_word != sent_word:
            while not(sent_word.index(input_word)+len(input_word) == len(sent_word)):
                idx += 1
                h_idx = encoded[idx]
                input_word = tokenizer.decode(torch.tensor([h_idx])).strip()
                #replace tokenizer flag
                input_word = input_word.replace("##", '')

        h = hidden_states[idx].unsqueeze(0).data.cpu().squeeze()

        assert len(h) == len(baseline)

        sim = np.corrcoef(baseline, h)[0, 1]

        sims.append((sent_word, sim))

        idx += 1

    SIMS.append(sims)

    return SIMS

def get_dummy_values(sent):

    values = []

    metrics = []
    for word in sent.split(' '):
        metrics.append((word, 99999, 99999))
    values.append(metrics)


    return values

if __name__ == "__main__":

    #stim_file = 'stimuli/large_degree.xlsx'
    #header = True
    #layer = 0

    #EXP = run_BERT_RSA(stim_file, layer, header)

    #sents = ['the man frightens the woman because [MASK] was there.']
    #sents = ["el hombre alcanzó a la mujer porque estaba [MASK].", "la mujer alcanzó al hombre porque estaba [MASK]."]
    #stim_file = 'stimuli/IC_mismatch_BERT.csv'
    #stim_file = 'stimuli/IC_mismatch_ES.csv'
    stim_file = 'stimuli/IC_mismatch_ES_cont.csv'
    sents = load_data(stim_file)
    #print(sents[:2])
    #English
    #model = 'bert-base-uncased'
    #Italian
    #model="Musixmatch/umberto-wikipedia-uncased-v1",
    #Spanish
    #model = "dccuchile/bert-base-spanish-wwm-uncased"
    #Dutch
    #model = "wietsedv/bert-base-dutch-cased"

    #scores = get_BERT_scores(sents, model, 'he', 'she')
    #scores = get_BERT_scores(sents, model, 'él', 'ella')
    #for score in scores:
    #    print(score)
    scores = get_BERT_cont(sents, model)
    for score in scores:
        print(score)
