import warnings
warnings.filterwarnings("ignore")
import torch
from transformers import AutoTokenizer, BertTokenizer, BertModel
from transformers import pipeline
import pandas as pd

device = torch.device("cuda:0")

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

            sents.append(sent)

    return sents

def save_results(stim_file, outfilename, results):

    stim = pd.read_csv(stim_file)
    for heading in results:
        stim[heading] = results[heading]

    stim.to_csv(outfilename, index=False)

def get_BERT_scores(sents, model, he, she=None, topk=10):

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    he_token = tokenizer.encode(he)[-2]
    if she:
        she_token = tokenizer.encode(she)[-2]

    unmasker = pipeline('fill-mask', model=model, tokenizer=model, framework='pt', top_k=topk)
    scores = []
    for sent in sents:
        result = unmasker(sent)
        try:
            he_result = list(filter(lambda x: x['token'] == he_token, result))[0]['score']
        except:
            he_result = 0
        scores.append(he_result)
        if she:
            try:
                she_result = list(filter(lambda x: x['token'] == she_token, result))[0]['score']
            except:
                she_result = 0
            scores.append(she_result)

    return scores

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Implicit Causality Experiments for (Ro)BERT(a) Models')

    parser.add_argument('--lang', type=str, default='en',
                        help='language to evaluate [en|zh|it|es]')
    parser.add_argument('--model', type=str, default='bert',
                        help='model to run [bert|roberta|gilberto|umberto|mbert|all]')
    parser.add_argument('--exp', type=str, default='og', 
                        help='Experiment to run [og|base|pro|all]')

    args = parser.parse_args()

    #header X values
    RESULTS = {}

    #English
    if args.lang == 'en':
        stim_file = 'stimuli/IC_mismatch_BERT.csv'
        he = 'he'
        she = 'she'
        if args.model == 'bert' or args.model == 'all':
            sents = load_data(stim_file)
            if args.exp == 'og' or args.exp == 'all':
                model = 'bert-base-uncased'
                print(model)
                scores = get_BERT_scores(sents, model, he, she)
                RESULTS['score_bert'] = scores
            if args.exp == 'base' or args.exp == 'all':
                model = './finetuning/models/ProBERT_BASE'
                print(model)
                scores = get_BERT_scores(sents, model, he, she)
                RESULTS['score_bert_base'] = scores
            if args.exp == 'pro' or args.exp == 'all':
                model = './finetuning/models/ProBERT'
                print(model)
                scores = get_BERT_scores(sents, model, he, she)
                RESULTS['score_bert_pro'] = scores
        if args.model == 'mbert' or args.model == 'all':
            sents = load_data(stim_file)
            model = 'bert-base-multilingual-uncased'
            print(model)
            scores = get_BERT_scores(sents, model, he, she)
            RESULTS['score_mbert'] = scores

        if args.model == 'roberta' or args.model == 'all':
            sents = load_data(stim_file, '<mask>')
            if args.exp == 'og' or args.exp == 'all':
                model = 'roberta-base'
                print(model)
                scores = get_BERT_scores(sents, model, he, she)
                RESULTS['score_roberta'] = scores
            if args.exp == 'base' or args.exp == 'all':
                model = './finetuning/models/ProRoBERTa_BASE'
                print(model)
                scores = get_BERT_scores(sents, model, he, she)
                RESULTS['score_roberta_base'] = scores
            if args.exp == 'pro' or args.exp == 'all':
                model = './finetuning/models/ProRoBERTa'
                print(model)
                scores = get_BERT_scores(sents, model, he, she)
                RESULTS['score_roberta_pro'] = scores

        template_file = 'results/IC_mismatch_template_EN.csv'
        save_results(template_file, 'results/IC_mismatch_EN.csv', RESULTS)

    #Chinese
    if args.lang == 'zh':
        stim_file = 'stimuli/IC_mismatch_ZH.csv'
        he = '他'
        she = '她'
        sents = load_data(stim_file)
        if args.model == 'bert' or args.model == 'all':
            if args.exp == 'og' or args.exp == 'all':
                model = 'hfl/chinese-bert-wwm-ext'
                print(model)
                scores = get_BERT_scores(sents, model, he, she)
                RESULTS['score_bert'] = scores
            if args.exp == 'base' or args.exp == 'all':
                model = './finetuning/models/ProCHINESE_BASE'
                print(model)
                scores = get_BERT_scores(sents, model, he, she)
                RESULTS['score_bert_base'] = scores
            if args.exp == 'pro' or args.exp == 'all':
                model = './finetuning/models/ProCHINESE'
                print(model)
                scores = get_BERT_scores(sents, model, he, she)
                RESULTS['score_bert_pro'] = scores
        if args.model == 'mbert' or args.model == 'all':
            sents = load_data(stim_file)
            model = 'bert-base-multilingual-uncased'
            print(model)
            scores = get_BERT_scores(sents, model, he, she)
            RESULTS['score_mbert'] = scores

        if args.model == 'roberta' or args.model == 'all':
            if args.exp == 'og' or args.exp == 'all':
                model = 'hfl/chinese-roberta-wwm-ext'
                print(model)
                scores = get_BERT_scores(sents, model, he, she)
                RESULTS['score_roberta'] = scores
            if args.exp == 'base' or args.exp == 'all':
                model = './finetuning/models/ProROCHINESE_BASE'
                print(model)
                scores = get_BERT_scores(sents, model, he, she)
                RESULTS['score_roberta_base'] = scores
            if args.exp == 'pro' or args.exp == 'all':
                model = './finetuning/models/ProROCHINESE'
                print(model)
                scores = get_BERT_scores(sents, model, he, she)
                RESULTS['score_roberta_pro'] = scores

        template_file = 'results/IC_mismatch_template_ZH.csv'
        save_results(template_file, 'results/IC_mismatch_ZH.csv', RESULTS)

    #Italian
    if args.lang == 'it':
        stim_file = 'stimuli/IC_mismatch_IT.csv'
        he = 'lui'
        she = 'lei'
        if args.model == 'bert' or args.model == 'all':
            sents = load_data(stim_file)
            if args.exp == 'og' or args.exp == 'all':
                model = 'dbmdz/bert-base-italian-uncased'
                print(model)
                scores = get_BERT_scores(sents, model, he, she)
                RESULTS['score_bert'] = scores
            if args.exp == 'base' or args.exp == 'all':
                model = './finetuning/models/ProITALIAN_BASE'
                print(model)
                scores = get_BERT_scores(sents, model, he, she)
                RESULTS['score_bert_base'] = scores
            if args.exp == 'pro' or args.exp == 'all':
                model = './finetuning/models/ProITALIAN'
                print(model)
                scores = get_BERT_scores(sents, model, he, she)
                RESULTS['score_bert_pro'] = scores
        if args.model == 'mbert' or args.model == 'all':
            sents = load_data(stim_file)
            model = 'bert-base-multilingual-uncased'
            print(model)
            scores = get_BERT_scores(sents, model, he, she)
            RESULTS['score_mbert'] = scores

        if args.model == 'umberto' or args.model == 'all':
            sents = load_data(stim_file, '<mask>')
            if args.exp == 'og' or args.exp == 'all':
                model="Musixmatch/umberto-wikipedia-uncased-v1"
                print(model)
                scores = get_BERT_scores(sents, model, he, she)
                RESULTS['score_umberto'] = scores
            if args.exp == 'base' or args.exp == 'all':
                model = './finetuning/models/ProUmBERTo_BASE'
                print(model)
                scores = get_BERT_scores(sents, model, he, she)
                RESULTS['score_umberto_base'] = scores
            if args.exp == 'pro' or args.exp == 'all':
                model = './finetuning/models/ProUmBERTo'
                print(model)
                scores = get_BERT_scores(sents, model, he, she)
                RESULTS['score_umberto_pro'] = scores

        if args.model == 'gilberto' or args.model == 'all':
            sents = load_data(stim_file, '<mask>')
            if args.exp == 'og' or args.exp == 'all':
                model = "idb-ita/gilberto-uncased-from-camembert"
                print(model)
                scores = get_BERT_scores(sents, model, he, she)
                RESULTS['score_gilberto'] = scores
            if args.exp == 'base' or args.exp == 'all':
                model = './finetuning/models/ProGilBERTo_BASE'
                print(model)
                scores = get_BERT_scores(sents, model, he, she)
                RESULTS['score_gilberto_base'] = scores
            if args.exp == 'pro' or args.exp == 'all':
                model = './finetuning/models/ProGilBERTo'
                print(model)
                scores = get_BERT_scores(sents, model, he, she)
                RESULTS['score_gilberto_pro'] = scores

        template_file = 'results/IC_mismatch_template_IT.csv'
        save_results(template_file, 'results/IC_mismatch_IT.csv', RESULTS)

    #Spanish
    if args.lang == 'es':
        stim_file = 'stimuli/IC_mismatch_ES.csv'
        he = 'él'
        she = 'ella'
        if args.model == 'bert' or args.model == 'all':
            sents = load_data(stim_file)
            if args.exp == 'og' or args.exp == 'all':
                model = "dccuchile/bert-base-spanish-wwm-uncased"
                print(model)
                scores = get_BERT_scores(sents, model, he, she)
                RESULTS['score_bert'] = scores
            if args.exp == 'base' or args.exp == 'all':
                model = './finetuning/models/ProBETO_BASE'
                print(model)
                scores = get_BERT_scores(sents, model, he, she)
                RESULTS['score_bert_base'] = scores
            if args.exp == 'pro' or args.exp == 'all':
                model = './finetuning/models/ProBETO'
                print(model)
                scores = get_BERT_scores(sents, model, he, she)
                RESULTS['score_bert_pro'] = scores

        if args.model == 'mbert' or args.model == 'all':
            sents = load_data(stim_file)
            model = 'bert-base-multilingual-uncased'
            print(model)
            scores = get_BERT_scores(sents, model, he, she)
            RESULTS['score_mbert'] = scores

        if args.model == 'roberta' or args.model == 'all':
            sents = load_data(stim_file, '<mask>')
            if args.exp == 'og' or args.exp == 'all':
                model = "mrm8488/RuPERTa-base"
                print(model)
                scores = get_BERT_scores(sents, model, he, she)
                RESULTS['score_roberta'] = scores
            if args.exp == 'base' or args.exp == 'all':
                model = './finetuning/models/ProRuPERTa_BASE'
                print(model)
                scores = get_BERT_scores(sents, model, he, she)
                RESULTS['score_roberta_base'] = scores
            if args.exp == 'pro' or args.exp == 'all':
                model = './finetuning/models/ProRuPERTa'
                print(model)
                scores = get_BERT_scores(sents, model, he, she)
                RESULTS['score_roberta_pro'] = scores

        template_file = 'results/IC_mismatch_template_ES.csv'
        save_results(template_file, 'results/IC_mismatch_ES.csv', RESULTS)
