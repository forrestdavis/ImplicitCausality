# ImplicitCausality

There are two projects that draw from this repo: BERT/RoBERTa Models for Implicit Causality and Auto-Regressive Models for Implicit Causality. They were created some time apart so they make use of different versions of huggingface. 
The first refers to my work at ACL 2021: ["Uncovering Constraint-Based Behavior in Neural Models via Targeted Fine-Tuning"](https://aclanthology.org/2021.acl-long.93/). The second refers to my work at CoNLL 2020: ["Discourse structure interacts with reference but not syntax in neural language models"](https://www.aclweb.org/anthology/2020.conll-1.32/). Both projects center on an exploration of the well studied 
phenomenon of implicit causality (IC) in verbs ([Catherine Garvey & Alfonso Caramazza, 1974](www.jstor.org/stable/4177835)). 

## For BERT and RoBERTa Models (ACL 2021)

Project for exploring whether BERT and RoBERTa models learn the same discourse structure, Implicit Causality, across 4 languages: English, Chinese, Italian, and Spanish. We find that another process, namely pro-drop, can compete with Implicit Causality (IC)
in a given language to obscure underlying knowledge of IC. Targeted finetuning that demotes this pro-drop process (ie evidence that pro-drop doesn't apply) can uncover this knowledge. 

### Dependencies 
Requires the following python packages (available through pip):
* [pytorch](https://pytorch.org/) == 1.8.1
* [transformers](https://github.com/huggingface/transformers) == 4.6.1
* [pandas](https://pandas.pydata.org/)

## Usage
To recreate the experiments use bert.py:

    usage: bert.py [-h] [--lang LANG] [--model MODEL] [--exp EXP]

    Implicit Causality Experiments for (Ro)BERT(a) Models

    optional arguments:
      -h, --help     show this help message and exit
      --lang LANG    language to evaluate [en|zh|it|es]
      --model MODEL  model to run [bert|roberta|gilberto|umberto|mbert|all]
      --exp EXP      Experiment to run [og|base|pro|all]

You will need to download the pretrained models from [here](http://doi.org/10.5281/zenodo.4798711) and put the extracted directory, models, in the 
directory finetuning. 
There are three main experiment groups under --exp. og refers to the base BERT/RoBERTa models, base to the baseline
fine-tuned models reported in the paper, pro to the models fine-tuned with/without pro-drop, all runs all the 
experiments. To recreate the figures open bert\_figures.R in R and run the relevant blocks. Precompiled results 
for all the experiments are given, per language, in the results directory. Stimuli are given in the stimuli directory, 
per language. 

Say you want to recreate Figure 4 from the paper. First run

        bert.py --lang it --model all --exp pro && bert.py --lang es --model all --exp pro

This may take a bit of time, depending on your compute resources, but it runs fine on my not so great laptop. The
output of the above code will be two files in the results directory: IC\_mismatch\_IT.csv and IC\_mismatch\_ES.csv
Open bert\_figures.R in R. Run the first 30 lines and then the block starting at line 411 will make the relevant 
figure. 

If you want to recreate the finetuned models, add the finetuning directory to your Google Drive and run the collab
script (finetune.ipynb). The fine-tuning data can be recreated from add\_pro.py. The source conllu files are included
for ease of use, see the paper for more details about the data. 

## For Auto-Regressive Models (CoNLL 2020)

Project for exploring the acquisiton of discourse structure by RNN LMs and Transformers. The code centers on an exploration of the well studied 
phenomenon of implicit causality (IC) in verbs ([Catherine Garvey & Alfonso Caramazza, 1974](www.jstor.org/stable/4177835)). 
We explored the influence of IC on reference and syntax. For reference, 
we utilized the experimental stimuli from [Ferstl et al. (2011)](https://link.springer.com/article/10.3758/s13428-010-0023-2), 
which scored 305 verbs on IC bias for pronimal continuations. 
For syntax, we utilized the experimental stimuli from [Rohde et al. (2011)](https://www.sciencedirect.com/science/article/abs/pii/S0010027710002532?via%3Dihub), which looked 
at sentence completion and self-paced reading. 

The main finding is that LSTM LMs are unable to acquire IC, but 
transformer models (TransformerXL ([Dai et al., 2019](https://doi.org/10.18653/v1/P19-1285)) and GPT-2 XL ([Radford et al., 2019](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)))
do. For both reference and syntax, the hidden representations of these
transformers distinguish cases of IC. Strikingly, IC only influences model behavior (i.e., surprisal) 
for reference, not syntactic attachment pointing to a disconnect between 
model representations of syntax and model behavior. Moreover, we find 
that the final layers of the transformer LMs override unambiguous 
syntactic information (e.g., agreement), instead showing a preference 
for a more local, yet ungrammatical, attachment. 

### Dependencies 

Requires the following python packages (available through pip):
* [pytorch](https://pytorch.org/) <= v1.2.0
* [scipy](https://www.scipy.org)
* [numpy](https://numpy.org)
* [transformers](https://github.com/huggingface/transformers) = 2.11.0
* [spaCy](https://spacy.io) v2.2.4

From spaCy you need the pretrained English model "en_core_web_sm":

    python -m spacy download en_core_web_sm

## Usage

To recreate experiments, download the [LSTM models](https://doi.org/10.5281/zenodo.4053572) from Zenodo and
 uncomment the relevant code block at the bottom of main.py. Pre-trained
transformer models are downloaded via HuggingFace's interface. 

Pre-generated results are included in the results dir. Figures from the 
paper can be found in figures. R code to recreate those figures 
as well as the statistical tests in the paper can be found 
in stats.R. 

The dir tools includes a script (create_stim.py) which recreates the 
stimuli used in this experiment. All stimuli are taken directly 
from human experiments, but the stim script allows one to tweak 
various components of the stimuli (e.g., the nouns). Also 
included is a script used for printing out the results from main.py 
(utils.py), an awk script for getting frequencies, and
information about replaced nouns and missing 
IC verbs. 

The vocab for the LSTM LMs is given in wikitext103_vocab (top 50K words). 
If you have any questions feel free to email me :)

## References

Forrest Davis and Marten van Schijndel. ["Uncovering Constraint-Based Behavior in Neural Models via Targeted Fine-Tuning".](https://aclanthology.org/2021.acl-long.93/) In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL 2021). 2021.

Forrest Davis and Marten van Schijndel. ["Discourse structure interacts with reference but not syntax in neural language models".](https://www.aclweb.org/anthology/2020.conll-1.32/) In Proceedings of the 2020 Conference on Computational Natural Language Learning (CoNLL 2020). 2020.
