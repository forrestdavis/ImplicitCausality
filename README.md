# ImplicitCausality

Project for exploring the acquisiton of discourse structure by RNN LMs and Transformers. The code centers on an exploration of the well studied 
phenomenon of implicit causality (IC) in verbs ([Catherine Garvey & Alfonso Caramazza, 1974](www.jstor.org/stable/4177835)). 
We explored the influence of IC on reference and syntax. For reference, 
we utilized the experimental stimuli from [Ferstl et al. (2010)](https://link.springer.com/article/10.3758/s13428-010-0023-2), 
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
* [pytorch](https://pytorch.org/) >= v1.0.0
* [scipy] (https://www.scipy.org)
* [numpy](https://numpy.org)
* [transformers] (https://github.com/huggingface/transformers)
* [spaCy](https://spacy.io) v2.2.4

From spaCy you need the pretrained English model "en_core_web_sm":

    python -m spacy download en_core_web_sm

## Usage



## Extra Details


## References
