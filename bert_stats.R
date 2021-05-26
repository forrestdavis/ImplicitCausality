path = "/home/forrestdavis/Projects/ImplicitCausality/"

bert_en_score <- read.csv(paste(path, "results/IC_mismatch_BERT.csv", sep=''))
bert_es_score <- read.csv(paste(path, "results/IC_mismatch_ES.csv", sep=''))
bert_it_score <- read.csv(paste(path, "results/IC_mismatch_IT.csv", sep=''))
bert_zh_score <- read.csv(paste(path, "results/IC_mismatch_ZH.csv", sep=''))

bert_en_score$hasIC <- as.numeric(bert_en_score$bias>0)
bert_en_score$isHigh <- factor(bert_en_score$isHigh)
bert_en_score$gender <- factor(bert_en_score$gender)
bert_en_score$hasIC <- factor(bert_en_score$hasIC)

bert_es_score$hasIC <- as.numeric(bert_es_score$bias>50)
bert_es_score$isHigh <- factor(bert_es_score$isHigh)
bert_es_score$gender <- factor(bert_es_score$gender)
bert_es_score$hasIC <- factor(bert_es_score$hasIC)

bert_it_score$hasIC <- as.numeric(bert_it_score$bias>50)
bert_it_score$isHigh <- factor(bert_it_score$isHigh)
bert_it_score$gender <- factor(bert_it_score$gender)
bert_it_score$hasIC <- factor(bert_it_score$hasIC)

bert_zh_score$hasIC <- as.numeric(bert_zh_score$bias>50)
bert_zh_score$isHigh <- factor(bert_zh_score$isHigh)
bert_zh_score$gender <- factor(bert_zh_score$gender)
bert_zh_score$hasIC <- factor(bert_zh_score$hasIC)

###swap in the relevant data
OBJ_IC <- subset(bert_it_score, isHigh == 0 & hasIC == 1)
OBJ_nonIC <- subset(bert_it_score, isHigh == 0 & hasIC == 0)

SUBJ_IC <- subset(bert_it_score, isHigh == 1 & hasIC == 1)
SUBJ_nonIC <- subset(bert_it_score, isHigh == 1 & hasIC == 0)

t.test(OBJ_nonIC$score_bert_base, OBJ_IC$score_bert_base)
t.test(SUBJ_IC$score_bert_base, SUBJ_nonIC$score_bert_base)
