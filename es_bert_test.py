from bert import *

#Spanish
model = "dccuchile/bert-base-spanish-wwm-uncased"

sents = ['el sobrino cenó ayer con la reina del amigo que estaba [MASK].']
sents = ['el sobrino cenó ayer con la reina del amigo que estaba [MASK].', 'el sobrino cenó ayer con el amigo de la reina que estaba [MASK].',
         'la sobrina cenó ayer con la reina del amigo que estaba [MASK].', 'la sobrina cenó ayer con el amigo de la reina que estaba [MASK].']

sents = ['el sobrino cenó ayer con las reinas del amigo que estaban [MASK].', 'el sobrino cenó ayer con los amigos de la reina que estaban [MASK].',
         'la sobrina cenó ayer con las reina del amigo que estaban [MASK].', 'la sobrina cenó ayer con los amigos de la reina que estaban [MASK].']

#Agrees with female and number
sents = ['el sobrino cenó ayer con las reinas del amigo que [MASK] altas.', 'el sobrino cenó ayer con los amigos de la reina que [MASK] alta.',
         'la sobrina cenó ayer con las reina del amigo que [MASK] altas.', 'la sobrina cenó ayer con los amigos de la reina que [MASK] alta.']

#Agrees with male and number
sents = ['el sobrino cenó ayer con las reinas del amigo que [MASK] alto.', 'el sobrino cenó ayer con los amigos de la reina que [MASK] altos.',
         'la sobrina cenó ayer con las reina del amigo que [MASK] alto.', 'la sobrina cenó ayer con los amigos de la reina que [MASK] altos.']

#Agrees with full factorial
sents = ['el sobrino cenó ayer con las reinas del amigo que [MASK] altas.',
        'el sobrino cenó ayer con la reina de los amigos que [MASK] alta.',
        'el sobrino cenó ayer con las reinas del amigo que [MASK] alto.',
        'el sobrino cenó ayer con la reina de los amigos que [MASK] altos.',
        'el sobrino cenó ayer con los amigos de la reina que [MASK] alta.',
        'el sobrino cenó ayer con el amigo de las reinas que [MASK] altas.',
        'el sobrino cenó ayer con los amigos de la reina que [MASK] altos.',
        'el sobrino cenó ayer con el amigo de las reinas que [MASK] alto.']
scores = get_BERT_cont(sents, model, 'es', 10, 'num')

print(sents)
i = 0
for x, score in enumerate(scores):
    if x%2 == 0:
        print(sents[i])
        print('Singular', score)
        i += 1
    else:
        print('Plural', score)
