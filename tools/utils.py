def save_sims(outname, results, models, model_type = 'LSTM'):

    out_str = []
    num_layers = len(results[models[0]])
    #Header
    if model_type == 'LSTM':
        for i in range(num_layers):
            for model in models:
                m = 'LSTM_'+model.split('-')[1].split('_')[-1]+'_layer_'+str(i)+'_HIGH_SIM'
                out_str.append(m)

                m = 'LSTM_'+model.split('-')[1].split('_')[-1]+'_layer_'+str(i)+'_LOW_SIM'
                out_str.append(m)

            out_str.append('LSTM_avg_layer_'+str(i)+'_HIGH_SIM')
            out_str.append('LSTM_avg_layer_'+str(i)+'_LOW_SIM')
    else:
        for i in range(len(results[models[0]])):
            for model in results:
                m = 'tf_layer_'+str(i)+'_HIGH_SIM' 
                out_str.append(m)

                m = 'tf_layer_'+str(i)+'_LOW_SIM' 
                out_str.append(m)

    out_str = ','.join(out_str)+'\n'

    data = []

    out = ''
    for x in range(len(results[models[0]][0])):
        all_out = []
        for i in range(num_layers):
            high_rho = []
            low_rho = []
            human_rho = []
            for model in models:
                measures = results[model][i][x]

                high_r = measures[0]
                high_rho.append(high_r)
                all_out.append(str(high_r))

                low_r = measures[1]
                low_rho.append(low_r)
                all_out.append(str(low_r))

            if model_type == "LSTM":
                all_out.append(str(sum(high_rho)/len(high_rho)))
                all_out.append(str(sum(low_rho)/len(low_rho)))
        out_str += ','.join(all_out) + '\n'

    with open(outname, 'w') as f:
        f.write(out_str)

def save_results(outname, results, models, model_type ='LSTM'):

    out_str = []
    num_layers = len(results[models[0]])
    multi = len(results[models[0]][0][0])
    #Header
    if model_type == 'LSTM':
        for i in range(num_layers):
            for model in models:
                m = 'LSTM_'+model.split('-')[1].split('_')[-1]+'_layer_'+str(i)+'_HIGH_RSA'
                out_str.append(m)
                m = 'LSTM_'+model.split('-')[1].split('_')[-1]+'_layer_'+str(i)+'_HIGH_pvalue'
                out_str.append(m)

                m = 'LSTM_'+model.split('-')[1].split('_')[-1]+'_layer_'+str(i)+'_LOW_RSA'
                out_str.append(m)
                m = 'LSTM_'+model.split('-')[1].split('_')[-1]+'_layer_'+str(i)+'_LOW_pvalue'
                out_str.append(m)

                m = 'LSTM_'+model.split('-')[1].split('_')[-1]+'_layer_'+str(i)+'_HUMAN_RSA'
                out_str.append(m)
                m = 'LSTM_'+model.split('-')[1].split('_')[-1]+'_layer_'+str(i)+'_HUMAN_pvalue'
                out_str.append(m)

            out_str.append('LSTM_avg_layer_'+str(i)+'_HIGH_RSA')
            out_str.append('LSTM_avg_layer_'+str(i)+'_LOW_RSA')
            out_str.append('LSTM_avg_layer_'+str(i)+'_HUMAN_RSA')
    else:
        for i in range(len(results[models[0]])):
            for model in results:
                m = 'tf_layer_'+str(i)+'_HIGH_RSA' 
                out_str.append(m)
                m = 'tf_layer_'+str(i)+'_HIGH_pvalue' 
                out_str.append(m)

                m = 'tf_layer_'+str(i)+'_LOW_RSA' 
                out_str.append(m)
                m = 'tf_layer_'+str(i)+'_LOW_pvalue' 
                out_str.append(m)

                m = 'tf_layer_'+str(i)+'_HUMAN_RSA' 
                out_str.append(m)
                m = 'tf_layer_'+str(i)+'_HUMAN_pvalue' 
                out_str.append(m)

    out_str = ','.join(out_str)+'\n'

    data = []

    out = ''
    for x in range(len(results[models[0]][0])):
        all_out = []
        for i in range(num_layers):
            high_rho = []
            low_rho = []
            human_rho = []
            for model in models:
                measures = results[model][i][x][0]

                high_r = measures[0]
                high_rho.append(high_r)
                high_p = measures[1]
                all_out.append(str(high_r))
                all_out.append(str(high_p))

                low_r = measures[2]
                low_rho.append(low_r)
                low_p = measures[3]
                all_out.append(str(low_r))
                all_out.append(str(low_p))

                human_r = measures[4]
                human_rho.append(human_r)
                human_p = measures[5]
                all_out.append(str(human_r))
                all_out.append(str(human_p))

            if model_type == "LSTM":
                all_out.append(str(sum(high_rho)/len(high_rho)))
                all_out.append(str(sum(low_rho)/len(low_rho)))
                all_out.append(str(sum(human_rho)/len(human_rho)))
        out_str += ','.join(all_out) + '\n'

    with open(outname, 'w') as f:
        f.write(out_str)

def save_who_results(outname, results, models, model_type ='LSTM'):

    out_str = []
    num_layers = len(results[models[0]])
    multi = len(results[models[0]][0][0])
    #Header
    if model_type == 'LSTM':
        for i in range(num_layers):
            for model in models:
                m = 'LSTM_'+model.split('-')[1].split('_')[-1]+'_layer_'+str(i)+'_who_HIGH_RSA'
                out_str.append(m)
                m = 'LSTM_'+model.split('-')[1].split('_')[-1]+'_layer_'+str(i)+'_who_HIGH_pvalue'
                out_str.append(m)

                m = 'LSTM_'+model.split('-')[1].split('_')[-1]+'_layer_'+str(i)+'_who_LOW_RSA'
                out_str.append(m)
                m = 'LSTM_'+model.split('-')[1].split('_')[-1]+'_layer_'+str(i)+'_who_LOW_pvalue'
                out_str.append(m)

                m = 'LSTM_'+model.split('-')[1].split('_')[-1]+'_layer_'+str(i)+'_who_HUMAN_RSA'
                out_str.append(m)
                m = 'LSTM_'+model.split('-')[1].split('_')[-1]+'_layer_'+str(i)+'_who_HUMAN_pvalue'
                out_str.append(m)

            out_str.append('LSTM_avg_layer_'+str(i)+'_who_HIGH_RSA')
            out_str.append('LSTM_avg_layer_'+str(i)+'_who_LOW_RSA')
            out_str.append('LSTM_avg_layer_'+str(i)+'_who_HUMAN_RSA')
    else:
        for i in range(len(results[models[0]])):
            for model in results:
                m = 'tf_layer_'+str(i)+'_who_HIGH_RSA' 
                out_str.append(m)
                m = 'tf_layer_'+str(i)+'_who_HIGH_pvalue' 
                out_str.append(m)

                m = 'tf_layer_'+str(i)+'_who_LOW_RSA' 
                out_str.append(m)
                m = 'tf_layer_'+str(i)+'_who_LOW_pvalue' 
                out_str.append(m)

                m = 'tf_layer_'+str(i)+'_who_HUMAN_RSA' 
                out_str.append(m)
                m = 'tf_layer_'+str(i)+'_who_HUMAN_pvalue' 
                out_str.append(m)

    out_str = ','.join(out_str)+'\n'

    data = []

    out = ''
    for x in range(len(results[models[0]][0])):
        all_out = []
        for i in range(num_layers):
            high_rho = []
            low_rho = []
            human_rho = []
            for model in models:
                measures = results[model][i][x][0]

                high_r = measures[0]
                high_rho.append(high_r)
                high_p = measures[1]
                all_out.append(str(high_r))
                all_out.append(str(high_p))

                low_r = measures[2]
                low_rho.append(low_r)
                low_p = measures[3]
                all_out.append(str(low_r))
                all_out.append(str(low_p))

                human_r = measures[4]
                human_rho.append(human_r)
                human_p = measures[5]
                all_out.append(str(human_r))
                all_out.append(str(human_p))

            if model_type == 'LSTM':
                all_out.append(str(sum(high_rho)/len(high_rho)))
                all_out.append(str(sum(low_rho)/len(low_rho)))
                all_out.append(str(sum(human_rho)/len(human_rho)))
        out_str += ','.join(all_out) + '\n'

    with open(outname, 'w') as f:
        f.write(out_str)

def save_were_results(outname, results, models, model_type ='LSTM'):

    out_str = []
    num_layers = len(results[models[0]])
    multi = len(results[models[0]][0][0])
    #Header
    if model_type == 'LSTM':
        for i in range(num_layers):
            for model in models:
                m = 'LSTM_'+model.split('-')[1].split('_')[-1]+'_layer_'+str(i)+'_were_HIGH_RSA'
                out_str.append(m)
                m = 'LSTM_'+model.split('-')[1].split('_')[-1]+'_layer_'+str(i)+'_were_HIGH_pvalue'
                out_str.append(m)

                m = 'LSTM_'+model.split('-')[1].split('_')[-1]+'_layer_'+str(i)+'_were_LOW_RSA'
                out_str.append(m)
                m = 'LSTM_'+model.split('-')[1].split('_')[-1]+'_layer_'+str(i)+'_were_LOW_pvalue'
                out_str.append(m)

                m = 'LSTM_'+model.split('-')[1].split('_')[-1]+'_layer_'+str(i)+'_were_HUMAN_RSA'
                out_str.append(m)
                m = 'LSTM_'+model.split('-')[1].split('_')[-1]+'_layer_'+str(i)+'_were_HUMAN_pvalue'
                out_str.append(m)

            out_str.append('LSTM_avg_layer_'+str(i)+'_were_HIGH_RSA')
            out_str.append('LSTM_avg_layer_'+str(i)+'_were_LOW_RSA')
            out_str.append('LSTM_avg_layer_'+str(i)+'_were_HUMAN_RSA')
    else:
        for i in range(len(results[models[0]])):
            for model in results:
                m = 'tf_layer_'+str(i)+'_were_HIGH_RSA' 
                out_str.append(m)
                m = 'tf_layer_'+str(i)+'_were_HIGH_pvalue' 
                out_str.append(m)

                m = 'tf_layer_'+str(i)+'_were_LOW_RSA' 
                out_str.append(m)
                m = 'tf_layer_'+str(i)+'_were_LOW_pvalue' 
                out_str.append(m)

                m = 'tf_layer_'+str(i)+'_were_HUMAN_RSA' 
                out_str.append(m)
                m = 'tf_layer_'+str(i)+'_were_HUMAN_pvalue' 
                out_str.append(m)

    out_str = ','.join(out_str)+'\n'

    data = []

    out = ''
    for x in range(len(results[models[0]][0])):
        all_out = []
        for i in range(num_layers):
            high_rho = []
            low_rho = []
            human_rho = []
            for model in models:
                measures = results[model][i][x][1]

                high_r = measures[0]
                high_rho.append(high_r)
                high_p = measures[1]
                all_out.append(str(high_r))
                all_out.append(str(high_p))

                low_r = measures[2]
                low_rho.append(low_r)
                low_p = measures[3]
                all_out.append(str(low_r))
                all_out.append(str(low_p))

                human_r = measures[4]
                human_rho.append(human_r)
                human_p = measures[5]
                all_out.append(str(human_r))
                all_out.append(str(human_p))

            if model_type == 'LSTM':
                all_out.append(str(sum(high_rho)/len(high_rho)))
                all_out.append(str(sum(low_rho)/len(low_rho)))
                all_out.append(str(sum(human_rho)/len(human_rho)))
        out_str += ','.join(all_out) + '\n'

    with open(outname, 'w') as f:
        f.write(out_str)
