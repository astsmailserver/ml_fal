################################################################################################################################
# IMPORT
################################################################################################################################
import collections
import math
import pandas as pd
import json
from attrdict import AttrDict
import os

#Loads environments settings
with open('./conf/env_setting.json') as env_setting:
    env = AttrDict(json.load(env_setting))

################################################################################################################################
# UTILITIES
################################################################################################################################


def log(path, log_line):
    if(os.path.exists(path)):
        with open(path, 'a') as f:
            f.write(f'{log_line}\n')
    else:
        with open(path, 'w') as f:
            f.write(f'{log_line}\n')



def print_subprocess_out(subprocess_out, ext_type):
    for l in [f'{ext_type}: {e}' for e in subprocess_out.split('\n')]:
        print(l)


def define_bins_pi(m, M, n):
    if M == m:
        return [m + (i*int((m + 1000)/n)) if i != 0 else -float('inf') for i in range(n)] + [float('inf')]
    else:
        return [m + (i*int(M/n)) if i != 0 else -float('inf') for i in range(n)] + [float('inf')]


def define_bins_train(m, M, n):
    return [m + (i*int(M/n)) if i != 0 else -float('inf') for i in range(n)] + [float('inf')]


def normalize_column(dataframe, column_name):
    return (dataframe[column_name] - dataframe[column_name].min()) / (dataframe[column_name].max() - dataframe[column_name].min())


def normalize_column(dataframe, column_name, c_max, c_min):
    return (dataframe[column_name] - c_min) / (c_max - c_min)


def bins_frequency(l,b_n):
    count = collections.Counter(l)
    return [count[str(e)] if str(e) in count.keys() else 0 for e in range(b_n)]


def calculate_weighted_feature(r, feature):
    return sum([(v*r[f'CLASS_FREQ_{feature}'][int(r[f'LABEL_{feature}'][i])]) for i, v in enumerate(r[feature])])/sum([pow(e,2) for e in r[f'CLASS_FREQ_{feature}']])


def max_min_distance(l):
    if (len(l) == 1)|(len(list(set(l))) == 1):
        return 0.00000000001
    else:
        ll = [min([math.sqrt(sum([pow(p_i[i] - p[i], 2) for i in range(len(p_i))])) for p_i in [e for e in l if e != p]]) for p in l] # MEDIA TRA MASSIMO E MINIMO DELLE DISTANZE
        return (max(ll) + min(ll))/2


def extract_alarm_input(df, alarm_id, id, id_column):
    out_df = df[df[id_column] == id]
    out_df.drop(columns=[id_column, 'POSSIBILI_SUCCESSIVI', 'PROVENIENZA'], inplace=True)
    return (alarm_id, out_df)


def get_alarm_train_input_path(alarm_id, train_id):
    return f'{env.path.details_directory}'


def get_alarm_pi_input_path(alarm_id, id_macroarea, id_area, id_pi):
    return f'{env.path.details_directory}'


def weight_error(mean, weight):
    return mean - (mean*weight)/10


def single_or_parallel_pi(macroarea, area, pi):
    #Loads environments settings
    with open('./conf/env_setting.json') as env_setting:
        env = AttrDict(json.load(env_setting))
    df_catene = pd.read_csv('./conf/Catene.csv', sep=';')
    df_catene.fillna('', inplace=True)
    id_text = f'{macroarea}-{area}-{pi}'
    df_catene['POSSIBILI_SUCCESSIVI'] = df_catene.POSSIBILI_SUCCESSIVI.apply(lambda s: s.split(','))
    pos_list = df_catene[df_catene.POSSIBILI_SUCCESSIVI.apply(lambda l: id_text in l)].POSSIBILI_SUCCESSIVI.to_list()
    if(len(pos_list) > 0):
        if(len(pos_list[0]) > 1):
            return env.alarms_types.multiple_pi_alarms_desc
        else:    
            return env.alarms_types.single_pi_alarms_desc
    else:
        return env.alarms_types.single_pi_alarms_desc