import importlib
import math
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

import json
from attrdict import AttrDict

utils = importlib.import_module("utils")

#Loads environments settings
with open('./conf/env_setting.json') as env_setting:
    env = AttrDict(json.load(env_setting))


def get_model_results(df_boa, air_max, air_min, air_mean, last_checkpoint, upper_limit):
    df_allarmi = pd.DataFrame(columns=['ALARM_ID', 'NID_MACROAREA', 'NID_AREA', 'NID_PI', 'TMS_ALLARME', 'DESCRIZIONE_ALLARME','PATH_DATI_INPUT_COINVOLTI'])
    output_input_df_list = []
    series_slice_dimension = env.series_slice_dimension
    df_boa, df_boa_g = build_augmented_df_boa(df_boa, air_mean, air_min, air_max, series_slice_dimension)
    points = list(zip(df_boa_g[f'AIRGAP_INDEX_NORMALIZED_WEIGHTED_MEAN'].to_list(),df_boa_g[f'AIRGAP_INDEX_NORMALIZED_MEAN'].to_list()))
    current_eps = utils.max_min_distance(points)
    dbscan = DBSCAN(eps=current_eps, min_samples=5)
    df_boa_g[f'PREDICTED_CLUSTER_AIRGAP_INDEX'] = dbscan.fit_predict(np.array(points))
    # Estraggo quelli che sono stati etichettati con -1 (outlier) e che risiedono al di sotto del limite stabilito sia con la media che con la media pesata
    output_df = df_boa_g[(df_boa_g.PREDICTED_CLUSTER_AIRGAP_INDEX == -1)& \
                            (df_boa_g.AIRGAP_INDEX_NORMALIZED_WEIGHTED_MEAN < upper_limit)& \
                            (df_boa_g.AIRGAP_INDEX_NORMALIZED_MEAN < upper_limit)].copy(deep=True)
    # Le serie devono avere meno di <serie_slice_dimension> (15) giorni tra la prima lettura e l'ultima per poter essere considerate per l'output di un allarme
    output_df['TIMEDELTA'] = output_df.TIMESTAMP.apply(lambda l: (l[-1] - l[0]).days)
    output_df = output_df[output_df.TIMEDELTA <= series_slice_dimension]
    # Considero come data dell'errore il Timestamp dell'ultima lettura tra le 15 che lo hanno fatto alzare
    output_df['TIMESTAMP'] = output_df.TIMESTAMP.apply(lambda l: l[-1])
    output_df.reset_index(inplace=True)
    output_df['NID_MACROAREA'] = output_df.NID_MACROAREA.apply(lambda l: l[0])
    output_df['NID_AREA'] = output_df.NID_AREA.apply(lambda l: l[0])
    output_df['NID_PI'] = output_df.NID_PI.apply(lambda l: l[0])
    output_df['TMS_ALLARME'] = output_df.NID_MACROAREA.apply(lambda e: datetime.now())
    output_df['DESCRIZIONE_ALLARME'] = ''
    output_df['DESCRIZIONE_ALLARME'] = output_df.apply(lambda r: utils.single_or_parallel_pi(r.NID_MACROAREA, r.NID_AREA, r.NID_PI), axis=1)
    if len(output_df) > 0:
        output_df['N'] = output_df.apply(lambda r: len([id for (id, prov) in list(zip(r.ID, r.PROVENIENZA)) if id > last_checkpoint[prov]]),axis=1)
        output_df = output_df[output_df.N > 0]
        if len(output_df) > 0:
            if math.isnan(df_allarmi.ALARM_ID.max()):
                max_id = 0
            else:
                max_id = df_allarmi.ALARM_ID.max()
            output_df['ALARM_ID'] = [id for id in range(max_id + 1, max_id + 1 + len(output_df))]
            output_input_df_list = output_input_df_list + output_df.apply(lambda r: utils.extract_alarm_input(df_boa, r.ALARM_ID, r.GROUPBY_ID, 'GROUPBY_ID'), axis=1).to_list()
            output_df['PATH_DATI_INPUT_COINVOLTI'] = output_df.apply(lambda r: utils.get_alarm_pi_input_path(r.ALARM_ID, r.NID_MACROAREA, r.NID_AREA, r.NID_PI), axis=1)
            df_allarmi = df_allarmi.append(output_df[['ALARM_ID', 'NID_MACROAREA', 'NID_AREA', 'NID_PI', 'TMS_ALLARME', 'DESCRIZIONE_ALLARME', 'PATH_DATI_INPUT_COINVOLTI']])
    return df_allarmi, output_input_df_list


def build_augmented_df_boa(df_boa, air_mean, air_min, air_max, series_slice_dimension):
    bins_number = env.bins_number.pi
    df_valutazione_errori = pd.read_csv('./conf/Valutazione_errori_modelli.csv', sep=';', index_col=['Errore', 'Sotto_errore'])
    # Genero gli identificativi delle sequenze di 15 letture a partire dalla più recente
    df_boa['GROUPBY_ID'] = [e for l in [[i]*series_slice_dimension for i in range(math.ceil(len(df_boa)/series_slice_dimension))] for e in l][:len(df_boa)][::-1]
    # Media calcolata escludendo gli errori tra le 15 letture raggruppate
    df_mean = df_boa.loc[df_boa.ERRORE.isna(), ['AIRGAP_INDEX', 'GROUPBY_ID']].groupby(by=['GROUPBY_ID']).mean()
    # La media assegnata ai GROUPBY_ID che hanno problemi è pari alla media di airgap di tutte i punti informativi
    missing_ids = [e for e in df_boa.GROUPBY_ID.unique() if e not in df_mean.index]
    if len(missing_ids) > 0:
        df_mean = df_mean.append(pd.DataFrame([{'GROUPBY_ID': grp_id, 'AIRGAP_INDEX': air_mean} for grp_id in \
            [e for e in df_boa.GROUPBY_ID.unique() if e not in df_mean.index]]).set_index('GROUPBY_ID'))
    # Peso i valori di airgap_index in base alla presenza o meno di errori nella riga
    for errore, sotto_errore in df_valutazione_errori.index:
        if errore == 58:
            df_boa.loc[df_boa.ERRORE == errore, 'AIRGAP_INDEX'].fillna(0, inplace=True)
        elif (errore == 52) | (errore == 53):
            df_boa.loc[(df_boa.ERRORE == errore)&(df_boa.SOTTOERRORE == sotto_errore), 'AIRGAP_INDEX'] =\
                df_boa.loc[(df_boa.ERRORE == errore)&(df_boa.SOTTOERRORE == sotto_errore)]\
                .apply(lambda r: (df_mean.loc[r.GROUPBY_ID].AIRGAP_INDEX - \
                df_mean.loc[r.GROUPBY_ID].AIRGAP_INDEX *(df_valutazione_errori.loc[(r.ERRORE,r.SOTTOERRORE)].Peso_pi/10))*int(r.VELOCITA == 0), axis=1)
        else:
            df_boa.loc[(df_boa.ERRORE == errore)&(df_boa.SOTTOERRORE == sotto_errore), 'AIRGAP_INDEX'] =\
                df_boa.loc[(df_boa.ERRORE == errore)&(df_boa.SOTTOERRORE == sotto_errore)]\
                .apply(lambda r: df_mean.loc[r.GROUPBY_ID].AIRGAP_INDEX - \
                df_mean.loc[r.GROUPBY_ID].AIRGAP_INDEX *(df_valutazione_errori.loc[(r.ERRORE,r.SOTTOERRORE)].Peso_pi/10), axis=1)
    df_boa.AIRGAP_INDEX.fillna(df_boa.tail(15).AIRGAP_INDEX.mean(), inplace=True)
    # min e max non sono più calcolati tra quelli della boa corrente ma passati come parametri 
    bins = utils.define_bins_pi(air_min, air_max, bins_number)
    df_boa['LABEL_NORMALIZED_AIRGAP_INDEX'] = pd.cut(df_boa.AIRGAP_INDEX, bins=bins, labels=[str(e) for e in range(len(bins) - 1)])
    df_boa['NORMALIZED_AIRGAP_INDEX'] = utils.normalize_column(df_boa, 'AIRGAP_INDEX', air_max, air_min)
    df_boa_g = df_boa.groupby(by=['GROUPBY_ID']).agg(list)
    df_boa.drop(columns=['LABEL_NORMALIZED_AIRGAP_INDEX', 'NORMALIZED_AIRGAP_INDEX'], inplace=True)
    df_boa_g['CLASS_FREQ_NORMALIZED_AIRGAP_INDEX'] = df_boa_g.LABEL_NORMALIZED_AIRGAP_INDEX.apply(lambda l: utils.bins_frequency(l, bins_number))
    df_boa_g['AIRGAP_INDEX_NORMALIZED_WEIGHTED_MEAN'] = df_boa_g.apply(lambda r: utils.calculate_weighted_feature(r, 'NORMALIZED_AIRGAP_INDEX'), axis=1)
    df_boa_g['AIRGAP_INDEX_NORMALIZED_WEIGHTED_MEAN'].fillna(0, inplace=True)
    df_boa_g['AIRGAP_INDEX_NORMALIZED_MEAN'] = df_boa_g['NORMALIZED_AIRGAP_INDEX'].apply(lambda l: sum(l)/len(l))
    df_boa_g['AIRGAP_INDEX_NORMALIZED_MEAN'].fillna(0, inplace=True)
    return df_boa, df_boa_g