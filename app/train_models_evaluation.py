import importlib
import itertools
import math
from datetime import datetime
import json
from attrdict import AttrDict
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

utils = importlib.import_module("utils")

#Loads environments settings
with open('./conf/env_setting.json') as env_setting:
    env = AttrDict(json.load(env_setting))

def get_model_results(df_treno, zone_param_dict, last_checkpoint, upper_limit):
    df_allarmi = pd.DataFrame(columns=['ALARM_ID', 'MATRICOLA_TRENO', 'CABINA', 'FREQUENZA', 'TMS_ALLARME', 'DESCRIZIONE_ALLARME','PATH_DATI_INPUT_COINVOLTI'])
    output_input_df_list = []
    # Divido i dati in base ai canali (cab - freq)
    df_canali = [df for df in [df_treno[(df_treno.CAB == cabina)&(df_treno.FREQUENZA == frequenza)].copy(deep=True) \
                    for cabina, frequenza in list(itertools.product(*[df_treno.CAB.unique(), df_treno.FREQUENZA.unique()]))] if not df.empty]
    for df_canale in df_canali:
        cab = df_canale.CAB.unique()[0]
        freq = df_canale.FREQUENZA.unique()[0]
        augmented_df_canale = build_augmented_df_canale(df_canale, cab, freq, zone_param_dict)
        points = list(zip(augmented_df_canale['AIRGAP_INDEX_NORMALIZED_WEIGHTED_MEAN'].to_list(),augmented_df_canale['AIRGAP_INDEX_NORMALIZED_MEAN'].to_list()))
        eps = utils.max_min_distance(points)
        dbscan = DBSCAN(eps=eps, min_samples=5)
        augmented_df_canale['PREDICTED_CLUSTER'] = dbscan.fit_predict(np.array(points))
        # Estraggo quelli che sono stati etichettati con -1 (outlier) e che risiedono al di sotto del limite stabilito sia con la media che con la media pesata
        output_df = augmented_df_canale[(augmented_df_canale.PREDICTED_CLUSTER == -1)& \
                    (augmented_df_canale.AIRGAP_INDEX_NORMALIZED_WEIGHTED_MEAN < upper_limit)& \
                    (augmented_df_canale.AIRGAP_INDEX_NORMALIZED_MEAN < upper_limit)].copy(deep=True)
        output_df.reset_index(inplace=True)
        output_df['TIMESTAMP'] = output_df.TIMESTAMP.apply(lambda l: l[0])
        output_df['MATRICOLA_TRENO'] = output_df.MATRICOLA_TRENO.apply(lambda l: l[0])
        output_df['CABINA'] = output_df.CAB.apply(lambda l: l[0])
        output_df['FREQUENZA'] = output_df.FREQUENZA.apply(lambda l: l[0])
        output_df['TMS_ALLARME'] = output_df.CAB.apply(lambda e: datetime.now())
        output_df['DESCRIZIONE_ALLARME'] = output_df.CAB.apply(lambda e: f'{env.alarms_types.train_alarms_desc}')
        if len(output_df) > 0:
            output_df['N'] = output_df.apply(lambda r: len([i for (i, p) in list(zip(r.ID, r.PROVENIENZA)) if i > last_checkpoint[p]]), axis=1)
            output_df = output_df[output_df.N > 0]
            if len(output_df) > 0:
                if math.isnan(df_allarmi.ALARM_ID.max()):
                    max_id = 0
                else:
                    max_id = df_allarmi.ALARM_ID.max()
                output_df['ALARM_ID'] = [id for id in range(max_id + 1, max_id + 1 + len(output_df))]
                output_input_df_list = output_input_df_list + output_df.apply(lambda r: utils.extract_alarm_input(df_treno, r.ALARM_ID, r.ID_RUN_BUILD, 'ID_RUN_BUILD'), axis=1).to_list()
                output_df['PATH_DATI_INPUT_COINVOLTI'] = output_df.apply(lambda r: utils.get_alarm_train_input_path(r.ALARM_ID, r.MATRICOLA_TRENO), axis=1)
                df_allarmi = df_allarmi.append(output_df[['ALARM_ID', 'MATRICOLA_TRENO', 'CABINA', 'FREQUENZA', 'TMS_ALLARME', 'DESCRIZIONE_ALLARME','PATH_DATI_INPUT_COINVOLTI']])
    return (df_allarmi, output_input_df_list)


def build_augmented_df_canale(df_canale, cab, freq, zone_param_dict):
    bins_number = env.bins_number.train
    df_valutazione_errori = pd.read_csv('./conf/Valutazione_errori_modelli.csv', sep=';', index_col=['Errore', 'Sotto_errore'])
    # Media calcolata escludendo gli errori tra le letture raggruppate
    df_mean = df_canale.loc[df_canale.ERRORE.isna(), ['AIRGAP_INDEX', 'ID_RUN_BUILD']].groupby(by=['ID_RUN_BUILD']).mean()
    # La media assegnata agli ID_RUN_BUILD che hanno problemi è quella del canale specifico calcolata su tutti i treni della zona di competenza
    air_mean = zone_param_dict[(cab, freq)]["air_mean"]        
    missing_ids = [e for e in df_canale.ID_RUN_BUILD.unique() if e not in df_mean.index]
    if len(missing_ids) > 0:
        df_mean = df_mean.append(pd.DataFrame([{'ID_RUN_BUILD': grp_id, 'AIRGAP_INDEX': air_mean} for grp_id in \
            [e for e in df_canale.ID_RUN_BUILD.unique() if e not in df_mean.index]]).set_index('ID_RUN_BUILD'))
    # Peso i valori di airgap_index in base alla presenza o meno di errori nella riga
    for errore, sotto_errore in df_valutazione_errori.index:
        if errore == 58:
            df_canale.loc[df_canale.ERRORE == errore, 'AIRGAP_INDEX'].fillna(0, inplace=True)
        elif (errore == 52) | (errore == 53):
            df_canale.loc[(df_canale.ERRORE == errore)&(df_canale.SOTTOERRORE == sotto_errore), 'AIRGAP_INDEX'] =\
                df_canale.loc[(df_canale.ERRORE == errore)&(df_canale.SOTTOERRORE == sotto_errore)]\
                .apply(lambda r: (df_mean.loc[r.ID_RUN_BUILD].AIRGAP_INDEX - \
                df_mean.loc[r.ID_RUN_BUILD].AIRGAP_INDEX *(df_valutazione_errori.loc[(r.ERRORE,r.SOTTOERRORE)].Peso_canale/10))*int(r.VELOCITA == 0), axis=1)
        else:
            df_canale.loc[(df_canale.ERRORE == errore)&(df_canale.SOTTOERRORE == sotto_errore), 'AIRGAP_INDEX'] =\
                df_canale.loc[(df_canale.ERRORE == errore)&(df_canale.SOTTOERRORE == sotto_errore)]\
                .apply(lambda r: df_mean.loc[r.ID_RUN_BUILD].AIRGAP_INDEX - \
                df_mean.loc[r.ID_RUN_BUILD].AIRGAP_INDEX *(df_valutazione_errori.loc[(r.ERRORE,r.SOTTOERRORE)].Peso_canale/10), axis=1)
    # Non dovrebbero più esserci valori null di airgap ma, per sicurezza, nel caso ci fossero vengono settati a 0.
    df_canale.AIRGAP_INDEX.fillna(0, inplace=True)
    air_min = zone_param_dict[(cab, freq)]["air_min"]
    air_max = zone_param_dict[(cab, freq)]["air_max"]
    # Le label sono calcolate con il minimo e il massimo storico del canale specifico della zona di competenza del treno in esame
    # allo stesso modo sono normalizzati i dati tra il minimo e il massimo storico dello specifico canale della zona di competenza del treno in esame
    bins = utils.define_bins_train(air_min, air_max, bins_number)
    df_canale[f'LABEL_NORMALIZED_AIRGAP_INDEX'] = pd.cut(df_canale.AIRGAP_INDEX, bins=bins, labels=[str(e) for e in range(len(bins) - 1)])
    df_canale['NORMALIZED_AIRGAP_INDEX'] = utils.normalize_column(df_canale, 'AIRGAP_INDEX', air_max, air_min)
    df_canale = df_canale.groupby(by=['ID_RUN_BUILD']).agg(list)
    df_canale[f'CLASS_FREQ_NORMALIZED_AIRGAP_INDEX'] = df_canale.LABEL_NORMALIZED_AIRGAP_INDEX.apply(lambda l: utils.bins_frequency(l, bins_number))
    df_canale['AIRGAP_INDEX_NORMALIZED_WEIGHTED_MEAN'] = df_canale.apply(lambda r: utils.calculate_weighted_feature(r, 'NORMALIZED_AIRGAP_INDEX'), axis=1)
    df_canale['AIRGAP_INDEX_NORMALIZED_MEAN'] = df_canale.NORMALIZED_AIRGAP_INDEX.apply(lambda l: sum(l)/len(l))
    return df_canale