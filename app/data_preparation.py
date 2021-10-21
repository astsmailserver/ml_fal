import importlib
import itertools
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import parser as dateParser
from dateutil.relativedelta import relativedelta
from sqlalchemy import create_engine
from attrdict import AttrDict
import json

utils = importlib.import_module('utils')

module_log_file = ''

# Loads environments settings
with open('./conf/env_setting.json') as env_setting:
    env = AttrDict(json.load(env_setting))

# Create the engine for db comunication
engine = create_engine(f'postgresql://{env.db.username}:{env.db.password}@{env.db.server_name}:{env.db.port}/{env.db.name}')

def get_clean_data(last_checkpoint, upper_limit, log_file):
    global module_log_file
    module_log_file = log_file
    utils.log(module_log_file, f'Inizio ottenimento e preparazione dei dati...')
    df_treni, df_boe = load_data_from_database(last_checkpoint)
    if (df_treni.empty)&(df_boe.empty):
        return [(df_treni,0,0,0)], [(df_boe,0,0,0)]
    else:
        return generate_train_df_list(df_treni, last_checkpoint, upper_limit), generate_pi_df_list(df_boe, last_checkpoint, upper_limit)


def load_data_from_database(last_checkpoint):
    new_checkpoint = AttrDict(last_checkpoint.copy())
    with engine.connect() as dbConnection:
        df_captazioni = pd.read_sql(f"select * from {env.db.tabella_captazioni} where date >= '{str((datetime.now() - relativedelta(years=env.history_lenght)).date())}'", dbConnection)
        new_checkpoint.captazioni = int(df_captazioni.id.max())
        rename_dict = {c:c.upper() for c in df_captazioni.columns}
        df_captazioni.rename(columns=rename_dict, inplace=True)
    utils.log(module_log_file, f'Righe lette da captazioni: {len(df_captazioni)}')
    if len(df_captazioni) != 0:
        df_captazioni = clean_captazioni(df_captazioni)
        with engine.connect() as dbConnection:
            df_errori = pd.read_sql(f"select * from {env.db.tabella_errori}  where date >= '{str((datetime.now() - relativedelta(years=env.history_lenght)).date())}'", dbConnection)
            new_checkpoint.errori = int(df_errori.id.max())
            rename_dict = {c:c.upper() for c in df_errori.columns}
            df_errori.rename(columns=rename_dict, inplace=True)
        utils.log(module_log_file, f'Righe lette da errori: {len(df_errori)}')
        df_errori = clean_errori(df_errori, df_captazioni)
        train_list = list(df_captazioni[df_captazioni.ID > last_checkpoint.captazioni].MATRICOLA_TRENO.unique())
        with open('./conf/checkpoints.json', 'w') as fp:
            json.dump(new_checkpoint, fp)
        if(last_checkpoint.captazioni == new_checkpoint.captazioni)&\
            (last_checkpoint.errori == new_checkpoint.errori):
            return pd.DataFrame(), pd.DataFrame()
        return clean_data_for_train_analisys(df_captazioni, df_errori, train_list), clean_data_for_pi_analisys(df_captazioni, df_errori)
    else :
        return df_captazioni, df_captazioni


def clean_captazioni(df):
    df.loc[df.ERRORE == '\\N','ERRORE'] = np.nan
    df['ERRORE'] = df.ERRORE.astype('float')
    df['TIMESTAMP'] = df.apply(lambda r: dateParser.parse(f'{r.DATE} {r.ORARIO_MMI}'), axis=1)
    df['PROVENIENZA'] = 'captazioni'
    df['NID_MACROAREA'] = df.NID_MACROAREA.astype('int')
    df['NID_AREA'] = df.NID_AREA.astype('int')
    df['NID_PI'] = df.NID_PI.astype('int')
    return df


def clean_errori(df, df_c):
    if(len(df) > 0):
        init_lenght = len(df)
        df['ERRORE'] = df.ERRORE.astype('float')
        df.drop(df[(df.ERRORE < 50)|(df.ERRORE > 62)].index, inplace=True)
        utils.log(module_log_file, f'Righe rimosse da errori per codice errore non considerato: {init_lenght - len(df)}')
        df.rename(columns={'ID_TRAIN': 'MATRICOLA_TRENO'}, inplace=True)
        df.MATRICOLA_TRENO.fillna(df.LOG_CANMVB_PATH.apply(lambda e: e.split('\\')[0].replace('_', ' ')), inplace=True)
        df['TIMESTAMP'] = df.apply(lambda r: dateParser.parse(f'{r.DATE} {r.ORARIO_MMI}'), axis=1)
        df_out = pd.merge(df, df_c[['MATRICOLA_TRENO', 'CAB', 'ID_RUN', 'TIMESTAMP', 'AIRGAP_INDEX']],\
                            on=['MATRICOLA_TRENO', 'CAB', 'ID_RUN', 'TIMESTAMP'], how='left')
        df_out = df_out[df_out.AIRGAP_INDEX.isna()]
        utils.log(module_log_file, f'Righe lette da errori e non in comune con captazioni: {len(df) - len(df_out)}')
        df_out['NID_MACROAREA'] = df_out.NID_MACROAREA.astype('int')
        df_out['NID_AREA'] = df_out.NID_AREA.astype('int')
        df_out['NID_PI'] = df_out.NID_PI.astype('int')
        df_out['ERRORE'] = df_out.ERRORE.astype('float')
        df_out['PROVENIENZA'] = 'errori'
        return df_out
    else:
        return pd.merge(df, df_c[['MATRICOLA_TRENO', 'CAB', 'ID_RUN', 'TIMESTAMP', 'AIRGAP_INDEX']],\
                            on=['MATRICOLA_TRENO', 'CAB', 'ID_RUN', 'TIMESTAMP'], how='left')


def clean_data_for_train_analisys(df_captazioni, df_errori, train_list):
    df_captazioni = df_captazioni[df_captazioni.MATRICOLA_TRENO.isin(train_list)]
    df_errori = df_errori[df_errori.MATRICOLA_TRENO.isin(train_list)]
    # Rimuovo gli errori 59 che creerebbero problemi nella costruzione delle corse
    init_lenght = len(df_captazioni)
    df_captazioni.drop(df_captazioni[(df_captazioni.ERRORE == '59')].index, inplace=True)
    utils.log(module_log_file, f'Righe rimosse da captazioni con codice errore 59 per analisi treno: {init_lenght - len(df_captazioni)}')
    init_lenght = len(df_errori)
    df_errori.drop(df_errori[(df_errori.ERRORE == '59')].index, inplace=True)
    utils.log(module_log_file, f'Righe rimosse da captazioni con codice errore 59 per analisi treno: {init_lenght - len(df_errori)}')
    # Rimuovo gli ID_RUN che hanno il problema con le date
    id_run_to_remove = id_run_to_remove_date(df_captazioni)
    init_lenght = len(df_captazioni)
    df_captazioni = df_captazioni[~df_captazioni.ID_RUN.isin(id_run_to_remove)]
    utils.log(module_log_file, f'Righe rimosse da captazioni per ID_RUN con problema date: {init_lenght - len(df_captazioni)}')
    init_lenght = len(df_errori)
    df_errori = df_errori[~df_errori.ID_RUN.isin(id_run_to_remove)]
    utils.log(module_log_file, f'Righe rimosse da errori per ID_RUN con problema date: {init_lenght - len(df_errori)}')
    # Unisco errori e captazioni e ordino secondo il TIMESTAMP per poter costruire poi correttamente le corse
    df = pd.concat([df_captazioni, df_errori])
    df.sort_values(by=['MATRICOLA_TRENO', 'TIMESTAMP'], inplace=True)
    # Correggo le informazioni di quale punto è stato mancato in presenza di errore 51
    df = set_correct_pi_info_error_51(df, 'treno')
    # Rimuovo i 51 che seguono i 58
    init_lenght = len(df)
    df['PREV_ERR'] = df.ERRORE.shift(1)
    df.loc[len(df) - 1, 'PREV_ERR'] = np.nan
    df['CONS_ERR'] = df.apply(lambda r: int((r.ERRORE == 51)&(r.PREV_ERR == 58)), axis=1)
    df = df[df.CONS_ERR == 0]
    df.drop(columns=['PREV_ERR', 'CONS_ERR'], inplace=True)
    utils.log(module_log_file, f'Righe con errori 51 che seguono un 58 rimosse: {init_lenght - len(df)}')
    # Calcolo l'ID_RUN_BUILD che mi rappresenta le corse del treno
    df = evaluate_id_run_build(df)
    # Rimuovo le corse con meno di 3 letture
    df_g = df[['MATRICOLA_TRENO', 'ID_RUN_BUILD']].groupby(by=['ID_RUN_BUILD']).count().sort_values(by=['MATRICOLA_TRENO'], ascending=False)
    init_lenght = len(df)
    df.drop(df[(df.ID_RUN_BUILD.isin(df_g.loc[df_g.MATRICOLA_TRENO < 3].index.to_list()))].index, inplace=True)
    utils.log(module_log_file, f'Righe rimosse con meno di 3 letture per analisi treno: {init_lenght - len(df)}')
    return df


def clean_data_for_pi_analisys(df_captazioni, df_errori):
    df = pd.concat([df_captazioni, df_errori])
    df = set_correct_pi_info_error_51(df, 'punto_informativo')
    utils.log(module_log_file, f'Numero di linee con NID_PI pari a 0 rimosse: {len(df[df.NID_PI == 0])}')
    df.drop(df[df.NID_PI == 0].index, inplace=True)
    return df


def id_run_to_remove_date(df_captazioni):
    df_captazioni_copy = df_captazioni.copy(deep=True)
    df_captazioni_copy.sort_values(by=['ID_RUN', 'ID'], inplace=True)
    df_captazioni_copy['CAMBIO_GIORNO'] = df_captazioni_copy.TIMESTAMP.gt(df_captazioni_copy.TIMESTAMP.shift(-1)).astype('int')
    df_captazioni_copy['CAMBIO_ID_RUN'] = df_captazioni_copy.ID_RUN.eq(df_captazioni_copy.ID_RUN.shift(-1)).astype('int')
    id_run_to_remove = df_captazioni_copy[(df_captazioni_copy.CAMBIO_GIORNO == 1)&(df_captazioni_copy.CAMBIO_ID_RUN == 1)].ID_RUN.to_list()
    utils.log(module_log_file, f'Numero di ID_RUN con problema sulle date: {len(id_run_to_remove)}')
    df_captazioni_copy.drop(columns=['CAMBIO_GIORNO', 'CAMBIO_ID_RUN'], inplace=True)
    return id_run_to_remove


def evaluate_id_run_build(df):
    df['PREV_POS_SUC'] = df.POSSIBILI_SUCCESSIVI.shift(1)
    df.loc[df.PREV_POS_SUC.isna(), 'PREV_POS_SUC'] = 'FIRST'
    df['PREV_CAB'] = df.CAB.shift(1)
    df.loc[df.PREV_CAB.isna(), 'PREV_CAB'] = 'FIRST'
    df['NOT_PI_IN_PREV'] = df.apply(lambda r: int(not((f"{r.NID_MACROAREA}-{r.NID_AREA}-{r.NID_PI}" in r.PREV_POS_SUC)|(r.PREV_POS_SUC == 'FIRST'))), axis=1)
    df['NOT_SAME_CAB'] = df.apply(lambda r: int(not((r.CAB == r.PREV_CAB)|(r.PREV_CAB == 'FIRST'))), axis=1)
    df['ID_RUN_BUILD'] = df.apply(lambda r: r.NOT_PI_IN_PREV | r.NOT_SAME_CAB, axis=1)
    df['ID_RUN_BUILD'] = df['ID_RUN_BUILD'].cumsum()
    df.drop(columns=['PREV_POS_SUC', 'PREV_CAB', 'NOT_PI_IN_PREV', 'NOT_SAME_CAB'], inplace=True)
    return df


def set_correct_pi_info_error_51(df_captazioni, analisys_type):
    df_catene = pd.read_csv('./conf/Catene.csv', sep=';')
    df_catene.fillna('', inplace=True)
    df_catene['NID_MACROAREA'] = df_catene.PUNTO_INFORMATIVO.apply(lambda s: int(s.split('-')[0]))
    df_catene['NID_AREA'] = df_catene.PUNTO_INFORMATIVO.apply(lambda s: int(s.split('-')[1]))
    df_catene['NID_PI'] = df_catene.PUNTO_INFORMATIVO.apply(lambda s: int(s.split('-')[2]))
    df_catene.set_index('PUNTO_INFORMATIVO', inplace=True)
    df_catene['POSSIBILI_SUCCESSIVI'] = df_catene.POSSIBILI_SUCCESSIVI.apply(lambda s: sorted(list([e if e != 'FINE_SERIE' else '0-0-0' for e in s.split(',')])))
    df_captazioni_next = pd.merge(df_captazioni, df_catene, on=['NID_MACROAREA','NID_AREA','NID_PI'], how='left')
    df_captazioni_next.loc[(df_captazioni_next.ERRORE == 51), 'NID_MACROAREA'] = df_captazioni_next[df_captazioni_next.ERRORE == 51]['POSSIBILI_SUCCESSIVI'].apply(lambda l: int(l[0].split('-')[0]))
    df_captazioni_next.loc[(df_captazioni_next.ERRORE == 51), 'NID_AREA'] = df_captazioni_next[df_captazioni_next.ERRORE == 51]['POSSIBILI_SUCCESSIVI'].apply(lambda l: int(l[0].split('-')[1]))
    df_captazioni_next.loc[(df_captazioni_next.ERRORE == 51), 'NID_PI'] = df_captazioni_next[df_captazioni_next.ERRORE == 51]['POSSIBILI_SUCCESSIVI'].apply(lambda l: int(l[0].split('-')[2]))
    df_captazioni_next.loc[(df_captazioni_next.ERRORE == 51), 'AIRGAP_INDEX'] = 0
    if analisys_type == 'punto_informativo':
        list_dict = df_captazioni_next.loc[df_captazioni_next.ERRORE == 51].to_dict(orient='records')
        to_add = []
        for d in list_dict:
            for b in d['POSSIBILI_SUCCESSIVI'][1:]:
                new_d = d.copy()
                boa = b.split('-')
                new_d['NID_MACROAREA'] = int(boa[0])
                new_d['NID_AREA'] = int(boa[1])
                new_d['NID_PI'] = int(boa[2])
                to_add.append(new_d)
        df_captazioni_next = df_captazioni_next.append(to_add)
        df_captazioni_next.sort_values(by=['NID_MACROAREA', 'NID_AREA', 'NID_PI', 'TIMESTAMP'], inplace=True)
    df_captazioni_next.drop(columns=['POSSIBILI_SUCCESSIVI'], inplace=True)
    df_captazioni_next = pd.merge(df_captazioni_next, df_catene, on=['NID_MACROAREA','NID_AREA','NID_PI'], how='left')
    df_punti_informativi = pd.read_csv('./conf/Punti_Informativi.csv', sep=';')
    df_captazioni_next = pd.merge(df_captazioni_next, df_punti_informativi, on=['NID_MACROAREA','NID_AREA','NID_PI'], how='left')
    df_captazioni_next.loc[df_captazioni_next.FREQUENZA.isna(), 'FREQUENZA'] = df_captazioni_next.loc[df_captazioni_next.FREQUENZA.isna(), 'FREQUENZA_PI']
    df_captazioni_next.loc[df_captazioni_next.DIREZIONE.isna(), 'DIREZIONE'] = df_captazioni_next.loc[df_captazioni_next.DIREZIONE.isna(), 'DIREZIONE_PI']
    df_captazioni_next.drop(columns=['FREQUENZA_PI', 'DIREZIONE_PI'], inplace=True)
    return df_captazioni_next


def generate_train_df_list(df_treni, last_checkpoint, upper_limit):
    train_order_list = list(df_treni.MATRICOLA_TRENO.unique())
    df_train_list = [df_treni[df_treni.MATRICOLA_TRENO == treno].copy(deep=True) for treno in train_order_list]
    zone_value_param_list = get_train_zone_value_param_list(df_treni, train_order_list)
    return [(df_train_list[i], zone_value_param_list[i], last_checkpoint, upper_limit) for i, treno in enumerate(train_order_list)]


def get_train_zone_value_param_list(df_treni, train_list):
    df_zone = pd.read_csv('./conf/Zone_competenza.csv', sep=';')
    # lista di tuple (air_min, air_mean, air_max, cabina, frequenza, zona)
    zone_value_list = [t for l in [[(df_copy.loc[(df_copy.CAB == cabina)&(df_copy.FREQUENZA == frequenza), "AIRGAP_INDEX"].min(),\
                            df_copy.loc[(df_copy.CAB == cabina)&(df_copy.FREQUENZA == frequenza), "AIRGAP_INDEX"].mean(),\
                            df_copy.loc[(df_copy.CAB == cabina)&(df_copy.FREQUENZA == frequenza), "AIRGAP_INDEX"].max(), cabina, frequenza, zona) \
                            for cabina,frequenza in list(itertools.product(*[df_treni.CAB.unique(),df_treni.FREQUENZA.unique()]))]\
                            for (df_copy, zona) in [(df_treni.loc[df_treni.MATRICOLA_TRENO.isin(train_list)&(df_treni.AIRGAP_INDEX != 65535)].copy(deep=True), zona)  \
                            for (zona, train_list) in [(zona,  df_zone.loc[df_zone.ZONA_COMPETENZA == zona, "MATRICOLA_TRENO"].to_list()) \
                            for zona in df_zone.ZONA_COMPETENZA.unique()]]] for t in l]
    # dizionario con chiave zona e valore dizionario con chiavi (cabina, frequenza) e valori un dizionario di con min, media e massimo relativi
    zone_dict = {l[0][0]:{(cabina, frequenza):d for (zona, cabina, frequenza, d) in l}\
                    for l in [[(zona, cabina, frequenza, {"air_min":air_min, "air_mean":air_mean, "air_max":air_max}) \
                    for (air_min, air_mean, air_max, cabina, frequenza, zona) in zone_value_list if zona == zona_to_ext] \
                    for zona_to_ext in df_zone.ZONA_COMPETENZA.unique()]}
    df_zone.set_index("MATRICOLA_TRENO", inplace=True)
    return [zone_dict[df_zone.loc[t].ZONA_COMPETENZA] for t in train_list]


def generate_pi_df_list(df_boe, last_checkpoint, upper_limit):
    df_boe_list = [((macroarea, area, pi), df_boe[(df_boe.NID_MACROAREA == macroarea)&(df_boe.NID_AREA == area)&(df_boe.NID_PI == pi)].copy(deep=True))\
                        for macroarea, area, pi in df_boe[['NID_MACROAREA','NID_AREA', 'NID_PI']] \
                            .groupby(by=['NID_MACROAREA','NID_AREA', 'NID_PI']).count().index.to_list()]
    max_airgap = df_boe.loc[df_boe.AIRGAP_INDEX != 65535, "AIRGAP_INDEX"].max()
    min_airgap = df_boe.AIRGAP_INDEX.min()
    mean_airgap = df_boe.loc[df_boe.AIRGAP_INDEX != 65535, "AIRGAP_INDEX"].mean()
    return [(df, max_airgap, min_airgap, mean_airgap, last_checkpoint, upper_limit) for ((macroarea, area, pi), df) in df_boe_list if not df.empty]