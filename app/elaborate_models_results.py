import pandas as pd
from attrdict import AttrDict
import json
from datetime import datetime
from sqlalchemy import create_engine
from pathlib import Path

# Colonne finali della tabella issues
# ID_ISSUE, fkID_ALARM, ID_REPOSITORY, ID_DATALOGGER, SIGLA_COMPLESSO, MATRICOLA_TRENO, VER_SW_OBU_ERTMS, 
# VER_SW_OBU_SCMT, OPENING_DATETIME, OPENING_USER, CLOSURE_DATETIME, CLOSURE_USER, LAST_UPDATE_DATETIME, 
# ASSIGNED_TO_USER, STATE, RESOLUTION, SEVERITY, DETAILS, ACTIONS_DONE, ACTIONS_TO_DO, NOTE, AFFECTED_RUNS, 
# FIRST_AFFECTED_RUN, LAST_AFFECTED_RUN, DISTANCE_AFTER_FIRST, POWER_ONS_AFTER_FIRST, MISSIONS_AFTER_FIRST, 
# DISTANCE_AFTER_LAST, POWER_ONS_AFTER_LAST, MISSIONS_AFTER_LAST, LAST_RUN, ALARM_MODE, ALARM_EXPRESSION, 
# HISTORY_LAST_UPDATE_BY_USER, UNDER_OBSERVATION_DATETIME, DISTANCE_AFTER_UNDER_OBSERVATION, 
# POWER_ONS_AFTER_UNDER_OBSERVATION, MISSIONS_AFTER_UNDER_OBSERVATION, IS_OBSERVATION_FINISHED

# COLONNE NON DEFAULT
# fkID_ALARM, ID_REPOSITORY, ID_DATALOGGER, SIGLA_COMPLESSO, MATRICOLA_TRENO, VER_SW_OBU_ERTMS, 
# VER_SW_OBU_SCMT, DETAILS

# DETAILS Può contenere tutte le info aggiuntive che ci pare

#Colonne finali della tabella table_issues_linking_runs_summary
# ID, CATEGORY, TS_AT, DISTANCE_AT, SPEED_AT, fkID_ISSUE, fkID_RUN

#Loads environments settings
with open('./conf/env_setting.json') as env_setting:
    env = AttrDict(json.load(env_setting))

engine = create_engine(f'postgresql://{env.db.username}:{env.db.password}@{env.db.server_name}:{env.db.port}/{env.db.name}')

def elaborate_train_results(df_allarmi, alarms_input_list):
    df, df_details = aggregate_results(df_allarmi, alarms_input_list, 'train')
    if len(df) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df[['ID_REPOSITORY', 'ID_DATALOGGER', 'VER_SW_OBU_SCMT', 'VER_SW_OBU_ERTMS', 'SIGLA_COMPLESSO']] = \
        df.apply(lambda r: get_run_config_values(df_details.loc[(df_details.ALARM_ID == r.ALARM_ID)
            &(df_details.MATRICOLA_TRENO == r.MATRICOLA_TRENO)].ID_RUN.to_list()[0]), axis=1, result_type='expand')
    df.rename(columns={'ALARM_ID':'fkID_ALARM'},inplace=True)
    df_details.rename(columns={'ALARM_ID':'fkID_ALARM'},inplace=True)
    df, df_details = reset_alarms_id(df, df_details, 'train')
    df, df_details = check_if_already_open(df, df_details)
    df, df_linking_runs =  build_df_linking_runs(df, df_details, 'train')
    df['fkID_ALARM'] = df['DESCRIZIONE_ALLARME']
    df.drop(columns=['DESCRIZIONE_ALLARME'], inplace=True)
    df_details['fkID_ALARM'] = df_details['DESCRIZIONE_ALLARME']
    df_details.drop(columns=['DESCRIZIONE_ALLARME'], inplace=True)
    if len(df) > 0:
        temp_df_path = './temp.csv'
        df.to_csv(temp_df_path, index=False)
    return df, df_details, df_linking_runs

def elaborate_pi_results(df_allarmi, alarms_input_list):
    df, df_details = aggregate_results(df_allarmi, alarms_input_list, 'pi')
    if len(df) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df[['ID_REPOSITORY', 'ID_DATALOGGER', 'VER_SW_OBU_SCMT', 'VER_SW_OBU_ERTMS', 'SIGLA_COMPLESSO']] = \
        df.apply(lambda r: get_run_config_values(df_details.loc[(df_details.ALARM_ID == r.ALARM_ID)
            &(df_details.NID_MACROAREA == r.NID_MACROAREA)
            &(df_details.NID_AREA == r.NID_AREA)
            &(df_details.NID_PI == r.NID_PI)].ID_RUN.to_list()[-1]), axis=1, result_type='expand')
    df.rename(columns={'ALARM_ID':'fkID_ALARM'}, inplace=True)
    df_details.rename(columns={'ALARM_ID':'fkID_ALARM'}, inplace=True)
    df, df_details = reset_alarms_id(df, df_details, 'pi')
    df.rename(columns={'ALARM_ID':'fkID_ALARM','NID_PI':'MATRICOLA_TRENO'}, inplace=True)
    df.drop(columns=['NID_MACROAREA', 'NID_AREA'], inplace=True)
    df['MATRICOLA_TRENO'] = df.MATRICOLA_TRENO.astype('string')
    df_details['MATRICOLA_TRENO'] = df_details.MATRICOLA_TRENO.astype('string')
    df, df_details = check_if_already_open(df, df_details)
    df, df_linking_runs =  build_df_linking_runs(df, df_details, 'pi')
    df['fkID_ALARM'] = df['DESCRIZIONE_ALLARME']
    df.drop(columns=['DESCRIZIONE_ALLARME'], inplace=True)
    df_details['fkID_ALARM'] = df_details['DESCRIZIONE_ALLARME']
    df_details.drop(columns=['DESCRIZIONE_ALLARME'], inplace=True)
    return df, df_details, df_linking_runs

def aggregate_results(df_allarmi, alarms_input_list, type):
    df = pd.concat(df_allarmi)
    if len(df) == 0:
        return pd.DataFrame(), pd.DataFrame()
    for l in alarms_input_list:
        for (id, det_df) in l:
            det_df['ALARM_ID'] = id
    df_details = pd.concat([det_df for l in alarms_input_list for (_, det_df) in l])
    for col in env.default_issues_table_values.keys():
        if env.default_issues_table_values[col] != 'CURRENT_TIMESTAMP':
            df[col] = env.default_issues_table_values[col]
        else:
            df[col] = datetime.now()
    df = build_details(df, type)
    return df, df_details

def get_run_config_values(idrun):
    with engine.connect() as dbConnection:
        df_run_summary = pd.read_sql(f"select id_repository, id_datalogger, fkid_scmt_config, fkversione_ssb_ertms from {env.db.tabella_runs_summary} where id='{idrun}'", dbConnection)
        if len(df_run_summary) == 0 :
            return (env.default_run_summary_values.id_repository,
                    env.default_run_summary_values.id_datalogger,
                    env.default_scmt_config_values.fkversione_ssb_scmt,
                    env.default_run_summary_values.fkversione_ssb_ertms,
                    env.default_scmt_config_values.sigla_complesso)
        else:
            df_scmt_config = pd.read_sql(f"select fkversione_ssb_scmt, sigla_complesso from {env.db.tabella_scmt_config} where fkid_scmt_config='{df_run_summary.fkid_scmt_config.loc[0]}'", dbConnection)
            if len(df_scmt_config) == 0:
                return (df_run_summary.loc[0, 'id_repository'],
                        df_run_summary.loc[0, 'id_datalogger'],
                        env.default_scmt_config_values.fkversione_ssb_scmt,
                        df_run_summary.loc[0, 'fkversione_ssb_ertms'],
                        env.default_scmt_config_values.sigla_complesso)
            else:
                return (df_run_summary.loc[0, 'id_repository'],
                        df_run_summary.loc[0, 'id_datalogger'],
                        df_scmt_config.loc[0, 'fkversione_ssb_scmt'],
                        df_run_summary.loc[0, 'fkversione_ssb_ertms'],
                        df_scmt_config.loc[0, 'sigla_complesso'])


def check_if_already_open(df, df_details):
    with engine.connect() as dbConnection:
        df_issue = pd.read_sql(f"select * from {env.db.tabella_issues} where state != 'CLOSED'", dbConnection)
        if len(df_issue) == 0:
            max_id = 0
        else:
            max_id = int(df_issue.id_issue.max()) + 1
    df[['ALREADY_EXIST', 'ID_ISSUE']] = df.apply(lambda r: already_open(r.MATRICOLA_TRENO, df_issue), axis=1, result_type='expand')
    temp_train_df_path = './temp.csv'
    if Path(temp_train_df_path).is_file():    
        temp_train_df = pd.read_csv(temp_train_df_path)
        temp_train_df_file_path = Path(temp_train_df_path)
        max_id = temp_train_df.ID_ISSUE.max() + 1
        temp_train_df_file_path.unlink()
    new_ids = len(df.loc[df.ALREADY_EXIST == False])
    df.loc[df.ALREADY_EXIST == False, 'ID_ISSUE'] = [e for e in range(max_id, max_id + new_ids)]
    df_details['ID_ISSUE'] = pd.merge(df[['MATRICOLA_TRENO', 'fkID_ALARM', 'ID_ISSUE']], df_details[['MATRICOLA_TRENO', 'fkID_ALARM']], \
            on=['MATRICOLA_TRENO', 'fkID_ALARM']).ID_ISSUE
    return df, df_details


def already_open(id, df_issue):
    if(len(df_issue.loc[df_issue.matricola_treno == str(id)]) > 0):
        return (True, df_issue.loc[df_issue.matricola_treno == str(id)].id_issue.to_list()[0])
    else:
        return (False, -1)


def build_details(dataframe, type):
    df = dataframe.copy(deep=True)
    if type == 'train':    
        df['DETAILS'] = dataframe.apply(lambda r: f'CABINA: {r.CABINA} FREQUENZA: {r.FREQUENZA} CARTELLA_DETTAGLI: {r.PATH_DATI_INPUT_COINVOLTI}', axis=1)
        df.drop(columns=['CABINA','FREQUENZA','PATH_DATI_INPUT_COINVOLTI','TMS_ALLARME'], inplace=True)
    else:
        df['DETAILS'] = dataframe.apply(lambda r: build_details_pi(r), axis=1)
        df.drop(columns=['PATH_DATI_INPUT_COINVOLTI','TMS_ALLARME'], inplace=True)
    return df


def build_details_pi(r):
    if r.DESCRIZIONE_ALLARME == env.alarms_types.multiple_pi_alarms_desc:
        df_catene = pd.read_csv('./conf/Catene.csv', sep=';')
        df_catene.fillna('', inplace=True)
        id_text = f'{r.NID_MACROAREA}-{r.NID_AREA}-{r.NID_PI}'
        df_catene['POSSIBILI_SUCCESSIVI'] = df_catene.POSSIBILI_SUCCESSIVI.apply(lambda s: s.split(','))
        return f'NID_MACROAREA: {r.NID_MACROAREA} NID_AREA: {r.NID_AREA} CARTELLA_DETTAGLI: {r.PATH_DATI_INPUT_COINVOLTI} PUNTI PARALLELI: {df_catene[df_catene.POSSIBILI_SUCCESSIVI.apply(lambda l: id_text in l)].POSSIBILI_SUCCESSIVI.to_list()[0]}'
    else:
        return f'NID_MACROAREA: {r.NID_MACROAREA} NID_AREA: {r.NID_AREA} CARTELLA_DETTAGLI: {r.PATH_DATI_INPUT_COINVOLTI}'


def reset_alarms_id(df, df_details, type):
    if type == 'train':
        new_df = df.groupby(by=['MATRICOLA_TRENO']).first()
        new_df.reset_index(inplace=True)
        new_df['NEW_ID'] = [i for i in range(len(new_df))]
        temp_df = pd.merge(df_details, new_df[['MATRICOLA_TRENO', 'NEW_ID', 'DESCRIZIONE_ALLARME']], on=['MATRICOLA_TRENO'])
        new_df['MATRICOLA_TRENO'] = new_df.MATRICOLA_TRENO.astype('string')
        temp_df['MATRICOLA_TRENO'] = temp_df.MATRICOLA_TRENO.astype('string')
    else:
        new_df = df.groupby(by=['NID_PI', 'NID_MACROAREA', 'NID_AREA']).first()
        new_df.reset_index(inplace=True)
        new_df['NEW_ID'] = [i for i in range(len(new_df))]
        temp_df = pd.merge(df_details, new_df[['NID_PI', 'NID_MACROAREA', 'NID_AREA', 'NEW_ID', 'DESCRIZIONE_ALLARME']], \
            on=['NID_PI', 'NID_MACROAREA', 'NID_AREA'])
    new_df['fkID_ALARM'] = new_df.NEW_ID
    temp_df['fkID_ALARM'] = temp_df.NEW_ID
    temp_df.drop(columns=['NEW_ID'], inplace=True)
    new_df.drop(columns=['NEW_ID'], inplace=True) 
    new_df['fkID_ALARM'] = new_df.fkID_ALARM.astype('int')
    temp_df['fkID_ALARM'] = temp_df.fkID_ALARM.astype('int')

    return new_df, temp_df


def build_df_linking_runs(df, df_details, type):
    if type == 'train':
        out_df = pd.merge(df_details[['MATRICOLA_TRENO', 'fkID_ALARM', 'ID_RUN', 'DISTANZA', 'VELOCITA', 'TIMESTAMP']],\
            df[['MATRICOLA_TRENO', 'fkID_ALARM', 'ID_ISSUE']], on=['MATRICOLA_TRENO', 'fkID_ALARM']).groupby(by=['fkID_ALARM']).first()
        out_df.drop(columns=['MATRICOLA_TRENO'], inplace=True)
    else :
        out_df = pd.merge(df_details[['NID_PI', 'fkID_ALARM', 'ID_RUN', 'DISTANZA', 'VELOCITA', 'TIMESTAMP']],\
        df[['MATRICOLA_TRENO', 'fkID_ALARM', 'ID_ISSUE']], left_on=['NID_PI', 'fkID_ALARM'], right_on=['MATRICOLA_TRENO', 'fkID_ALARM'])
        out_df.drop(columns=['NID_PI','MATRICOLA_TRENO', 'fkID_ALARM'], inplace=True)
    for col in env.default_issues_linking_table_values.keys():
        if env.default_issues_linking_table_values[col] != 'CURRENT_TIMESTAMP':
            out_df[col] = env.default_issues_linking_table_values[col]
        else:
            out_df[col] = datetime.now()
    out_df['TIMESTAMP'] = out_df.TIMESTAMP.apply(lambda d: int(d.timestamp()))
    out_df.DISTANZA.fillna(0, inplace=True)
    out_df.rename(columns={'TIMESTAMP' : 'TS_AT', 'DISTANZA' : 'DISTANCE_AT', 'VELOCITA' : 'SPEED_AT', 'ID_ISSUE' : 'fkID_ISSUE', 'ID_RUN' : 'fkID_RUN'}, inplace=True)
    new_df = df[df.ALREADY_EXIST == False]
    new_df.drop(columns=['ALREADY_EXIST'], inplace=True)
    return new_df, out_df