import pandas as pd
from datetime import datetime
from pathlib import Path
import importlib
import json
from multiprocessing import Pool
from attrdict import AttrDict
import sys

utils = importlib.import_module("utils")
data_preparation = importlib.import_module("data_preparation")
train_models_evaluation = importlib.import_module("train_models_evaluation")
pi_models_evaluation = importlib.import_module("pi_models_evaluation")
elaborate_models_results = importlib.import_module("elaborate_models_results")
save_models_results = importlib.import_module("save_models_results")

importlib.reload(utils)
importlib.reload(data_preparation)
importlib.reload(train_models_evaluation)
importlib.reload(pi_models_evaluation)
importlib.reload(elaborate_models_results)
importlib.reload(save_models_results)

#Loads environments settings
with open('./conf/env_setting.json') as env_setting:
    env = AttrDict(json.load(env_setting))

#PARAMETRI
#Loads checkpoints
with open('./conf/checkpoints.json') as c:
    last_checkpoints = AttrDict(json.load(c))
upper_limit = env.upper_limit

#CHECK INIZIALI
# PATH DEI FILE DI LOG
log_path = f'{env.path.logs_directory}{datetime.now().date()}/'
Path(log_path).mkdir(parents=True, exist_ok=True)
# PATH CARTELLA SALVATAGGIO DETTAGLI
Path(env.path.details_directory).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # Ottenimento dei dati
    starting_datetime = datetime.now()
    starting_time = str(starting_datetime).replace(" ","_").replace(":","").replace(".","")
    log_file_path = f'{log_path}log_{starting_time}'
    utils.log(log_file_path, f'Inizio processo di analisi: {starting_datetime}')
    utils.log(log_file_path, f'Parametri del processo:\n\tlast_checkpoints:{last_checkpoints}\n\tupper_limit:{upper_limit}')
    # La data preparation torna 2 liste:
    # 1. [per ogni treno una tupla (df_treno, zone_param_dict, last_checkpoints, upper_limit)]
    # 2. [per ogni pi una tupla (df_pi, max_airgap, min_airgap, mean_airgap, last_last_checkpoints, upper_limit)]
    df_treni_info_list, df_boe_info_list = data_preparation.get_clean_data(last_checkpoints, upper_limit, log_file_path)
    if (df_treni_info_list[0][0].empty)&(df_boe_info_list[0][0].empty):
        utils.log(log_file_path, "Non ci sono nuove captazioni da analizzare. Programma interrotto")
        utils.log(log_file_path, f'Analisi completata con orario: {datetime.now()}')
        sys.exit()
    utils.log(log_file_path, 'Treni analizzati durante questa sessione:\n' + \
          ', '.join([s.replace(' ', '') for s in sorted(list([t[0].MATRICOLA_TRENO.unique()[0] for t in df_treni_info_list]))]))
    utils.log(log_file_path, 'Punti informativi analizzati durante questa sessione:\n' + \
         ', '.join([s.replace(' ', '') for s in sorted(list([t[0]\
             .apply(lambda r: f'{r.NID_MACROAREA}_{r.NID_AREA}_{r.NID_PI}', axis=1).unique()[0] for t in df_boe_info_list]))]))
    with Pool(processes = 8) as pool:
        utils.log(log_file_path, f'Inizio analisi sui treni: {datetime.now()}')
        # train_alarms_df = dataframe con gli allarmi creati e train_alarms_input_df_list = lista di dataframe che contengono gli input che hanno scatenato gli allarmi
        train_alarms_df, train_alarms_input_df_list = [list(e) for e in list(zip(*pool.starmap(train_models_evaluation.get_model_results, df_treni_info_list)))]
        utils.log(log_file_path, f'Fine analisi sui treni: {datetime.now()}')
        utils.log(log_file_path, f'Inizio analisi sui punti informativi: {datetime.now()}')
        # pi_alarms_df = dataframe con gli allarmi creati e pi_alarms_input_df_list = lista di dataframe che contengono gli input che hanno scatenato gli allarmi
        pi_alarms_df, pi_alarms_input_df_list = [list(e) for e in list(zip(*pool.starmap(pi_models_evaluation.get_model_results, df_boe_info_list)))]
        utils.log(log_file_path, f'Fine analisi sui punti informativi: {datetime.now()}')
    # unione dei risultati dei treni e delle boe
    train_results, train_results_details, train_linking_runs = elaborate_models_results.elaborate_train_results(train_alarms_df, train_alarms_input_df_list)
    pi_results, pi_results_details, pi_linking_runs = elaborate_models_results.elaborate_pi_results(pi_alarms_df, pi_alarms_input_df_list)
    # Salvataggio dei risultati su DB e FTP
    if (train_results.empty)&(pi_results.empty):
        utils.log(log_file_path, f"L'analisi non ha prodotto alcun allarme da aggiungere.")
    elif (train_results.empty)&(not pi_results.empty):
        save_models_results.save_results(pi_results, pi_results_details, pi_linking_runs,
                                        f'RUN_{datetime.now().strftime("%Y%d%m%H%M%S")}')
    elif (not train_results.empty)&(pi_results.empty):
        save_models_results.save_results(train_results, train_results_details, train_linking_runs,
                                        f'RUN_{datetime.now().strftime("%Y%d%m%H%M%S")}')
    else:
        save_models_results.save_results(pd.concat([train_results, pi_results]),\
                                        pd.concat([train_results_details, pi_results_details]),
                                        pd.concat([train_linking_runs, pi_linking_runs]),
                                        f'RUN_{datetime.now().strftime("%Y%d%m%H%M%S")}')
    utils.log(log_file_path, f'Analisi completata con orario: {datetime.now()}')