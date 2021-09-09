import pandas as pd
from attrdict import AttrDict
from sqlalchemy import create_engine
import json
from ftplib import FTP
from pathlib import Path

#Loads environments settings
with open('./conf/env_setting.json') as env_setting:
    env = AttrDict(json.load(env_setting))

engine = create_engine(f'postgresql://{env.db.username}:{env.db.password}@{env.db.server_name}:{env.db.port}/{env.db.name}')

def save_results(results_alarms, results_details, results_linking_runs, run_id):
    send_results_to_ftp_server(results_alarms, results_details, results_linking_runs, run_id)
    insert_results(results_alarms, results_details, results_linking_runs, run_id)
    return "done"


def send_results_to_ftp_server(results_alarms, results_details, results_linking_runs, run_id):
    temp_details_file = f'./{env.ftp.details_file_header}{run_id}.csv'
    temp_alarms_file = f'./{env.ftp.alarms_file_header}{run_id}.csv'
    temp_alarms_linking_runs_file = f'./{env.ftp.alarms_linking_runs_file_header}{run_id}.csv'
    results_details.to_csv(temp_details_file, index=False)
    results_alarms.to_csv(temp_alarms_file, index=False)
    results_linking_runs.to_csv(temp_alarms_linking_runs_file, index=False)
    details_file_path = Path(temp_details_file)
    alarms_file_path = Path(temp_alarms_file)
    alarms_linking_runs_file_path = Path(temp_alarms_linking_runs_file)
    with FTP(env.ftp.server, env.ftp.user, env.ftp.password) as ftp:
        with open(details_file_path, 'rb') as file:
            ftp.storbinary(f'STOR {env.ftp.details_file_path}{details_file_path.name}', file)
        with open(alarms_file_path, 'rb') as file:
            ftp.storbinary(f'STOR {env.ftp.alarms_file_path}{alarms_file_path.name}', file)
        with open(alarms_linking_runs_file_path, 'rb') as file:
            ftp.storbinary(f'STOR {env.ftp.alarms_linking_runs_file_path}{alarms_linking_runs_file_path.name}', file)
    details_file_path.unlink()
    alarms_file_path.unlink()
    alarms_linking_runs_file_path.unlink()
    return "done"

def insert_results(results_alarms, results_details, results_linking_runs, run_id):
    with engine.connect() as dbConnection:
        rename_dict = {c:c.lower() for c in results_alarms.columns}
        results_alarms.rename(columns=rename_dict, inplace=True)
        results_alarms.to_sql(env.db.tabella_issues, dbConnection, if_exists='append', index=False)
        rename_dict = {c:c.lower() for c in results_linking_runs.columns}
        results_linking_runs.rename(columns=rename_dict, inplace=True)
        results_linking_runs.drop(columns=['id'], inplace=True)
        results_linking_runs.to_sql(env.db.tabella_issues_linking_runs_summary, dbConnection, if_exists='append', index=False)
        results_details.to_csv(f'./{env.path.details_directory}{run_id}.csv', index=False)
    return "done"