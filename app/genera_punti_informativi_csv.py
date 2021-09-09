import pandas as pd
from sqlalchemy import create_engine
import json
from attrdict import AttrDict

#Loads environments settings
with open('./conf/env_setting.json') as env_setting:
    env = AttrDict(json.load(env_setting))

engine = create_engine(f'postgresql://{env.db.username}:{env.db.password}@{env.db.server_name}:{env.db.port}/{env.db.name}')

# LEGGO ULTIMI N MESI DAL DB
with engine.connect() as dbConnection:
    df = pd.read_sql(f"select * from {env.db.tabella_captazioni}", dbConnection)

# ECCEZIONI - non sono mai lette dal DB (334 e 338 servono perchè sono le ultime degli id_run delle corse che finiscono a Potenza)
ecc = {
    (3,1,338):{'FREQUENZA':'F1', 'DIREZIONE':'PI_REVERSE'},
    (3,1,334):{'FREQUENZA':'F1', 'DIREZIONE':'PI_REVERSE'}
    }

df = df[['NID_MACROAREA', 'NID_AREA', 'NID_PI', 'FREQUENZA', 'DIREZIONE']]
df = df.groupby(by=['NID_MACROAREA', 'NID_AREA', 'NID_PI']).last()
for e in ecc.keys():
    if e not in df.index:
        df.loc[e] = ecc[e]
df.sort_index(inplace=True)
df.reset_index(inplace=True)
df.to_csv('./conf/Punti_informativi_new.csv', sep=';', index=False)