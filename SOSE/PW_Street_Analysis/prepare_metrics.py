import os

import pandas as pd
import pymongo

from detector import utils

MONGO_HOST = os.environ.get('MONGO_HOST')
if MONGO_HOST:
    print(f'Found ENV value for MONGO_HOST: {MONGO_HOST}')
else:
    MONGO_HOST = "localhost"
    print(f"Didn't find ENV value for MONGO_HOST, default to: {MONGO_HOST}")

mongo_client = pymongo.MongoClient(MONGO_HOST)["metrics"]
orin = pd.DataFrame(list(mongo_client['Processor-Orin'].find()))

samples = utils.prepare_samples(orin, export_path="W_metrics_Analysis.csv")
