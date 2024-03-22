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


@utils.print_execution_time
def retrieve_data():
    laptop = pd.DataFrame(list(mongo_client['Processor-Laptop'].find({}, batch_size=1)))


# Idea: possible parameters are batch size (what is it actually), caching yes/no --> freshness

retrieve_data()
retrieve_data()
retrieve_data()
retrieve_data()
retrieve_data()
