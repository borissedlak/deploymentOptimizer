import os

import pymongo

from detector import utils

MONGO_HOST = os.environ.get('MONGO_HOST')
if MONGO_HOST:
    print(f'Found ENV value for MONGO_HOST: {MONGO_HOST}')
else:
    MONGO_HOST = "localhost"
    print(f"Didn't find ENV value for MONGO_HOST, default to: {MONGO_HOST}")

sample_file = "samples.csv"


def extract_footprint(service, host):
    mongo_client = pymongo.MongoClient(MONGO_HOST)["metrics"]

    list_of_collections = mongo_client.list_collection_names()  # Return a list of collections in 'test_db'

    # Case 1: Exact match evaluated empirically
    if utils.get_mb_name(service, host) in list_of_collections:
        pass  # return MB of the composition
    # Case 2:
    elif 1 == 1:
        pass


if __name__ == "__main__":
    extract_footprint("Processor", "Laptop")
