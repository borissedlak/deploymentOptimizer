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

sample_file = "samples.csv"


# footprint = Service-Host Deployment implications as MB
def extract_footprint(service, host):
    mongo_client = pymongo.MongoClient(MONGO_HOST)["metrics"]

    list_of_collections = mongo_client.list_collection_names()  # Return a list of collections in 'test_db'
    # model = None

    # Case 1: Exact match evaluated empirically
    if utils.get_mb_name(service, host) in list_of_collections:

        raw_samples = pd.DataFrame(list(mongo_client[utils.get_mb_name(service, host)].find()))
        samples = utils.prepare_samples(raw_samples)
        return utils.train_to_MB(samples)

    # Case 2.1: Comparable service evaluated at the target device [Metadata]
    elif 1 == 1:
        pass

    # Case 2.2 Comparable service evaluated at the target device [Footprint]


if __name__ == "__main__":
    extract_footprint("Processor", "Laptop")
    extract_footprint("Consumer", "Laptop")
