import os

import pandas as pd
import pymongo
from pgmpy.readwrite import XMLBIFReader

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

    list_of_collections = mongo_client.list_collection_names()

    # Case 1: Exact match evaluated empirically
    if utils.get_mb_name(service, host) in list_of_collections:
        raw_samples = pd.DataFrame(list(mongo_client[utils.get_mb_name(service, host)].find()))
        samples = utils.prepare_samples(raw_samples)
        return utils.train_to_MB(samples)

    # Case 2.1: Comparable service evaluated at the target device [Metadata]
    # Idea: Should also check the online footprints, not only locally from the dummies
    # Idea: For this it would need a metadata clustering in some representation, e.g., text
    similar_services = utils.check_similar_services_same_host(host)
    service_mb = XMLBIFReader(f'../dummy/dummy_C_model.xml').get_model()
    if len(similar_services) > 0:
        for potential_host_mb in similar_services:

            # Only when no edges between blankets
            if utils.check_no_edges_with_service(potential_host_mb):
                # Just take the first, no comparison between them
                return utils.plug_in_service_variables(service_mb, potential_host_mb)

    # Case 2.2 Comparable service evaluated at the target device [Footprint]

    raise RuntimeError("Should not happen :(")


if __name__ == "__main__":
    extract_footprint("Processor", "Laptop")
    extract_footprint("Consumer_C", "Laptop")
